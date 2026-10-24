from julia.api import Julia
Julia(compiled_modules=False) #these 2 are required for IPOPT portion(else portion inside make_warmstart_run_opf)
import os
import sys
import pickle
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pandapower as pp
import time
import math

from src.global_evaluation_AC import evaluate_opf_baseline
from src.baselines.DeepOPF import vars
from src.baselines.DeepOPF.train_model import DeepOPFPenaltyNet
from src.baselines.DeepOPF.test_model import deepopf_denorm_outputs, deepopf_apply_prediction_to_net
import src.global_vars as gvars 

def make_warmstart_run_opf(X_all, Yhat_all, meta, device, use_ipopt= False, use_PF_warmstart= False):
    state = {"i": 0}

    def run_opf_warmstart(net):
        i = state["i"]
        if not use_PF_warmstart:   #change net with ML predictions
            orig_pq = net.get("_orig_pq_bus_idx", None)

            net2 = deepopf_apply_prediction_to_net(
                net, meta, X_all[i], Yhat_all[i], deepopf_denorm_outputs, device=device
            )
            net.clear()
            net.update(net2)

            if orig_pq is not None:
                net["_orig_pq_bus_idx"] = orig_pq

            for idx in net.gen.index:
                if net.gen.at[idx, "vm_pu"] > net.bus.at[net.gen.at[idx, "bus"], "max_vm_pu"]:
                    net.gen.at[idx, "vm_pu"] = net.bus.at[net.gen.at[idx, "bus"], "max_vm_pu"]
            
            for idx in net.gen.index:
                if net.gen.at[idx, "vm_pu"] < net.bus.at[net.gen.at[idx, "bus"], "min_vm_pu"]:
                    net.gen.at[idx, "vm_pu"] = net.bus.at[net.gen.at[idx, "bus"], "min_vm_pu"]

        if not use_ipopt: pp.runopp(net, init="pf")       #pandapower's default solver, warm-started by DeepOPF solutions

        else:                                           #use IPOPT
            pp.runpp(net, enforce_q_lims=True, calculate_voltage_angles=True)
            try:
                pp.runpm_ac_opf(net, pm_solver="ipopt", init="results")  #use the results form runpp to warm-start ipopt
            except Exception as e:
                print('didnt converge')
                print(e)
        
        state["i"] = i + 1

    return run_opf_warmstart



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--use_ipopt", action="store_true", help="False (default): use Pandapower's default solver (PDIPM) or True: use IPOPT?")
    parser.add_argument("--use_PF_ws", action="store_true", help="False (default): use ML warmstart or True: use pf warmstart")
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model_path = f"src/baselines/DeepOPF/model/{args.case}/trained_model.pth"
    
    
    which_case_list= list(gvars.stress_cases)
    which_case_list.append("in_distribution")
    for which_data in which_case_list:
        fname = f"src/baselines/DeepOPF/datasets/{args.case}/test_dataset_{which_data}"
        
        try:
            loaded = torch.load(fname + ".pt", map_location="cpu")
            print(f"Loaded Test file: {fname}")
        except FileNotFoundError:
            print(f"File {fname}.pt does not exist. Please generate this data before loading it.")
            sys.exit(1)
        dataset = loaded["dataset"]
        meta = loaded["meta"]
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

        X0, Y0 = next(iter(loader))
        in_dim, out_dim = int(X0.shape[1]), int(Y0.shape[1])

        model = DeepOPFPenaltyNet(in_dim=in_dim, hidden=vars.model_hidden, out_dim=out_dim).to(device)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        scen_dir = os.path.join("src", "scenarios", args.case, "test", which_data)
        manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
        manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
        paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]

        out_dir = os.path.join("results", args.case)
        os.makedirs(out_dir, exist_ok=True)

        X_all, Yhat_all= None , None 
        pred_time_per_sample_s= 0
        if not args.use_PF_ws:
            print("Collecting DeepOPF predictions")
            X_all, Yhat_all = [], []
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_pred0 = time.perf_counter()
            n_pred_samples = 0
            with torch.inference_mode():
                for Xb, _ in loader:
                    n_pred_samples += Xb.shape[0]
                    Xb = Xb.to(device)
                    Yhat = model(Xb)
                    X_all.append(Xb.detach().cpu())
                    Yhat_all.append(Yhat.detach().cpu())
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_pred1 = time.perf_counter()

            pred_time_total_s = t_pred1 - t_pred0
            pred_time_per_sample_s = pred_time_total_s / max(1, n_pred_samples)

            X_all = torch.cat(X_all, dim=0)
            Yhat_all = torch.cat(Yhat_all, dim=0)

        run_opf_warmstart = make_warmstart_run_opf(X_all, Yhat_all, meta, str(device), 
                                                   use_ipopt=args.use_ipopt, use_PF_warmstart=args.use_PF_ws)

        print("Running with warm-start evaluation")
        ipopt_ws_res = evaluate_opf_baseline(paths, run_opf_fn=run_opf_warmstart, verbose=False)
        
        if not args.use_PF_ws:
            #add deepopf prediction time with the time in the result dict
            ipopt_ws_res["deepopf_pred_time_total_s"] = float(pred_time_total_s)
            ipopt_ws_res["deepopf_pred_time_per_sample_s"] = float(pred_time_per_sample_s)

        opf_part = ipopt_ws_res.get("infer_time_per_sample_s", float("nan"))
        if opf_part is None:
            opf_part = float("nan")

        # end-to-end: DeepOPF prediction + warmstart OPF runtime
        if math.isfinite(opf_part):
            ipopt_ws_res["infer_time_per_sample_s"] = float(opf_part + pred_time_per_sample_s)
        else:
            # if OPF timing is nan, still store what we have
            ipopt_ws_res["infer_time_per_sample_s"] = float("nan")
            ipopt_ws_res["infer_time_per_sample_s_note"] = "OPF time was NaN"

        fnameStart= "pdipm"
        if args.use_ipopt: fnameStart= "ipopt"
        ML_ws= 'ML'
        if args.use_PF_ws: ML_ws= 'PF'
        ipopt_ws_out = os.path.join(out_dir, f"{fnameStart}Warmstart{ML_ws}_{which_data}.pkl")
        with open(ipopt_ws_out, "wb") as f:
            pickle.dump(ipopt_ws_res, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
