import copy, argparse, pickle 
import numpy as np, pandas as pd 
import torch
import sys, os 
from torch.utils.data import DataLoader
from src import util, view, global_vars as gvars
from src.global_evaluation_AC import PFEvaluator
from . import vars
from .make_dataset import build_deepopf_test_data_loader
from .train_model import prep_constraints, DeepOPFPenaltyNet


# -----------------------------
# DeepOPF-specific adapters
# -----------------------------
def deepopf_denorm_outputs(Y_norm, meta, device="cpu"):
    
    PG_lo = torch.as_tensor(meta["PG_lo"], device=device, dtype=Y_norm.dtype).view(1, -1)
    PG_hi = torch.as_tensor(meta["PG_hi"], device=device, dtype=Y_norm.dtype).view(1, -1)
    Vm_lo = torch.as_tensor(meta["Vm_lo"], device=device, dtype=Y_norm.dtype).view(1, -1)
    Vm_hi = torch.as_tensor(meta["Vm_hi"], device=device, dtype=Y_norm.dtype).view(1, -1)
    Slack_lo = torch.as_tensor(meta["Slack_lo"], device=device, dtype=Y_norm.dtype).view(1, 1)
    Slack_hi = torch.as_tensor(meta["Slack_hi"], device=device, dtype=Y_norm.dtype).view(1, 1)

    n_pg = PG_lo.shape[1]
    n_vm = Vm_lo.shape[1]
    PG_n= Y_norm[:, :n_pg]
    Slack_n= Y_norm[:, n_pg:n_pg+1]
    Vm_n = Y_norm[:, n_pg+1:n_pg+1+n_vm]

    PG = PG_lo + (PG_hi - PG_lo) * PG_n
    SlackV = Slack_lo + (Slack_hi - Slack_lo) * Slack_n
    Vm = Vm_lo + (Vm_hi - Vm_lo) * Vm_n
    return PG, SlackV, Vm


def deepopf_apply_prediction_to_net(base_net, meta, x_norm_i, y_norm_i, denorm_fn, device="cpu"):
    # maps
    def build_maps(net, meta):
        load_bus_to_idxs = {}
        if len(net.load):
            buses= net.load["bus"].to_numpy()
            idxs = net.load.index.to_numpy()
            for b in net.bus.index.to_numpy():
                m = (buses == b)
                if m.any():
                    load_bus_to_idxs[int(b)] = idxs[m]
        pv_pos = {int(b): i for i, b in enumerate(meta["pv_bus_list"])}
        return load_bus_to_idxs, pv_pos

    net = copy.deepcopy(base_net)
    load_bus_to_idxs, pv_pos = build_maps(net, meta)

    # meta unpack
    P_mu, P_sig= meta["P_mu"], meta["P_sig"]
    Q_mu, Q_sig = meta["Q_mu"], meta["Q_sig"]
    active_gen_idx = meta["active_gen_idx"]
    slack_bus = meta["slack_bus"]

    # outputs (1-sample)
    y1 = y_norm_i.detach().to(device).view(1, -1) if hasattr(y_norm_i, "detach") else y_norm_i
    PG_t, SlackV_t, VmPV_t = denorm_fn(y1, meta, device=device)
    PG_np = PG_t.squeeze(0).detach().cpu().numpy()
    SlackV = float(SlackV_t.item())
    VmPV_np= VmPV_t.squeeze(0).detach().cpu().numpy()

    # inputs (loads)
    x = x_norm_i.detach().cpu().numpy() if hasattr(x_norm_i, "detach") else np.asarray(x_norm_i)
    n_bus = len(base_net.bus)
    Pz= x[0:n_bus]; Qz = x[n_bus:2*n_bus]
    P_D = P_mu + Pz * P_sig
    Q_D = Q_mu + Qz * Q_sig

    # set loads
    if len(net.load):
        for bus in net.bus.index:
            idxs = load_bus_to_idxs.get(int(bus))
            if not idxs is None and len(idxs):
                share_p = float(P_D[bus]) / len(idxs)
                share_q = float(Q_D[bus]) / len(idxs)
                net.load.loc[idxs, "p_mw"]= share_p
                net.load.loc[idxs, "q_mvar"] = share_q

    # set gen P
    for k, gi in enumerate(active_gen_idx):
        net.gen.at[gi, "p_mw"] = float(PG_np[k])

    # slack V
    if len(net.ext_grid):
        eg_idx = net.ext_grid.index[net.ext_grid["bus"] == slack_bus]
        if len(eg_idx):
            net.ext_grid.at[eg_idx[0], "vm_pu"] = SlackV

    # PV voltages
    for k, gi in enumerate(active_gen_idx):
        bus_k = int(net.gen.at[gi, "bus"])
        pos = pv_pos.get(bus_k)
        if pos is not None:
            net.gen.at[gi, "vm_pu"] = float(VmPV_np[pos])

    prep_constraints(net)
    return net


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action="store_true", default=False)
    parser.add_argument("--which_data", type=str, default='in_distribution',
                        help='Which stress case or normal data to generate?')
    parser.add_argument("--which_model", type=str, default='baseline',
                        help='baseline or augmented? Which one to test?')
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    if(args.which_model == 'baseline'):
        model_path= f"src/baselines/DeepOPF/model/{args.case}/trained_model.pth"
        print('Loaded BASELINE model')
    elif (args.which_model == 'augmented'): 
        model_path= f"src/baselines/DeepOPF/model/{args.case}/trained_model_augmented.pth"
        print('Loaded AUGMENTED model')
    else: 
        print ('Invalid model name; should be either baseline or augmented.')
        sys.exit(1)

    sample_net, _ = util.get_base_net(args.case, 'in_distribution')  # sample_nets are always in_dist
    numTotalBus = len(sample_net.bus)

    # test dataset
    fname = f"src/baselines/DeepOPF/datasets/{args.case}/test_dataset_{args.which_data}"
    if args.gen_data:
        loader, meta = build_deepopf_test_data_loader(
            numTotalBus, args.case, fname, args.which_data, batch_size=args.batch
        )
    else:
        try:
            loaded = torch.load(fname + '.pt', map_location="cpu")
            print(f'Loaded Test file: {fname}')
        except FileNotFoundError:
            print(f"File {fname}.pt does not exist. Please generate this data before loading it.")
            sys.exit(1)
        dataset = loaded["dataset"]
        meta = loaded["meta"]
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    # infer I/O sizes
    try:
        X0, Y0 = next(iter(loader))
        in_dim, out_dim = int(X0.shape[1]), int(Y0.shape[1])
    except StopIteration:
        raise RuntimeError("Empty test loader.")

    # DeepOPF model
    model = DeepOPFPenaltyNet(in_dim=in_dim, hidden=vars.model_hidden, out_dim=out_dim).to(device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    scen_dir = os.path.join("src", "scenarios", args.case, "test", args.which_data)
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]
    print("Running DeepOPF evaluation")
    evaluator = PFEvaluator(
        denorm_fn=deepopf_denorm_outputs,
        apply_pred_fn=deepopf_apply_prediction_to_net,
        feas_tol=gvars.feas_tol,
        max_penalty= gvars.max_pen,
        ctol= gvars.ctol,
        runpp_kwargs=dict(enforce_q_lims=True, calculate_voltage_angles=True),
    )
    deepopf_res = evaluator.evaluate(
        model= model,
        loader= loader,
        meta= meta,
        net_paths= paths,
        device= str(device),
        collect_norm_mse= False,
        verbose= False,
    )

    print(deepopf_res["feasibility_rate_%"])
    os.makedirs(os.path.join("results", args.case), exist_ok=True)
    if(args.which_model == 'baseline'): fname = os.path.join('results', args.case, f"DeepOPF_result_{args.which_data}.pkl")
    if(args.which_model == 'augmented'): fname = os.path.join('results', args.case, f"DeepOPF_augmented_result_{args.which_data}.pkl")
    
    with open(fname, "wb") as f:
        pickle.dump(deepopf_res, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved DeepOPF results to {fname}")

if __name__ == "__main__":
    main()
