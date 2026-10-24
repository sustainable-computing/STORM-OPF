'''
get ground truth results using an iterative solver
this file is for AC-OPF (NOT the Security constrained one)
'''
from julia.api import Julia
Julia(compiled_modules=False)   #these 2 are required for IPOPT portion(else portion inside run_opf)
import os
import pickle
import pandas as pd
import pandapower as pp
from src.global_evaluation_AC import evaluate_opf_baseline
import argparse 
import src.global_vars as gvars 

def run_opf(net, use_default= True):
    
    if use_default: pp.runopp(net)      #uses Pandapower's internal PDIPM solver,cold-start
    
    #Do IPOPT instead
    else:
        try:
            pp.runpm_ac_opf(net, pm_solver="ipopt")
        except Exception as e:
            print('IPOPT didnt converge')
            print(e)


def generate_groundTruth(case: str, which_data: str, use_ipopt= False):
    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    net_paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]

    run_opf_fn = lambda net: run_opf(net, use_default=(not use_ipopt))
    gt_res_full = evaluate_opf_baseline(net_paths, run_opf_fn=run_opf_fn, verbose=False)

    out_dir = os.path.join('results', case)
    os.makedirs(out_dir, exist_ok=True)
    fnameStart = "ipopt" if use_ipopt else "pdipm"
    fname = os.path.join(out_dir, f"{fnameStart}_result_{which_data}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(gt_res_full, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved GT IPOPT results to {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth AC-OPF (IPOPT) results.")
    parser.add_argument("--case", type=str, default= 'case14', help="Case name, e.g. 'case30', 'case118'.")
    parser.add_argument("--use_ipopt", action="store_true",  help="False: use Pandapower's default solver (PDIPM) or True: use IPOPT")
    
    args = parser.parse_args()
    which_case_list= list(gvars.stress_cases)
    which_case_list.append("in_distribution")
    for which_data in which_case_list:
        generate_groundTruth(args.case, which_data, use_ipopt=args.use_ipopt)


if __name__ == "__main__":
    main()
