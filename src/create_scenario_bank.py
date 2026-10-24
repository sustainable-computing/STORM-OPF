'''
Use this file to generate different scenarios (i.e. different pandapower nets)
and save them. The goal is to use the SAME DATASET for each baseline algorithm. 
To achieve this, we save training nets, in-distribution test nets, and edge-case nets.
Later, from each baseline, we load these pre-saved nets and build dataset as needed
by that particular baseline.
'''
import os, copy, time, json, gzip, pickle
import numpy as np
import pandas as pd
import pandapower as pp
import argparse
from src import util 
from src.baselines.PINN.generate_data import run_opf  

##====================##
#----    HELPERS    ----
##====================##
def save_net_json_gz(net, path_json_gz):
    """save the scenario"""
    tmp = path_json_gz.replace(".json.gz", ".json")
    pp.to_json(net, tmp)                 # writes JSON
    with open(tmp, "rb") as f_in, gzip.open(path_json_gz, "wb") as f_out:
        f_out.writelines(f_in)
    os.remove(tmp)

def sample_uniform_loads(net):
    """uniformly scale loads within [0.9, 1.1].
    """  
    rng = np.random.default_rng() 

    for idx in net.load.index:
        # Original values
        p0 = net.load.at[idx, "p_mw"]
        q0 = net.load.at[idx, "q_mvar"]

        # Sample scaling factor independently for each bus
        scale = rng.uniform(0.9, 1.1)

        # Apply perturbation
        net.load.at[idx, "p_mw"] = p0 * scale
        net.load.at[idx, "q_mvar"] = q0 * scale
    
    return net


def run_opf(net):
    # Prefer PowerModels (IPOPT) if available; else fallback to pandapower’s OPF
    try:
        pp.runpm_ac_opf(net, pm_solver="ipopt")
    except Exception:
        pp.runopp(net)
    return net 

def run_pf(net):
    """Plain power flow (for PF-only scenarios)."""
    pp.runpp(net)


##=======================##
#----    MAIN BLOCK    ----
##=======================##

def generate_scenarios(
    case="case14",
    purpose= 'test',        #which data to generate, training or test?
    which_data="in_distribution",
    out_dir="src/scenarios/",
    n_scenarios=1000,
    solve="opf",          # 'opf' or 'pf'. Use 'opf' for ground truth generation
    seed=0
):
    
    #complete the save-path name
    if (purpose == 'test' or purpose =='training_stressed'): out_dir= out_dir + case  + "/" + purpose + "/" + which_data 
    else: out_dir= out_dir + case  + "/" + purpose 

    os.makedirs(out_dir, exist_ok=True)

    # Manifest of scenarios
    rows = []
    opf_latencies = []
    e2e_latencies = []

    kept = 0
    tries = 0
    max_tries = n_scenarios * 2 


    while kept < n_scenarios and tries < max_tries:
        tries += 1
        t0 = time.perf_counter()
        base_net, _ = util.get_base_net(case, which_data, tries, opf_solve_fn=run_opf)
        net = copy.deepcopy(base_net)
        net = sample_uniform_loads(net)   #perturbation per-load row

        status = "ok"
        converged = False
        err_msg = ""
        t1 = time.perf_counter()
        try:
            if solve == "opf":
                run_opf(net)
                converged = bool(getattr(net, "OPF_converged", False))
            else:
                run_pf(net)
                converged = bool(getattr(net, "converged", False))
        except Exception as e:
            status = "error"
            err_msg = str(e)
        t2 = time.perf_counter()

        opf_latencies.append(t2 - t1)
        e2e_latencies.append(t2 - t0)

        if not converged:
            # keep the failure in manifest in case of later use, but don't save the net
            print('Not converged... retrying')
            rows.append({
                "id": kept,
                "file": "",
                "solve": solve,
                "status": "fail" if status == "ok" else status,
                "error": err_msg,
                "latency_s_solve": t2 - t1,
                "latency_s_e2e": t2 - t0,
            })
            continue

        # Save net 
        fname = f"sc_{kept:05d}.json.gz"
        fpath = os.path.join(out_dir, fname)
        try:
            save_net_json_gz(net, fpath)
            saved_as = "json.gz"
        except Exception as e:
            # fallback to pickle 
            saved_as = "pkl"
            fpath = os.path.join(out_dir, f"sc_{kept:05d}.pkl")
            with open(fpath, "wb") as f:
                pickle.dump(net, f)

        rows.append({
            "id": kept,
            "file": os.path.basename(fpath),
            "format": saved_as,
            "solve": solve,
            "status": "ok",
            "error": "",
            "latency_s_solve": t2 - t1,
            "latency_s_e2e": t2 - t0,
        })
        kept += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    meta = {
        "case": case, "which_data": which_data, "n": int(kept),
        "solve": solve, "seed": int(seed),
        "mean_latency_s_solve": float(np.mean(opf_latencies)) if opf_latencies else None,
        "mean_latency_s_e2e": float(np.mean(e2e_latencies)) if e2e_latencies else None,
        "tries": int(tries)
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {kept} scenarios to {out_dir} (tries={tries}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--purpose", type=str, default="test",  help='purpose: *training*, *training_stressed*, or *test* scneario generation?')
    parser.add_argument("--which_data", type=str, default='in_distribution', help='Which stress case or normal data to generate?')
    parser.add_argument("--num_scen", type=str, default=500)
    args = parser.parse_args()

    #diff seed for stress case for train or test
    allSeeds={'training':0, 'test': 0, 'training_stressed': 74}
    generate_scenarios(case= args.case, purpose=args.purpose, which_data=args.which_data, n_scenarios= int(args.num_scen), seed=allSeeds[args.purpose])
