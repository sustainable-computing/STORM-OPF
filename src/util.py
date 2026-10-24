'''
This file contains common methods that are used by multiple baselines.
'''
import numpy as np 
import pandapower as pp
import pandapower.networks as pn
import os, pickle, gzip 
from src import global_vars as gvars, stress_test_scenarios


def load_net(path):
    if path.endswith(".json.gz"):
        tmp = path[:-3]  # remove .gz
        with gzip.open(path, "rb") as f_in, open(tmp, "wb") as f_out:
            f_out.write(f_in.read())
        net = pp.from_json(tmp)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except FileNotFoundError:
            pass
        return net
    elif path.endswith(".json"):
        return pp.from_json(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown scenario file type: {path}")


def add_simple_costs(net):
    state = np.random.get_state()      # save global RNG state
    np.random.seed(50)
    
    # Clear all existing costs
    if "poly_cost" in net:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    num_gen= len(net.gen.index)
    c0 = np.random.uniform(0, 10, num_gen)
    c1 = np.random.uniform(5, 20, num_gen)
    c2 = np.random.uniform(0.001, 0.05, num_gen) 
    for i, gi in enumerate(net.gen.index):
        pp.create_poly_cost(net, gi, "gen", cp0_eur=c0[i], cp1_eur_per_mw=c1[i], cp2_eur_per_mw2=c2[i])
    
    '''
    for slack bus
    '''
    for ei in net.ext_grid.index:
        pp.create_poly_cost(net, ei, "ext_grid", cp0_eur=0, cp1_eur_per_mw=20, cp2_eur_per_mw2=0.005)

    #restore random state
    np.random.set_state(state)

def get_base_net(caseName: str, which_data: str= None, seedAdd=0, opf_solve_fn=None):
    """
    Returns a pandapower base net given a case name string.
    Example: "case14", "case9", "case118"
    """
    if not hasattr(pn, caseName):
        raise ValueError(f"{caseName} is not a valid pandapower test case")

    # dynamically call pn.caseXX()
    base_net = getattr(pn, caseName)()
    add_simple_costs(base_net)

    net_meta= None 
    if (which_data in gvars.stress_cases):
        #base_net would be a 'stressed_net' now
        seedAdd= 42 + seedAdd    #different seed for different scenarios
        base_net, net_meta= stress_test_scenarios.make_scenario(base_net, which_data, seed= seedAdd, solve_opf_fn= opf_solve_fn)
    

    #make sure gen setpoints are within the bus limits, so we don't get
    #the warning 'gen vm_pu > bus max_vm_pu for gens [2 3]. Setting bus limit for these gens.'
    for idx in base_net.gen.index:
        if base_net.gen.at[idx, "vm_pu"] > base_net.bus.at[base_net.gen.at[idx, "bus"], "max_vm_pu"]:
            base_net.gen.at[idx, "vm_pu"] = base_net.bus.at[base_net.gen.at[idx, "bus"], "max_vm_pu"]
    
    for idx in base_net.gen.index:
        if base_net.gen.at[idx, "vm_pu"] < base_net.bus.at[base_net.gen.at[idx, "bus"], "min_vm_pu"]:
            base_net.gen.at[idx, "vm_pu"] = base_net.bus.at[base_net.gen.at[idx, "bus"], "min_vm_pu"]
    
    for idx in base_net.ext_grid.index:     #set limit for slack bus, so it doesn't contribute infinitely!
        base_net.ext_grid.at[idx, "max_p_mw"] = 1000.0    
        base_net.ext_grid.at[idx, "min_p_mw"] = -1000.0
        base_net.ext_grid.at[idx, "max_q_mvar"] = 1000.0
        base_net.ext_grid.at[idx, "min_q_mvar"] = -1000.0
    
    
    return base_net, net_meta 