'''
convert a pandapower net to a .m file. 
Usages:
A sample base_net.m is needed for the OPF-DNN implementation
'''
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandapower as pp
from pandapower.converter import to_ppc
from src import util  



#defaule .m file conventions
BUS_COLS = 13
GEN_COLS = 21  # up to 'apf'
BR_COLS  = 13

def clean_ppc_for_matpower(ppc: dict) -> dict:
    # ---------- PPC cleaning (fix NaN/extra columns) ----------
    """Trim to standard MATPOWER cols, fill NaNs (esp. gen mBase), remove infs."""
    ppc2 = dict(ppc)  # shallow copy

    # bus
    bus = np.asarray(ppc["bus"], dtype=float)
    if bus.ndim == 1:
        bus = bus.reshape(1, -1)
    if bus.shape[1] > BUS_COLS:
        bus = bus[:, :BUS_COLS]
    bus = np.nan_to_num(bus, nan=0.0, posinf=0.0, neginf=0.0)

    # gen
    gen = np.asarray(ppc.get("gen", []), dtype=float)
    if gen.size:
        if gen.ndim == 1:
            gen = gen.reshape(1, -1)
        if gen.shape[1] > GEN_COLS:
            gen = gen[:, :GEN_COLS]
        # mBase column (0-based index 6)
        mbase_col = 6
        if gen.shape[1] > mbase_col:
            mask = np.isnan(gen[:, mbase_col])
            if np.any(mask):
                gen[mask, mbase_col] = float(ppc["baseMVA"])
        gen = np.nan_to_num(gen, nan=0.0, posinf=0.0, neginf=0.0)
        gen[np.isclose(gen, 0.0, atol=1e-12)] = 0.0
    else:
        gen = np.zeros((0, GEN_COLS), dtype=float)

    # branch
    branch = np.asarray(ppc.get("branch", []), dtype=float)
    if branch.size:
        if branch.ndim == 1:
            branch = branch.reshape(1, -1)
        if branch.shape[1] > BR_COLS:
            branch = branch[:, :BR_COLS]
        branch = np.nan_to_num(branch, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        branch = np.zeros((0, BR_COLS), dtype=float)

    # gencost (variable width; just de-NaN)
    if "gencost" in ppc:
        gencost = np.asarray(ppc["gencost"], dtype=float)
        if gencost.size:
            gencost = np.nan_to_num(gencost, nan=0.0, posinf=0.0, neginf=0.0)
        ppc2["gencost"] = gencost

    ppc2["bus"] = bus
    ppc2["gen"] = gen
    ppc2["branch"] = branch
    return ppc2

# ---------- helpers ----------

def _maybe_shift_bus_numbers_inplace(ppc: dict) -> bool:
    """Shift bus/gen/branch indices +1 if buses start at 0 (common in PPC)."""
    bus = np.asarray(ppc["bus"])
    if bus.size == 0:
        return False
    if np.min(bus[:, 0]) < 1:
        bus[:, 0] += 1
        if "gen" in ppc and len(ppc["gen"]):
            g = np.asarray(ppc["gen"])
            g[:, 0] += 1
        if "branch" in ppc and len(ppc["branch"]):
            br = np.asarray(ppc["branch"])
            br[:, 0] += 1  # fbus
            br[:, 1] += 1  # tbus
        return True
    return False

def _fmt_matrix(A: np.ndarray, float_fmt: str = "{:.5f}") -> str:
    """Format a numeric matrix like NESTA (ints as ints; floats with 5 d.p.)."""
    A = np.asarray(A)
    lines = []
    for row in A:
        vals = []
        for x in np.asarray(row).ravel():
            xf = float(x)
            if xf.is_integer():
                vals.append(f"{int(xf)}")
            else:
                vals.append(float_fmt.format(xf))
        lines.append("\t" + "\t".join(vals) + ";")
    return "\n".join(lines)

def _nesta_header(meta: Optional[Dict[str, str]] = None) -> str:
    """Return a NESTA-like header banner + provenance comments."""
    meta = meta or {}
    title  = meta.get("title", "NICTA Energy System Test Case Archive (NESTA) - v0.7.0")
    theme  = meta.get("theme", "Optimal Power Flow - Typical Operation")
    date_s = meta.get("date", datetime.now().strftime("%d - %B - %Y"))
    blurb  = meta.get("blurb", "")
    case_desc = meta.get("case_desc", "")
    src_url   = meta.get("source_url", "")
    lines = [
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
        "%%%%                                                                  %%%%%",
        f"%%%%      {title:<58}%%%%",
        f"%%%%              {theme:<42}%%%%",
        f"%%%%                         {date_s:<29}%%%%",
        "%%%%                                                                  %%%%%",
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
        "%",
    ]
    if blurb:
        lines += ["%  " + blurb, "%"]
    if case_desc:
        for l in case_desc.splitlines():
            lines.append("%  " + l)
        lines.append("%")
    if src_url:
        lines += ["%  Source:", f"%      {src_url}", "%"]
    return "\n".join(lines)

# ---------- enforce correct L/T encoding in PPC ----------

def _count_branch_split(ppc: dict):
    """Return (n_lines, n_trafos, total) as they will appear in the .m."""
    br = np.asarray(ppc.get("branch", []), dtype=float)
    if br.size == 0:
        return 0, 0, 0
    ratio = br[:, 8]   # MATPOWER ratio column
    angle = br[:, 9]   # MATPOWER angle column (degrees)
    # After our encoding, lines have ratio==0 & angle==0; trafos have ratio!=0 or angle!=0
    is_trafo = (np.abs(ratio) > 0.0) | (np.abs(angle) > 0.0)
    nT = int(is_trafo.sum())
    n = br.shape[0]
    return n - nT, nT, n

def _encode_branch_types_in_ppc(ppc: dict, net) -> None:
    """
    Use pandapower's trafo endpoints to encode type in PPC:
      - for each trafo (hv_bus, lv_bus): ensure ratio != 0 (set to 1.0 if unity).
      - for every other branch row: set ratio = 0.0 and angle = 0.0 (line).
    This makes the L/T split unambiguous for MATPOWER/PowerModels.
    """
    br = ppc.get("branch")
    if br is None or len(br) == 0:
        return

    br = np.asarray(br, dtype=float)

    # Collect trafo endpoint pairs from pandapower (handle both 0- and 1-based PPC)
    if not len(getattr(net, "trafo", [])):
        # No trafos in the pp net; mark everything as lines
        br[:, 8] = 0.0
        br[:, 9] = 0.0
        ppc["branch"] = br
        return

    hv = np.asarray(net.trafo["hv_bus"], dtype=int)
    lv = np.asarray(net.trafo["lv_bus"], dtype=int)
    Tpairs0 = {tuple(sorted((int(h), int(l)))) for h, l in zip(hv, lv)}
    Tpairs1 = {tuple(sorted((int(h)+1, int(l)+1))) for h, l in zip(hv, lv)}

    fbus = br[:, 0].astype(int)
    tbus = br[:, 1].astype(int)

    ratio_col, angle_col = 8, 9
    fixed_t = 0
    fixed_l = 0
    for i, (f, t) in enumerate(zip(fbus, tbus)):
        p = tuple(sorted((int(f), int(t))))
        is_trafo = (p in Tpairs0) or (p in Tpairs1)
        if is_trafo:
            # If PPC currently encodes unity as line (ratio==0 & angle==0), make it explicit
            if br[i, ratio_col] == 0.0 and br[i, angle_col] == 0.0:
                br[i, ratio_col] = 1.0
                fixed_t += 1
        else:
            # Force lines to the canonical MATPOWER encoding
            if br[i, ratio_col] != 0.0 or br[i, angle_col] != 0.0:
                br[i, ratio_col] = 0.0
                br[i, angle_col] = 0.0
                fixed_l += 1

    ppc["branch"] = br
    nL, nT, n = _count_branch_split(ppc)
    print(f"[nesta_writer] L/T encoding: L={nL}, T={nT}, total={n} (lines fixed={fixed_l}, trafos fixed={fixed_t})")

# ---------- writer ----------

def _write_nesta_style_m(ppc: dict, out_path: Path, func_name: str, meta: Optional[Dict[str, str]] = None) -> None:
    """Write a real MATPOWER .m with NESTA-like layout/comments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean + conventions
    ppc = clean_ppc_for_matpower(ppc)
    _maybe_shift_bus_numbers_inplace(ppc)

    header = _nesta_header(meta)

    with out_path.open("w", encoding="utf-8") as f:
        # Banner + function
        f.write(header + "\n")
        f.write(f"function mpc = {func_name}\n")
        f.write("mpc.version = '2';\n")
        f.write(f"mpc.baseMVA = {float(ppc['baseMVA']):.1f};\n\n")

        # BUS
        f.write("%% bus data\n")
        f.write("%\tbus_i\ttype\tPd\tQd\tGs\tBs\tarea\tVm\tVa\tbaseKV\tzone\tVmax\tVmin\n")
        f.write("mpc.bus = [\n")
        f.write(_fmt_matrix(ppc["bus"]))
        f.write("\n];\n\n")

        # GEN
        if "gen" in ppc and len(ppc["gen"]):
            f.write("%% generator data\n")
            f.write("%\tbus\tPg\tQg\tQmax\tQmin\tVg\tmBase\tstatus\tPmax\tPmin\tPc1\tPc2\tQc1min\tQc1max\tQc2min\tQc2max\tramp_agc\tramp_10\tramp_30\tramp_q\tapf\n")
            f.write("mpc.gen = [\n")
            f.write(_fmt_matrix(ppc["gen"]))
            f.write("\n];\n\n")

        # GEN COST
        if "gencost" in ppc and len(ppc["gencost"]):
            f.write("%% generator cost data\n")
            f.write("%\t2\tstartup\tshutdown\tn\tc(n-1)\t...\tc0\n")
            f.write("mpc.gencost = [\n")
            f.write(_fmt_matrix(ppc["gencost"]))
            f.write("\n];\n\n")

        # BRANCH
        if "branch" in ppc and len(ppc["branch"]):
            f.write("%% branch data\n")
            f.write("%\tfbus\ttbus\tr\tx\tb\trateA\trateB\trateC\tratio\tangle\tstatus\tangmin\tangmax\n")
            f.write("mpc.branch = [\n")
            f.write(_fmt_matrix(ppc["branch"]))
            f.write("\n];\n\n")

        # Optional extras if present
        if "areas" in ppc and len(ppc["areas"]):
            f.write("%% areas\nmpc.areas = [\n")
            f.write(_fmt_matrix(ppc["areas"]))
            f.write("\n];\n\n")
        if "bus_name" in ppc and len(ppc["bus_name"]):
            f.write("%% bus_name\nmpc.bus_name = {\n")
            for s in ppc["bus_name"]:
                f.write(f"    '{str(s)}';\n")
            f.write("};\n\n")

# ---------- public APIs ----------

def pp_net_to_nesta_m(
    net: "pp.pandapowerNet",
    out_path: str | Path,
    func_name: Optional[str] = None,
    meta: Optional[Dict[str,str]] = None,
    try_runpp: bool = True,
) -> str:
    """Write a NESTA-style .m from any pandapower net (any case size)."""
    outp = Path(out_path)
    if func_name is None or not str(func_name).strip():
        func_name = outp.stem

    # Make a PPC. If results are missing, either run a PF or use flat init.
    init = "results" if hasattr(net, "res_bus") and len(net.res_bus) else "flat"
    if init != "results" and try_runpp:
        try:
            pp.runpp(net, algorithm="nr", init="flat",
                     calculate_voltage_angles=True,
                     enforce_q_lims=True, check_connectivity=True)
            init = "results"
        except Exception:
            init = "flat"

    ppc = to_ppc(net, init=init)

    #  encode L/T explicitly using pandapower trafo endpoints
    _encode_branch_types_in_ppc(ppc, net)

    _write_nesta_style_m(ppc, outp, func_name, meta=meta)
    return str(outp.resolve())

def convert_base_net_to_nesta_m(
    case: str,
    which: str = "in_distribution",
    out_path: Optional[str | Path] = None,
    func_name: Optional[str] = None,
    meta: Optional[Dict[str,str]] = None,
) -> str:
    """Use util.get_base_net(case, which) and write a NESTA-style .m."""
    net, _ = util.get_base_net(case, which)
    if out_path is None:
        out_path = Path("generated_cases") / f"{case}_{which}.m"
    if func_name is None:
        func_name = f"{case}_{which}"
    banner_meta = {
        "title": meta.get("title") if meta else "NICTA Energy System Test Case Archive (NESTA) - v0.7.0",
        "theme": meta.get("theme") if meta else "Optimal Power Flow - Typical Operation",
        "date":  meta.get("date")  if meta else datetime.now().strftime("%d - %B - %Y"),
        "case_desc": meta.get("case_desc", "") if meta else "",
        "source_url": meta.get("source_url", "") if meta else "",
        "info_lines": meta.get("info_lines", []) if meta else [],
    }
    return pp_net_to_nesta_m(net, out_path, func_name, meta=banner_meta)

def derive_default_out_from_in(in_path: str, mroot: str | Path) -> Path:
    """
    src/scenarios/<case>/<which>/<name>.json.gz
      -> <mroot>/<case>/<which_no_unders>/<name_no_trailing_unders>.m
    Example:
      in:  src/scenarios/test/in_distribution/scen_000_.json.gz
      out: <mroot>/test/indistribution/scen_000.m
    """
    p = Path(in_path).resolve()
    parts = p.parts
    if "scenarios" not in parts:
        raise ValueError("Expected 'scenarios' in the path (e.g., src/scenarios/...)")
    i = parts.index("scenarios")
    rel = parts[i+1:]  # [case, which, filename]
    if len(rel) < 3:
        raise ValueError("Expected path like src/scenarios/<case>/<which>/<file>.json.gz")
    case  = rel[0]
    which = rel[1].replace("_", "")  # in_distribution -> indistribution
    fname = rel[-1]
    stem = fname[:-len(".json.gz")] if fname.endswith(".json.gz") else Path(fname).stem
    stem = stem.rstrip("_")  # scen_000_ -> scen_000
    return Path(mroot) / case / which / f"{stem}.m"

def convert_scenario_jsongz_to_nesta_m(
    in_path: str | Path,
    out_path: Optional[str | Path] = None,
    func_name: Optional[str] = None,
    meta: Optional[Dict[str,str]] = None,
    mroot: str | Path = None,
) -> str:
    """Load a .json.gz scenario via util.load_net and write NESTA-style .m."""
    net = util.load_net(str(in_path))
    if out_path is None:
        if mroot is None:
            mroot = Path("generated_cases")
        out_path = derive_default_out_from_in(str(in_path), mroot)
    outp = Path(out_path)
    if func_name is None:
        func_name = outp.stem
    return pp_net_to_nesta_m(net, outp, func_name, meta=meta)

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pandapower -> NESTA-style MATPOWER .m")
    ap.add_argument("--in_path", help="scenario *.json.gz (uses util.load_net); if omitted, uses util.get_base_net")
    ap.add_argument("--case", default="case14")
    ap.add_argument("--which", default="in_distribution")
    ap.add_argument("--name", default=None, help="function name inside the .m (defaults to file stem)")
    ap.add_argument("--out", default=None, help="explicit output .m path")
    ap.add_argument("--mroot", default=f"src/scenarios", help="root where <case>_mfiles/ lives (only for scenarios)")
    args = ap.parse_args()

    meta = {
        "title": "NICTA Energy System Test Case Archive (NESTA) - v0.7.0",
        "theme": "Optimal Power Flow - Typical Operation",
        "date": datetime.now().strftime("%d - %B - %Y"),
        "case_desc": f"Auto-generated from pandapower for {args.case}",
        "source_url": "",
        "info_lines": [],
    }

    if args.in_path:
        # place under  src/scenarios/<case>_mfiles/<case>/<which...>/<stem>.m
        mroot = Path(args.mroot) / f"{args.case}_mfiles"
        print(convert_scenario_jsongz_to_nesta_m(args.in_path, out_path=args.out,
                                                 func_name=args.name, meta=meta, mroot=mroot))
    else:
        out = args.out or Path("src/scenarios") / f"{args.case}_mfiles" / "sample_baseNet.m"
        print(convert_base_net_to_nesta_m(args.case, args.which, out_path=out,
                                          func_name=args.name, meta=meta))
