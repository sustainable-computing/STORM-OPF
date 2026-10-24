"""
Restoration operators for the gap-2 study, applied to a net that ALREADY has a
model prediction written into it (generator setpoints + voltages).

Each operator returns: (feasible, n_pf, projected, vm_corrected)

  none    PF-verify only — the predictor's raw output (constraint-baked baseline).
  shield  constraint-CONDITIONED restoration: project generator P onto the
          scenario's observed limits (redistributing the residual to headroom),
          then PF-verify and nudge voltage setpoints if PQ-bus voltages violate.

The projection/voltage helpers are reused verbatim from the ext_input_shield
package so there is one implementation of the operator across baselines.

Slots for the remaining gap-2 operators (constraint-baked: FRMNet-style learned
repair, fixed gauge/RAYEN maps; constraint-conditioned: DC/L1-QP projection,
DC3-style gradient correction) plug into RESTORERS with the same signature.
FSNet/DC3 *end-to-end* feasibility methods are separate trained models, not
post-hoc operators, so they live in their own train variant rather than here.
"""
import numpy as np
import pandapower as pp

from src.global_evaluation_AC import is_feasible_by_hard_constraints
from src.baselines.ext_input_shield.shield import (
    RUNPP_KWARGS,
    _pq_vm_violation,
    _shift_vm_setpoints,
    actual_p_limits,
    envelope_project,
)


def _runpp(net):
    pp.runpp(net, **RUNPP_KWARGS)
    return bool(getattr(net, "converged", False))


def restore_none(net, meta, pq_bus_idx, **kw):
    """No restoration: just run PF and verify the raw prediction."""
    feasible = False
    n_pf = 0
    try:
        if _runpp(net):
            n_pf = 1
            feasible = is_feasible_by_hard_constraints(net, vm_bus_idx=pq_bus_idx)
    except Exception:
        pass
    return feasible, n_pf, False, False


def restore_shield(net, meta, pq_bus_idx, max_vm_iters=3, vm_step=0.01, **kw):
    """Constraint-conditioned restoration (envelope projection + vm correction)."""
    gen_idx = meta["active_gen_idx"]

    # ---- stage 1: project generator P onto the observed box ----
    lo, hi = actual_p_limits(net, gen_idx)
    pg_raw = np.array(
        [float(net.gen.at[gi, "p_mw"]) for gi in gen_idx], dtype=float
    )
    pg_proj = envelope_project(pg_raw, lo, hi)
    projected = bool(np.max(np.abs(pg_proj - pg_raw), initial=0.0) > 1e-9)
    for k, gi in enumerate(gen_idx):
        net.gen.at[gi, "p_mw"] = float(pg_proj[k])

    # ---- stage 2: PF-verify, correct voltages if needed ----
    n_pf = 0
    vm_corrected = False
    feasible = False
    for it in range(1 + max_vm_iters):
        try:
            if not _runpp(net):
                break
        except Exception:
            break
        n_pf += 1
        if is_feasible_by_hard_constraints(net, vm_bus_idx=pq_bus_idx):
            feasible = True
            break
        viol = _pq_vm_violation(net, pq_bus_idx)
        if viol == 0.0 or it == max_vm_iters:
            break
        vm_corrected = True
        _shift_vm_setpoints(net, meta, -np.sign(viol) * max(vm_step, abs(viol)))

    return feasible, n_pf, projected, vm_corrected


RESTORERS = {
    "none": restore_none,
    "shield": restore_shield,
}
