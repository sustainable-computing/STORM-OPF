import numpy as np
import matplotlib.pyplot as plt
import os, argparse
import pickle 
import matplotlib.patches as patches

stress_cases= ['load_extreme_high','load_extreme_low',
              'fuel_price_up','fuel_price_down',
              'geo_shift',
              'extreme_weather',
              'nk_contingencies',
              'reactive_scarcity',
              'islanded_operation']


def plot_costs(true_costs, obtained_costs, title="Costs"):
    true = np.asarray(true_costs, dtype=float)
    pred = np.asarray(obtained_costs, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("true_costs and obtained_costs must have the same length.")

    # keep only finite pairs
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    if true.size == 0:
        raise ValueError("No finite (true, obtained) pairs to plot.")

    x = np.arange(1, len(true) + 1)  #  index starting at 1

    plt.figure(figsize=(10, 4))
    plt.plot(x, true, label="True cost", linewidth=1.5)
    plt.plot(x, pred, label="Obtained cost", linewidth=1.2)
    plt.xlabel("Data Point")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


#load and show results dicts
def load_and_print_result_dicts():
    base_dir= 'results'
    dict_avg_time_count={
        "pdipm": {"time": 0, "count": 0},
        "DeepOPF": {"time": 0, "count": 0},
        "PINN": {"time": 0, "count": 0},
        "PINN_augmented": {"time": 0, "count": 0},
        "OPFDNN": {"time": 0, "count": 0},
        "OPFDNN_augmented": {"time": 0, "count": 0},
        "DeepOPF_augmented": {"time": 0, "count": 0},
        "DeepOPF_SCDCOPF_noproject_baseline": {"time": 0, "count": 0},
        "DeepOPF_SCDCOPF_project_baseline": {"time": 0, "count": 0},
        "pcscopf": {"time": 0, "count": 0},
        "iterative_dcscopf": {"time": 0, "count": 0},
        "cscopf": {"time": 0, "count": 0},
        "DeepOPF_ext_SCACOPF_augmented": {"time": 0, "count": 0},
        "DeepOPF_ext_SCACOPF": {"time": 0, "count": 0},
    }
    for case_name in os.listdir(base_dir):
        case_path= os.path.join(base_dir, case_name)
        print('\n'+ '='*10 + f'CASE: {case_name}' + '='*10)
        if not os.path.isdir(case_path):
            continue
        sorted_files= np.sort(os.listdir(case_path))
        for fname in sorted_files:
            baselineName = fname.rsplit("_result", 1)[0]  #everything before _result
            if fname.endswith(".pkl"):
                file_path = os.path.join(case_path, fname)

                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                costs = np.asarray(data["costs"], dtype=float)
                print(f"\n--- File: {fname} ---")
                #print(data.keys())
                flags = np.asarray(data["hard_feasible_flags"], dtype=bool)  # 1/0 -> True/False

                feas_costs = costs[flags]
                avg_cost_feasible = float(np.mean(feas_costs)) if feas_costs.size > 0 else float("nan")
            
                #if feas rate=0, avg_cost here would be nan as we only avg for feasible samples
                print(f"avg. cost: {avg_cost_feasible}, feas. rate: {data['feasibility_rate_%']}, infer time/sample: {data['infer_time_per_sample_s']}")
                 
                #dict_avg_time_count[baselineName]["time"] += data['infer_time_per_sample_s']
                #dict_avg_time_count[baselineName]["count"] += 1
        
                
def plot_scenario_cost_boxplots(
    base_dir="results",
    cases=("case14", "case118"),
    baselines=("pdipm", "DeepOPF", "PINN", "OPFDNN"),
    scenarios=None,
    feasible_only=True,
    save_dir=None,
    dots=True,
    dot_size=10,
    dot_alpha=0.45,
    dots_jitter=0.12,
    # --- dynamic y-axis controls (per case subplot) ---
    y_min=1.0,
    y_pad_frac=0.05,
    y_pad_abs=0.02,
    y_min_span=0.12,
):

    def _set_ticks(ax, ymax):
        span = ymax - y_min
        if span <= 0.6:
            step = 0.05
        elif span <= 1.5:
            step = 0.10
        elif span <= 3.0:
            step = 0.25
        else:
            top_int = int(np.ceil(ymax))
            ax.set_yticks(np.arange(1, top_int + 1, 1))
            return
        end_rounded = np.ceil(ymax / step) * step
        ticks = np.arange(y_min, end_rounded + 1e-9, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    scenarios_renaming = {
        "load_extreme_high": "high load",
        "load_extreme_low": "low load",
        "fuel_price_up": "fuel price up",
        "fuel_price_down": "fuel price down",
        "geo_shift": "geo shift",
        "extreme_weather": "extreme weather",
        "nk_contingencies": "N-k contingencies",
        "reactive_scarcity": "reactive scarcity",
        "islanded_operation": "islanded operation",
        "in_distribution": "in distribution",
    }

    baseline_renaming = {
        "ipopt": "IPOPT",
        "OPFDNN": "OPF-DNN",
        "PINN": "PINN",
        "DeepOPF": "DeepOPF",
        "pdipm": "PDIPM"
    }

    if scenarios is None:
        scenarios = stress_cases

    if save_dir is None:
        save_dir = os.path.join(base_dir, "boxplots")
    os.makedirs(save_dir, exist_ok=True)

    prefix_map = {
        "ipopt": "ipopt_result_",
        "DeepOPF": "DeepOPF_result_",
        "PINN": "PINN_result_",
        "OPFDNN": "OPFDNN_result_",
        "pdipm": "pdipm_result_"
    }

    plot_baselines = [b for b in baselines if (b != "ipopt" and b != "pdipm")]
    baseline_labels = [baseline_renaming[b] for b in plot_baselines]
    baseline_pos = {lbl: i + 1 for i, lbl in enumerate(baseline_labels)}  # 1..K
    K = len(baseline_labels)

    def _load_costs_flags(case_name, baseline, scenario):
        case_path = os.path.join(base_dir, case_name)
        if not os.path.isdir(case_path):
            return None, None

        fname = f"{prefix_map[baseline]}{scenario}.pkl"
        fpath = os.path.join(case_path, fname)
        if not os.path.exists(fpath):
            return None, None

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        costs = np.asarray(data.get("costs", []), dtype=float)
        flags = np.asarray(data.get("hard_feasible_flags", np.ones_like(costs)), dtype=bool)

        if flags.shape != costs.shape:
            m = min(len(costs), len(flags))
            costs, flags = costs[:m], flags[:m]

        return costs, flags

    def _dynamic_ylim(values):
        if not values:
            return y_min, y_min + max(y_min_span, 0.2)

        vmax = max(float(np.max(v)) for v in values)
        ymax = max(
            y_min + y_min_span,
            vmax * (1.0 + y_pad_frac),
            vmax + y_pad_abs,
        )
        return y_min, ymax

    for scenario in scenarios:
        ratios = {c: {lbl: np.array([], dtype=float) for lbl in baseline_labels} for c in cases}
        counts = {c: {lbl: 0 for lbl in baseline_labels} for c in cases}  

        for case_name in cases:
            ip_costs, ip_flags = _load_costs_flags(case_name, "pdipm", scenario)
            if ip_costs is None or len(ip_costs) == 0:
                continue

            for b in plot_baselines:
                b_costs, b_flags = _load_costs_flags(case_name, b, scenario)
                if b_costs is None or len(b_costs) == 0:
                    continue

                m = min(len(ip_costs), len(b_costs))
                if m == 0:
                    continue

                ip_c = ip_costs[:m]
                b_c = b_costs[:m]

                ip_f = ip_flags[:m] if ip_flags is not None else np.ones(m, dtype=bool)
                b_f = b_flags[:m] if b_flags is not None else np.ones(m, dtype=bool)

                keep = np.isfinite(ip_c) & np.isfinite(b_c) & (ip_c > 0)
                if feasible_only:
                    keep &= (ip_f & b_f)

                if not np.any(keep):
                    continue

                r = b_c[keep] / ip_c[keep]
                lbl = baseline_renaming[b]
                ratios[case_name][lbl] = r
                counts[case_name][lbl] = int(r.size)  

        n_cases = len(cases)
        fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 8), sharey=False)
        if n_cases == 1:
            axes = [axes]

        for ax, case_name in zip(axes, cases):
            # collect non-empty data and positions
            data_list, pos_list = [], []
            for lbl in baseline_labels:
                arr = ratios[case_name][lbl]
                if arr is not None and arr.size > 0:
                    data_list.append(arr)
                    pos_list.append(baseline_pos[lbl])

            # draw boxplots
            if data_list:
                ax.boxplot(
                    data_list,
                    positions=pos_list,
                    widths=0.55,
                    showfliers=True,
                    patch_artist=False,
                    manage_ticks=False,
                )

                if dots:
                    rng = np.random.default_rng(0)
                    for arr, x0 in zip(data_list, pos_list):
                        x = x0 + rng.uniform(-dots_jitter, dots_jitter, size=arr.size)
                        ax.scatter(x, arr, s=dot_size, alpha=dot_alpha)

            # x labels 
            ax.set_xlim(0.5, K + 0.5)
            ax.set_xticks(range(1, K + 1))
            ax.set_xticklabels(baseline_labels, rotation=30, ha="right", fontsize=30)

            ax.axhline(1.0, linestyle="--", linewidth=2)
            ax.set_title(case_name, fontsize=30)
            ax.grid(True, axis="y", alpha=0.3)

            ymin, ymax = _dynamic_ylim(data_list)
            ax.set_ylim(ymin, ymax)
            _set_ticks(ax, ymax)
            ax.tick_params(axis="y", labelsize=20)

            y_text = 0.97
            for i, lbl in enumerate(baseline_labels, start=1):
                n = counts[case_name][lbl]
                ax.text(
                    i, y_text, f"n={n}",
                    transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
                    ha="center", va="top",
                    fontsize=18,
                )

        axes[0].set_ylabel("Sub-optimality (cost / PDIPM cost)", fontsize=25)
        fig.suptitle(
            f"{scenarios_renaming.get(scenario, scenario)}",
            fontsize=32,
            y=1.02,
        )

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{scenario}_subopt_vs_pdipm_grouped.pdf")
        top_margin = 0.04   # % of figure height reserved for title
        left_margin = 0
        bottom_margin = 0
        right_margin = 0

        fig.add_artist(
            patches.Rectangle(
                (left_margin, bottom_margin),
                1 - left_margin - right_margin,
                1 - bottom_margin - top_margin,
                transform=fig.transFigure,
                fill=False,
                linewidth=2,
                edgecolor="black",
                zorder=10,
            )
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

        print(f"[OK] Saved: {save_path}")

def plot_nominal_radar_all_baselines(base_dir="results", case_name="case14", save_dir='results/'):
    scenarios = stress_cases + ["in_distribution"]

    baselines = ["DeepOPF", "PINN", "OPFDNN", "pdipm"]

    prefix_map = {
        "DeepOPF": "DeepOPF_result_",
        "PINN": "PINN_result_",
        "OPFDNN": "OPFDNN_result_",
        "pdipm": "pdipm_result_",
    }

    baseline_renaming = {
        "DeepOPF": "DeepOPF",
        "PINN": "PINN",
        "OPFDNN": "OPF-DNN",
        "pdipm": "PDIPM",
    }

    summary = {}

    for baseline in baselines:
        feas_rates = []
        runtimes = []
        all_cost_ratios = []

        for scenario in scenarios:
            with open(os.path.join(base_dir, case_name, f"{prefix_map[baseline]}{scenario}.pkl"), "rb") as f:
                base = pickle.load(f)
            with open(os.path.join(base_dir, case_name, f"{prefix_map['pdipm']}{scenario}.pkl"), "rb") as f:
                pdipm = pickle.load(f)

            b_costs = np.asarray(base["costs"], dtype=float)
            p_costs = np.asarray(pdipm["costs"], dtype=float)
            b_flags = np.asarray(base["hard_feasible_flags"], dtype=bool)
            p_flags = np.asarray(pdipm["hard_feasible_flags"], dtype=bool)

            m = min(len(b_costs), len(p_costs), len(b_flags), len(p_flags))
            b_costs = b_costs[:m]
            p_costs = p_costs[:m]
            b_flags = b_flags[:m]
            p_flags = p_flags[:m]

            feas_rates.append(float(base["feasibility_rate_%"]))
            runtimes.append(float(base["infer_time_per_sample_s"]))

            keep = np.isfinite(b_costs) & np.isfinite(p_costs) & (p_costs > 0) & b_flags & p_flags
            if np.any(keep):
                all_cost_ratios.extend((b_costs[keep] / p_costs[keep]).tolist())

        avg_feas = float(np.mean(feas_rates))
        avg_runtime = float(np.mean(runtimes))

        if baseline == "pdipm":
            avg_cost_ratio = 1.0
        else:
            avg_cost_ratio = float(np.mean(all_cost_ratios)) if len(all_cost_ratios) else np.nan

        summary[baseline] = {
            "feas": avg_feas,
            "cost_ratio": avg_cost_ratio,
            "runtime": avg_runtime,
        }


    # feasibility: raw percentage -> [0,1]
    feas_scores = np.array([summary[b]["feas"] / 100.0 for b in baselines], dtype=float)

    # cost: lower ratio is better; PDIPM ratio=1 gets score 1
    cost_scores = []
    for b in baselines:
        r = summary[b]["cost_ratio"]
        if np.isfinite(r) and r > 0:
            cost_scores.append(min(1.0, 1.0 / r))
        else:
            cost_scores.append(0.0)
    cost_scores = np.asarray(cost_scores, dtype=float)

    # runtime: use log compression so NN differences remain visible
    runtime_vals = np.array([summary[b]["runtime"] for b in baselines], dtype=float)
    runtime_ref = np.min(runtime_vals[runtime_vals > 0])

    runtime_scores = []
    for t in runtime_vals:
        # fastest -> 1.0, slower methods decrease smoothly on log scale
        score = 1.0 / (1.0 + np.log10(t / runtime_ref))
        runtime_scores.append(score)
    runtime_scores = np.asarray(runtime_scores, dtype=float)

    # ---------- radar ----------
    labels = ["Feasibility", "Cost", "Runtime"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    r_inner = 0.20  # prevents collapse at the center

    fig, ax = plt.subplots(figsize=(6.5,6.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(r_inner, 1.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=25)

    #tick_scores = [0.25, 0.50, 0.75, 1.00]
    #tick_pos = [r_inner + t * (1.0 - r_inner) for t in tick_scores]
    ax.set_yticks([])
    ax.set_yticklabels([])
    #ax.set_yticklabels([f"{t:.2f}" for t in tick_scores], fontsize=10)
    ax.set_rlabel_position(90)

    for i, baseline in enumerate(baselines):
        scores = np.array(
            [feas_scores[i], cost_scores[i], runtime_scores[i]],
            dtype=float
        )
        radii = r_inner + scores * (1.0 - r_inner)
        radii_closed = np.r_[radii, radii[0]]

        ax.plot(angles_closed, radii_closed, linewidth=2, label=baseline_renaming[baseline])
        #ax.fill(angles_closed, radii_closed, alpha=0.08)

    #ax.set_title(case_name, y=1.10, fontsize=25)
    ax.legend( bbox_to_anchor=(1.4, 1.3), frameon=True, fontsize=13, ncol=1)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{case_name}_nominal_radar.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")

    plt.show()

    print(f"\n===== {case_name} =====")
    for baseline in baselines:
        print(
            f"{baseline_renaming[baseline]:8s} | "
            f"feas = {summary[baseline]['feas']:.2f} | "
            f"cost_ratio = {summary[baseline]['cost_ratio']:.4f} | "
            f"runtime = {summary[baseline]['runtime']:.8e}"
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default="load_result", help='what to do? Options: load_result, cost_boxplot, radar_chart')
    args = parser.parse_args()

    if args.todo== 'radar_chart':
        for case_name in ["case14", "case118"]:
            plot_nominal_radar_all_baselines(base_dir="results", case_name=case_name)         


    elif args.todo== 'load_result':
        load_and_print_result_dicts()
    
    
    elif args.todo== 'cost_boxplot':
    # Create one plot per scenario 
        plot_scenario_cost_boxplots(
            base_dir="results",
            cases=("case14", "case118"),
            baselines=("pdipm", "DeepOPF", "PINN", "OPFDNN"),
            scenarios=[
                "load_extreme_high",
                "load_extreme_low",
                "fuel_price_up",
                "fuel_price_down",
                "geo_shift",
                "extreme_weather",
                "nk_contingencies",
                "reactive_scarcity",
                "islanded_operation",
                "in_distribution"
            ],
            feasible_only=True,
        )