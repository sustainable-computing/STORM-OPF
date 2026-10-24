"""
PINN OOD analysis utilities

Experiments:
1) PCA embedding of input space (train vs stress scenarios,
   optionally split by feasibility)

2) Distance-to-training vs feasibility (k-NN distance)

3) Simple scalar summaries (e.g., total load distributions)

PINN inputs are stored as CSVs:

  src/baselines/PINN/datasets/<num_bus_str>/NN_input_train_in_distribution.csv
  src/baselines/PINN/datasets/<num_bus_str>/NN_input_test_<which_data>.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ===========================
# Config
# ===========================

PINN_DATA_ROOT = os.path.join("src", "baselines", "PINN", "datasets")

TRAIN_FILE_NAME = "NN_input_train.csv"


# ===========================
# Loading helpers
# ===========================

def load_pinn_inputs(num_bus_str: str,
                     split: str = "train",
                     which_data: str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Load PINN input features X from CSV.

    Parameters
    ----------
    num_bus_str : str
        Subdirectory under src/baselines/PINN/datasets/.
        (e.g., "14", "118")
    split : {"train", "test"}
    which_data : str or None
        Stress scenario name for test data, e.g. "geo_shift".
        Ignored for split="train".

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    columns : list of str
        Column names from the CSV.
    """
    base = os.path.join(PINN_DATA_ROOT, num_bus_str)

    if split == "train":
        fX = os.path.join(base, TRAIN_FILE_NAME)
    else:
        if which_data is None:
            raise ValueError("which_data must be provided for split='test'")
        fX = os.path.join(base, f"NN_input_test_{which_data}.csv")

    if not os.path.exists(fX):
        raise FileNotFoundError(f"Missing PINN input CSV: {fX}")

    dfX = pd.read_csv(fX)
    X = dfX.to_numpy(dtype=float)
    return X, list(dfX.columns)


# ===========================
# 1) PCA embedding experiment
# ===========================

def run_pinn_pca_embedding(
    num_bus_str: str,
    stress_cases: list[str],
    feasibility_dict: dict[str, np.ndarray] | None = None,
    n_components: int = 2,
    max_points_per_set: int = 3000,
    random_state: int = 0,
):
    """
    PCA embedding of PINN inputs for:
      - train (in-distribution)
      - all stress_cases (test)

    Plots:
      - Grey cloud = train
      - For each stress case:
          feasible points (if feasibility_dict given) as filled markers
          infeasible points as 'x' markers
        If feasibility_dict is None, all points are treated as feasible.

    Parameters
    ----------
    num_bus_str : str
        Dataset folder (e.g., "case14" or "14bus").
    stress_cases : list of str
        Scenario names, e.g. ['geo_shift', 'extreme_weather', ...].
    feasibility_dict : dict[which_data -> np.ndarray of bool], optional
        Per-sample feasibility masks aligned with the raw CSV rows
        BEFORE any subsampling. If None, feasibility is ignored.
    n_components : int
        PCA components (2 for plotting).
    max_points_per_set : int
        Random subsample per dataset for readability.
    random_state : int
        RNG seed for subsampling and PCA.
    """
    sc_renamed={
        "in_distribution": "In distribution",
        "load_extreme_high": "High Load",
        "load_extreme_low": "Low Load",
        #"renewable_variability",
        "fuel_price_up": "Fuel price up",
        "fuel_price_down": "Fuel price down",
        "geo_shift": "Geo shift",
        "extreme_weather": "Extreme heat-wave",
        "nk_contingencies": "N-k contingencies",
        "reactive_scarcity": "Reactive scarcity",
        "islanded_operation": "Islanded operation"
    }
    rng = np.random.default_rng(random_state)

    # ---- Load training data ----
    X_train, _ = load_pinn_inputs(num_bus_str, split="train")
    if len(X_train) > max_points_per_set:
        idx = rng.choice(len(X_train), size=max_points_per_set, replace=False)
        X_train = X_train[idx]

    # ---- Load stress data ----
    scenario_X = {}
    scenario_feas_masks = {}

    for sc in stress_cases:
        X_sc, _ = load_pinn_inputs(num_bus_str, split="test", which_data=sc)

        feas_mask_full = None
        if feasibility_dict is not None and sc in feasibility_dict:
            feas_mask_full = np.asarray(feasibility_dict[sc], dtype=bool)
            if len(feas_mask_full) != len(X_sc):
                raise ValueError(
                    f"Feasibility mask length mismatch for '{sc}': "
                    f"{len(feas_mask_full)} vs {len(X_sc)}"
                )

        # subsample
        if len(X_sc) > max_points_per_set:
            idx = rng.choice(len(X_sc), size=max_points_per_set, replace=False)
            X_sc_sub = X_sc[idx]
            if feas_mask_full is not None:
                feas_mask_sub = feas_mask_full[idx]
            else:
                feas_mask_sub = np.ones(len(X_sc_sub), dtype=bool)
        else:
            X_sc_sub = X_sc
            if feas_mask_full is not None:
                feas_mask_sub = feas_mask_full
            else:
                feas_mask_sub = np.ones(len(X_sc_sub), dtype=bool)

        scenario_X[sc] = X_sc_sub
        scenario_feas_masks[sc] = feas_mask_sub

    # ---- Fit PCA on train only ----
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    X_train_2d = pca.transform(X_train)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))

    # Training cloud
    plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        s=8,
        alpha=0.3,
        label="Train",
    )

    # Stress scenarios
    for sc in stress_cases:
        X_sc = scenario_X[sc]
        feas_mask = scenario_feas_masks[sc]
        X_sc_2d = pca.transform(X_sc)

        mask_feas = feas_mask
        mask_infeas = ~feas_mask

        if np.any(mask_feas):
            plt.scatter(
                X_sc_2d[mask_feas, 0],
                X_sc_2d[mask_feas, 1],
                s=15,
                alpha=0.7,
                label=f"{sc_renamed[sc]}",
            )
        if np.any(mask_infeas):
            plt.scatter(
                X_sc_2d[mask_infeas, 0],
                X_sc_2d[mask_infeas, 1],
                s=15,
                alpha=0.7,
                marker="x",
                label=f"{sc}",
            )

    plt.xlabel("PC1", fontsize=18)
    plt.ylabel("PC2", fontsize=18)
    plt.title(f"PCA embedding of input features (case: {num_bus_str})", fontsize=18)
    plt.legend(fontsize=11, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(f'results/pinn_pca_case_{num_bus_str}.pdf')
    plt.show()


# ====================================
# 2) Distance-to-training vs feasibility
# ====================================

def run_pinn_distance_feasibility(
    num_bus_str: str,
    stress_cases: list[str],
    feasibility_dict: dict[str, np.ndarray] | None = None,
    k_neighbors: int = 10,
    max_train_points: int = 5000,
    max_test_points: int = 5000,
    random_state: int = 0,
):
    """
    For each stress scenario, compute:
      - average k-NN distance of its samples to the training cloud
      - feasibility rate (if feasibility_dict provided)

    Returns a dict[scenario] = stats and makes a quick scatter plot.

    feasibility_dict is expected to map which_data -> boolean mask
    aligned with the full test CSV rows for that scenario.
    """
    rng = np.random.default_rng(random_state)

    # ---- Load train and fit k-NN ----
    X_train, _ = load_pinn_inputs(num_bus_str, split="train")
    if len(X_train) > max_train_points:
        idx = rng.choice(len(X_train), size=max_train_points, replace=False)
        X_train = X_train[idx]

    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_train)

    results = {}

    for sc in stress_cases:
        X_sc, _ = load_pinn_inputs(num_bus_str, split="test", which_data=sc)

        feas_mask_full = None
        if feasibility_dict is not None and sc in feasibility_dict:
            feas_mask_full = np.asarray(feasibility_dict[sc], dtype=bool)
            if len(feas_mask_full) != len(X_sc):
                raise ValueError(
                    f"Feasibility mask length mismatch for '{sc}': "
                    f"{len(feas_mask_full)} vs {len(X_sc)}"
                )

        # subsample
        if len(X_sc) > max_test_points:
            idx = rng.choice(len(X_sc), size=max_test_points, replace=False)
            X_sc_sub = X_sc[idx]
            if feas_mask_full is not None:
                feas_mask_sub = feas_mask_full[idx]
            else:
                feas_mask_sub = None
        else:
            X_sc_sub = X_sc
            feas_mask_sub = feas_mask_full

        # k-NN distance to training cloud
        distances, _ = nn.kneighbors(X_sc_sub)
        d_mean_per_sample = distances.mean(axis=1)

        avg_distance = float(d_mean_per_sample.mean())
        med_distance = float(np.median(d_mean_per_sample))
        q25 = float(np.percentile(d_mean_per_sample, 25))
        q75 = float(np.percentile(d_mean_per_sample, 75))

        if feas_mask_sub is not None:
            feas_rate = float(feas_mask_sub.mean()) * 100.0
        else:
            feas_rate = np.nan  # unknown

        results[sc] = {
            "avg_distance": avg_distance,
            "median_distance": med_distance,
            "q25_distance": q25,
            "q75_distance": q75,
            "feas_rate": feas_rate,
        }

    # ---- Print summary ----
    print(f"PINN – distance vs feasibility ({num_bus_str})")
    print(f"{'Scenario':25s}  {'AvgDist':>8s}  {'Feas%':>8s}")
    for sc in stress_cases:
        r = results[sc]
        feas_str = f"{r['feas_rate']:.2f}" if not np.isnan(r["feas_rate"]) else "  n/a  "
        print(f"{sc:25s}  {r['avg_distance']:8.3f}  {feas_str:>8s}")

    # ---- Scatter plot (distance vs feas) if feasibility exists ----
    if any(not np.isnan(results[sc]["feas_rate"]) for sc in stress_cases):
        plt.figure(figsize=(7, 5))
        for sc in stress_cases:
            r = results[sc]
            if np.isnan(r["feas_rate"]):
                continue
            plt.scatter(r["avg_distance"], r["feas_rate"])
            plt.text(
                r["avg_distance"],
                r["feas_rate"],
                sc,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        plt.xlabel("Average k-NN distance to training cloud")
        plt.ylabel("Feasibility rate (%)")
        plt.title(f"PINN – distance vs feasibility ({num_bus_str})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


# ================================
# 3) Scalar summary (total load)
# ================================

def run_pinn_scalar_summary(
    num_bus_str: str,
    stress_cases: list[str],
    max_points_per_set: int = 5000,
    random_state: int = 0,
):
    """
    Simple scalar summary for PINN: total "load" per sample.

    Here we just use sum over all feature dimensions as a proxy.

    Produces a boxplot of total-load distributions for:
      - train
      - each stress scenario

    Returns dict[name] -> np.ndarray of scalar values.
    """
    rng = np.random.default_rng(random_state)

    scalar_data = {}

    # ---- Train ----
    X_train, _ = load_pinn_inputs(num_bus_str, split="train")
    if len(X_train) > max_points_per_set:
        idx = rng.choice(len(X_train), size=max_points_per_set, replace=False)
        X_train = X_train[idx]
    scalar_data["train"] = X_train.sum(axis=1)

    sc_renamed={
        "in_distribution": "In distribution",
        "load_extreme_high": "High Load",
        "load_extreme_low": "Low Load",
        #"renewable_variability",
        "fuel_price_up": "Fuel price up",
        "fuel_price_down": "Fuel price down",
        "geo_shift": "Geo shift",
        "extreme_weather": "Extreme heat-wave",
        "nk_contingencies": "N-k contingencies",
        "reactive_scarcity": "Reactive scarcity",
        "islanded_operation": "Islanded operation"
    }
    # ---- Stress ----
    for sc in stress_cases:
        X_sc, _ = load_pinn_inputs(num_bus_str, split="test", which_data=sc)
        if len(X_sc) > max_points_per_set:
            idx = rng.choice(len(X_sc), size=max_points_per_set, replace=False)
            X_sc = X_sc[idx]
        scalar_data[sc_renamed[sc]] = X_sc.sum(axis=1)

    # ---- Boxplot ----
    labels = list(scalar_data.keys())
    data_list = [scalar_data[k] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_list, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total load (sum of features)")
    plt.title(f"PINN – total-load envelope (train vs stress) ({num_bus_str})")
    plt.tight_layout()
    plt.show()

    return scalar_data


# ===========================
# Example usage (script mode)
# ===========================

if __name__ == "__main__":
    # Example stress case list
    stress_cases = [
        "in_distribution",
        "load_extreme_high",
        "load_extreme_low",
        #"renewable_variability",
        "fuel_price_up",
        "fuel_price_down",
        "geo_shift",
        "extreme_weather",
        "nk_contingencies",
        "reactive_scarcity",
        "islanded_operation",
        #"adversarial_perturb",
    ]

     
    num_bus_str = "118"   

    
    feasibility_dict = None
    run_pinn_pca_embedding(num_bus_str, stress_cases, feasibility_dict)
    #run_pinn_distance_feasibility(num_bus_str, stress_cases, feasibility_dict)
    #run_pinn_scalar_summary(num_bus_str, stress_cases)
