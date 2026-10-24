"""
DeepOPF OOD analysis utilities

Experiments:
1) PCA embedding of input space 

2) Distance-to-training vs feasibility (k-NN distance)

3) Simple scalar summaries (e.g., total load distributions)

DeepOPF inputs are stored in .pt files:

  src/baselines/DeepOPF/datasets/<case>/train_dataset_in_distribution.pt 
  src/baselines/DeepOPF/datasets/<case>/test_dataset_<which_data>.pt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ===========================
# Config
# ===========================

DEEPOPF_DATA_ROOT = os.path.join("src", "baselines", "DeepOPF", "datasets")




# ===========================
# Loading helpers
# ===========================

def _dataset_to_numpy_X(dataset) -> np.ndarray:
    """
    Convert a DeepOPF dataset object to a numpy array of inputs X.

    Assumes each dataset item is either:
        (X, Y) or X
    and, if it's a TensorDataset, X is dataset.tensors[0].
    """
    # TensorDataset and similar
    if hasattr(dataset, "tensors"):
        X_tensor = dataset.tensors[0]
        return X_tensor.detach().cpu().numpy()

    # Generic PyTorch dataset: iterate with a DataLoader
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    xs = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        xs.append(x.detach().cpu())
    X_tensor = torch.cat(xs, dim=0)
    return X_tensor.numpy()


def load_deepopf_inputs(
    case: str,
    split: str = "train",
    which_data: str | None = None,
) -> np.ndarray:
    """
    Load DeepOPF input features X from .pt files.

    Parameters
    ----------
    case : str
        Case name under src/baselines/DeepOPF/datasets/.
        (e.g., "case14", "case118")
    split : {"train", "test"}
    which_data : str or None
        Stress scenario name for test data, e.g. "geo_shift".
        Ignored for split="train".

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    """
    base = os.path.join(DEEPOPF_DATA_ROOT, case)

    if split == "train":
        pt_path = None
         
        cand= 'training_dataset.pt'
        candidate_path = os.path.join(base, cand)
        if os.path.exists(candidate_path):
            pt_path = candidate_path
            
        if pt_path is None:
            raise FileNotFoundError(
                f"Could not find any DeepOPF train dataset under '{base}'. "
            )
    else:
        if which_data is None:
            raise ValueError("which_data must be provided for split='test'")
        pt_path = os.path.join(base, f"test_dataset_{which_data}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing DeepOPF test dataset: {pt_path}")

    loaded = torch.load(pt_path, map_location="cpu")

    if isinstance(loaded, dict) and "dataset" in loaded:
        dataset = loaded["dataset"]
    else:
        dataset = loaded

    X = _dataset_to_numpy_X(dataset)
    return X


# ===========================
# 1) PCA embedding experiment
# ===========================

def run_deepopf_pca_embedding(
    case: str,
    stress_cases: list[str],
    feasibility_dict: dict[str, np.ndarray] | None = None,
    n_components: int = 2,
    max_points_per_set: int = 3000,
    random_state: int = 0,
):
    """
    PCA embedding of DeepOPF inputs for:
      - train (in-distribution)
      - all stress_cases (test)

    Parameters
    ----------
    case : str
        Dataset folder (e.g., "case14").
    stress_cases : list of str
        Scenario names, e.g. ['geo_shift', 'extreme_weather', ...].
    feasibility_dict : dict[which_data -> np.ndarray of bool], optional
        Per-sample feasibility masks aligned with the raw dataset rows
        BEFORE any subsampling. If None, feasibility is ignored.
    """
    rng = np.random.default_rng(random_state)

    # ---- Load training data ----
    X_train = load_deepopf_inputs(case, split="train")
    if len(X_train) > max_points_per_set:
        idx = rng.choice(len(X_train), size=max_points_per_set, replace=False)
        X_train = X_train[idx]

    # ---- Load stress data ----
    scenario_X = {}
    scenario_feas_masks = {}

    for sc in stress_cases:
        X_sc = load_deepopf_inputs(case, split="test", which_data=sc)

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
    os.makedirs("results", exist_ok=True)
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
                label=f"{sc}",
            )
        if np.any(mask_infeas):
            plt.scatter(
                X_sc_2d[mask_infeas, 0],
                X_sc_2d[mask_infeas, 1],
                s=15,
                alpha=0.7,
                marker="x",
                label=f"{sc} (infeas)",
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"DeepOPF - PCA embedding of input features (case: {case})")
    plt.legend(fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"deepopf_pca_{case}.pdf"))
    plt.show()


# ====================================
# 2) Distance-to-training vs feasibility
# ====================================

def run_deepopf_distance_feasibility(
    case: str,
    stress_cases: list[str],
    feasibility_dict: dict[str, np.ndarray] | None = None,
    k_neighbors: int = 10,
    max_train_points: int = 5000,
    max_test_points: int = 5000,
    random_state: int = 0,
):
    """
    For each DeepOPF stress scenario, compute:
      - average k-NN distance of its samples to the training cloud
      - feasibility rate (if feasibility_dict provided)

    Returns a dict[scenario] = stats and makes a quick scatter plot.

    feasibility_dict is expected to map which_data -> boolean mask
    aligned with the full dataset rows for that scenario.
    """
    rng = np.random.default_rng(random_state)

    # ---- Load train and fit k-NN ----
    X_train = load_deepopf_inputs(case, split="train")
    if len(X_train) > max_train_points:
        idx = rng.choice(len(X_train), size=max_train_points, replace=False)
        X_train = X_train[idx]

    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_train)

    results = {}

    for sc in stress_cases:
        X_sc = load_deepopf_inputs(case, split="test", which_data=sc)

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
    print(f"DeepOPF - distance vs feasibility ({case})")
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
        plt.title(f"DeepOPF - distance vs feasibility ({case})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


# ================================
# 3) Scalar summary (total load)
# ================================

def run_deepopf_scalar_summary(
    case: str,
    stress_cases: list[str],
    max_points_per_set: int = 5000,
    random_state: int = 0,
):
    """
    Simple scalar summary for DeepOPF: total "load" per sample.

    Here we just use sum over all feature dimensions as a proxy.

    Produces a boxplot of total-load distributions for:
      - train
      - each stress scenario

    Returns dict[name] -> np.ndarray of scalar values.
    """
    rng = np.random.default_rng(random_state)

    scalar_data = {}

    # ---- Train ----
    X_train = load_deepopf_inputs(case, split="train")
    if len(X_train) > max_points_per_set:
        idx = rng.choice(len(X_train), size=max_points_per_set, replace=False)
        X_train = X_train[idx]
    scalar_data["train"] = X_train.sum(axis= 1)

    # ---- Stress ----
    for sc in stress_cases:
        X_sc = load_deepopf_inputs(case, split="test", which_data=sc)
        if len(X_sc) > max_points_per_set:
            idx = rng.choice(len(X_sc), size= max_points_per_set, replace= False)
            X_sc = X_sc[idx]
        scalar_data[sc] = X_sc.sum(axis=1)

    # ---- Boxplot ----
    labels = list(scalar_data.keys())
    data_list = [scalar_data[k] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_list, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total load (sum of features)")
    plt.title(f"DeepOPF - total-load envelope (train vs stress) ({case})")
    plt.tight_layout()
    plt.show()

    return scalar_data



if __name__ == "__main__":
    
    stress_cases = [
        "in_distribution",
        "load_extreme_high",
        "load_extreme_low",
        # "renewable_variability",
        "fuel_price_up",
        "fuel_price_down",
        "geo_shift",
        "extreme_weather",
        "nk_contingencies",
        "reactive_scarcity",
        "islanded_operation",
        # "adversarial_perturb",
    ]

    case = "case118"

    feasibility_dict = None  # or fill with scenario -> boolean array
    run_deepopf_pca_embedding(case, stress_cases, feasibility_dict)
    run_deepopf_distance_feasibility(case, stress_cases, feasibility_dict)
    run_deepopf_scalar_summary(case, stress_cases)
