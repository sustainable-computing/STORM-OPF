import argparse
import gzip
import io
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src import util
from src.baselines.PINN.PinnLayer import PinnLayer
from src.baselines.PINN.PinnModel import PinnModel
from src.baselines.PINN.create_example_parameters import create_example_parameters
from src.baselines.PINN.create_data import create_data


class PinnLayerExt(PinnLayer):
    def __init__(self, simulation_parameters, n_lbus):
        super().__init__(simulation_parameters)
        self._n_lbus = n_lbus

    def call(self, inputs, **kwargs):
        L_Val = tf.convert_to_tensor(inputs)
        n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u = \
            self.DenseCoreNetwork.call_inference(L_Val, **kwargs)
        L_Val_phys = L_Val[:, :2 * self._n_lbus]
        KKT_error = self.Get_KKT_error(
            n_o_v, n_o_pq, L_Val_phys,
            n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u,
        )
        return n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u, KKT_error


class PinnModelExt(PinnModel):
    def __init__(self, w1, w2, simulation_parameters):
        tf.keras.Model.__init__(self)
        self.n_bus = simulation_parameters['general']['n_bus']
        n_lbus = simulation_parameters['general']['n_lbus']
        self.PinnLayer = PinnLayerExt(simulation_parameters=simulation_parameters, n_lbus=n_lbus)
        n_d_p = simulation_parameters['data_creation']['n_data_points']
        n_c   = simulation_parameters['data_creation']['n_collocation']
        n_t   = n_d_p + n_c
        loss_weights = [n_t/n_d_p, n_t/n_d_p,
                        w1*1e-6, w1*1e-6, w1*1e-6, w1*1e-6, w1*1e-6, w1*1e-6,
                        w2*1e-5]
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.mean_absolute_error,
                     loss_weights=loss_weights)


# -----------------------------------------------------------------------
# Scenario loading helpers
# -----------------------------------------------------------------------

def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _load_paths(case, group):
    d = ("src/scenarios/{}/training".format(case) if group == "in_distribution"
         else "src/scenarios/{}/training_stressed/{}".format(case, group))
    mf = pd.read_csv(os.path.join(d, "manifest.csv"))
    return [os.path.join(d, f) for f in mf[mf["status"] == "ok"]["file"].tolist()]


# -----------------------------------------------------------------------
# Parallel norm-stats workers
# -----------------------------------------------------------------------

_NORM_LBUS = None


def _norm_worker_init(l_bus):
    global _NORM_LBUS
    _NORM_LBUS = l_bus
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _norm_worker(path):
    with gzip.open(path, "rb") as f:
        obj = json.loads(f.read())["_object"]
    load_df = pd.read_json(io.StringIO(obj["load"]["_object"]), orient="split")
    n_bus = len(pd.read_json(io.StringIO(obj["bus"]["_object"]), orient="split"))
    P_D = np.zeros(n_bus)
    Q_D = np.zeros(n_bus)
    agg = load_df.groupby("bus")[["p_mw", "q_mvar"]].sum()
    P_D[agg.index.values.astype(int)] = agg["p_mw"].values
    Q_D[agg.index.values.astype(int)] = agg["q_mvar"].values
    return P_D[_NORM_LBUS], Q_D[_NORM_LBUS]


def _build_norm_stats(case, l_bus, build_workers=0):
    paths = _load_paths(case, "in_distribution")
    print("  Loading {} scenario files for norm stats...".format(len(paths)))

    if build_workers > 1:
        chunksize = max(1, len(paths) // (build_workers * 4))
        with _mp_context().Pool(
            processes=build_workers,
            initializer=_norm_worker_init,
            initargs=(l_bus,),
        ) as pool:
            rows = pool.map(_norm_worker, paths, chunksize=chunksize)
    else:
        _norm_worker_init(l_bus)
        rows = [_norm_worker(p) for p in paths]

    P_arr = np.vstack([p for p, _ in rows])
    Q_arr = np.vstack([q for _, q in rows])
    P_mu  = P_arr.mean(0)
    P_sig = P_arr.std(0) + 1e-8
    Q_mu  = Q_arr.mean(0)
    Q_sig = Q_arr.std(0) + 1e-8
    nominal_total = float(P_arr.sum(1).mean())
    return P_mu, P_sig, Q_mu, Q_sig, nominal_total


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gen_data", action="store_true", default=False,
                        help="Build dataset from scenario files and save. "
                             "If False (default), load the previously saved dataset.")
    parser.add_argument("--build_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel norm-stats workers")
    args = parser.parse_args()

    num_bus = args.case[4:]
    ncpu = os.cpu_count() or 1
    if args.build_workers == -1:
        args.build_workers = max(1, min(8, ncpu - 1))

    dataset_dir  = os.path.join("src", "baselines", "PINN", "datasets", args.case)
    dataset_path = os.path.join(dataset_dir, "ext_input_dataset.npz")

    base_net, _ = util.get_base_net(args.case, "in_distribution")
    simulation_parameters = create_example_parameters(base_net)
    n_lbus = simulation_parameters['general']['n_lbus']

    if args.gen_data:
        l_bus  = simulation_parameters['general']['l_bus']
        slack  = int(base_net.ext_grid.bus.iloc[0])
        active_gen_idx = [
            i for i, (m, b) in enumerate(zip(
                base_net.gen.in_service.values.astype(bool), base_net.gen.bus.values))
            if m and int(b) != slack
        ]
        trafo_idx = list(base_net.trafo.index)

        print("Build workers: {}".format(args.build_workers))
        print("Computing normalization stats from nominal scenarios...")
        P_mu, P_sig, Q_mu, Q_sig, nominal_total = _build_norm_stats(
            args.case, l_bus, args.build_workers)

        x_training_orig, y_training, Lg_Max = create_data(base_net, simulation_parameters)
        L_val_orig, x_type = x_training_orig
        n_total = L_val_orig.shape[0]

        P_D = L_val_orig[:, :n_lbus] * P_sig + P_mu
        total_P   = P_D.sum(axis=1, keepdims=True)
        p_frac    = P_D / (total_P + 1e-8)
        log_scale = np.clip(np.log((total_P + 1e-8) / (nominal_total + 1e-8)), -3, 3)

        n_gen   = len(active_gen_idx)
        n_trafo = len(trafo_idx)
        L_val_ext = np.concatenate(
            [L_val_orig, p_frac, log_scale,
             np.ones((n_total, n_gen),   dtype=float),
             np.ones((n_total, n_gen),   dtype=float),
             np.ones((n_total, n_gen),   dtype=float),
             np.ones((n_total, n_trafo), dtype=float)],
            axis=1
        ).astype(np.float32)

        os.makedirs(dataset_dir, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            L_val_ext=L_val_ext,
            x_type=x_type,
            Lg_Max=np.array([float(np.asarray(x).ravel()[0]) for x in Lg_Max]),
            P_mu=P_mu, P_sig=P_sig,
            Q_mu=Q_mu, Q_sig=Q_sig,
            nominal_total=np.array([nominal_total]),
            **{"y_{}".format(i): arr for i, arr in enumerate(y_training)},
        )
        print("Saved dataset -> {}  ({} samples, in_dim={})".format(
            dataset_path, n_total, L_val_ext.shape[1]))
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "Dataset not found: {}\n"
                "Run with --gen_data first to build it.".format(dataset_path)
            )
        d          = np.load(dataset_path, allow_pickle=False)
        L_val_ext  = d["L_val_ext"]
        x_type     = d["x_type"]
        Lg_Max     = list(d["Lg_Max"])          # keep as list so PinnLayer can index [0],[1],[2]
        n_y        = sum(1 for k in d.files if k.startswith("y_"))
        y_training = [d["y_{}".format(i)] for i in range(n_y)]
        print("Loaded dataset from {}  ({} samples, in_dim={})".format(
            dataset_path, L_val_ext.shape[0], L_val_ext.shape[1]))

    print("Extended input dim: {}  (original 2*n_lbus = {})".format(
        L_val_ext.shape[1], 2 * n_lbus))

    simulation_parameters.update({'Lg_Max': Lg_Max})
    model = PinnModelExt(0.05, 0.05, simulation_parameters)

    for n_epochs, batch_size in zip(
        simulation_parameters['training']['epoch_schedule'],
        simulation_parameters['training']['batching_schedule']
    ):
        model.fit(
            [L_val_ext, x_type],
            y_training,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
        )
        print("  completed {} epochs".format(n_epochs))

    save_dir = os.path.join("src", "baselines", "PINN", "trained_model_ext_input", num_bus)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    print("Saved -> {}".format(save_dir))


if __name__ == "__main__":
    main()
