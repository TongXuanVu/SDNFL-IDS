"""P4 - SDN-FL IDS: Flower server (cloud) voi FedAvg co trong so trang thai + tin cay.

Bai bao: Hbaieb, Ayed, Chaari, ARES 2022.

Khac FedAvg chuan: trong so cua controller k KHONG chi la so mau n_k, ma la

    w_k  ~  n_k^alpha  *  q_k^beta  *  t_k^gamma

trong do
    q_k = (thr_k / max_thr) * (min_lat / lat_k)      -- chat luong controller,
          controller bang thong cao & do tre thap thi q_k -> 1
    t_k = node_trust_k * behaviour_k                  -- tin cay tong hop
    behaviour_k = EMA cua cosine(update_k, update trung binh) da cat ve [0,1]

node_trust do client bao cao (node-properties). behaviour_k do server tu do:
controller nao lien tuc gui update lech huong dam dong se bi ha tin cay ->
day la phan bai bao mo ta bang loi ("trust metric based on node properties")
nhung khong cho cong thuc; can ghi ro day la LUA CHON CAI DAT cua ta.

  --weighting samples  : FedAvg chuan (doi chung, tat toan bo phan tren)
  --weighting state    : chi dung n_k * q_k
  --weighting trust    : day du n_k * q_k * t_k   (mac dinh)

Chay:
  python server_iov.py --rounds 30 --num-clients 10 --arch cnn
  python server_iov.py --rounds 30 --arch rnn --weighting samples
  python server_iov.py --mode test --ckpt out/checkpoints/latest.pth
"""
import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (FitRes, Parameters, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from models_sdn import build_model, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = r"C:\FederatedLearning\Rebuild-IOV\P4-SDNFL-IDS\out"


def flatten(ndarrays) -> np.ndarray:
    return np.concatenate([a.ravel() for a in ndarrays]).astype(np.float64)


class TrustWeightedFedAvg(fl.server.strategy.FedAvg):
    """FedAvg voi trong so = so mau x chat luong controller x tin cay."""

    def __init__(self, model, ckpt_dir, start_round=0, weighting="trust",
                 alpha=1.0, beta=1.0, gamma=1.0, trust_ema=0.7,
                 weight_log: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.start_round = start_round
        self.weighting = weighting
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.trust_ema = trust_ema
        self.behaviour: Dict[str, float] = {}          # cid -> tin cay hanh vi
        self.weight_log = weight_log
        # Chi tao moi khi chay tu dau. Neu resume thi ghi tiep, khong duoc ghi de
        # (truoc day resume lam mat sach lich su trong so cac round da chay).
        if weight_log and (start_round == 0 or not os.path.exists(weight_log)):
            with open(weight_log, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["round", "client", "n_samples", "throughput_mbps", "latency_ms",
                     "node_trust", "quality", "behaviour", "weight"])

    # ---- tinh trong so ----------------------------------------------------
    def _quality(self, results) -> np.ndarray:
        thr = np.array([r.metrics.get("throughput_mbps", 1.0) for _, r in results], float)
        lat = np.array([r.metrics.get("latency_ms", 1.0) for _, r in results], float)
        thr = np.clip(thr, 1e-6, None)
        lat = np.clip(lat, 1e-6, None)
        return (thr / thr.max()) * (lat.min() / lat)

    def _behaviour(self, results, flats, n_samples) -> np.ndarray:
        """Cosine giua update tung client va update trung binh, lam tron bang EMA."""
        mat = np.stack(flats)
        w = n_samples / n_samples.sum()
        mean = (mat * w[:, None]).sum(0)
        nm = np.linalg.norm(mean) + 1e-12
        cos = (mat @ mean) / (np.linalg.norm(mat, axis=1) * nm + 1e-12)
        cur = np.clip(cos, 0.0, 1.0)
        out = np.empty_like(cur)
        for i, (proxy, _) in enumerate(results):
            prev = self.behaviour.get(proxy.cid, 1.0)
            val = self.trust_ema * prev + (1.0 - self.trust_ema) * float(cur[i])
            self.behaviour[proxy.cid] = val
            out[i] = val
        return out

    def _weights(self, results, flats, rnd) -> np.ndarray:
        n = np.array([r.num_examples for _, r in results], float)
        if self.weighting == "samples":
            w = n.copy()
            q = np.ones_like(n)
            beh = np.ones_like(n)
            node = np.ones_like(n)
        else:
            q = self._quality(results)
            node = np.array([r.metrics.get("node_trust", 1.0) for _, r in results], float)
            if self.weighting == "state":
                beh = np.ones_like(n)
                t = np.ones_like(n)
            else:
                beh = self._behaviour(results, flats, n)
                t = np.clip(node * beh, 1e-6, None)
            w = (n ** self.alpha) * (q ** self.beta) * (t ** self.gamma)
        w = w / w.sum()

        if self.weight_log:
            with open(self.weight_log, "a", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                for i, (proxy, r) in enumerate(results):
                    cid = r.metrics.get("client_id", proxy.cid)
                    wr.writerow([rnd, cid, r.num_examples,
                                 round(r.metrics.get("throughput_mbps", 0.0), 3),
                                 round(r.metrics.get("latency_ms", 0.0), 3),
                                 round(float(node[i]), 4), round(float(q[i]), 4),
                                 round(float(beh[i]), 4), round(float(w[i]), 6)])
        return w

    # ---- Flower API -------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} controller loi")

        all_nd = [parameters_to_ndarrays(r.parameters) for _, r in results]
        flats = [flatten(a) for a in all_nd]
        abs_round = self.start_round + server_round
        w = self._weights(results, flats, abs_round)

        agg = [sum(w[i] * all_nd[i][j] for i in range(len(all_nd)))
               for j in range(len(all_nd[0]))]

        losses = np.array([r.metrics.get("train_loss", 0.0) for _, r in results])
        n = np.array([r.num_examples for _, r in results], float)
        metrics: Dict[str, Scalar] = {
            "train_loss": float((losses * n).sum() / max(n.sum(), 1)),
            "num_clients": len(results),
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "weight_entropy": float(-(w * np.log(w + 1e-12)).sum()),
        }
        if self.behaviour:
            metrics["behaviour_mean"] = float(np.mean(list(self.behaviour.values())))
        logger.info(f"[Round {server_round}] tong hop {len(results)} controller "
                    f"({self.weighting}) | trong so {w.min():.4f}..{w.max():.4f} "
                    f"entropy={metrics['weight_entropy']:.3f} "
                    f"train_loss={metrics['train_loss']:.4f}")

        sd = C.ndarrays_to_state_dict(self.model, agg)
        C.save_checkpoint(self.ckpt_dir, abs_round, sd,
                          extra={"train_loss": metrics["train_loss"],
                                 "behaviour": dict(self.behaviour)})
        return ndarrays_to_parameters(agg), metrics


# ----------------------------------------------------------------------------
def make_evaluate_fn(model, loader, criterion, device, csv_file, out_dir,
                     class_names, total_rounds, start_round, task, arch, cm_every=0):
    def evaluate_fn(server_round: int, parameters, config):
        if server_round == 0:
            return None
        abs_round = start_round + server_round
        model.load_state_dict(C.ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m, y_true, y_pred = C.evaluate(model, loader, criterion, device)
        C.log_and_save_metrics(abs_round, m, csv_file)
        # Ghi confusion matrix o cuoi task, VA dinh ky neu bat --cm-every,
        # de bi cat giua chung van con ban gan nhat.
        if server_round == total_rounds or (cm_every and abs_round % cm_every == 0):
            tag = f"{arch}_task{task}" if task is not None else f"{arch}_final"
            C.save_confusion_matrix(y_true, y_pred, out_dir, tag, class_names)
        return m["loss"], {k: v for k, v in m.items() if k != "loss"}
    return evaluate_fn


def fit_config_fn(local_epochs: int, lr: float):
    def fn(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round, "local_epochs": local_epochs, "lr": lr}
    return fn


def run_test(args, model, device):
    # checkpoint tach theo kien truc: checkpoints_cnn / checkpoints_rnn
    ckpt = args.ckpt or os.path.join(args.out_dir, f"checkpoints_{args.arch}", "latest.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt}")
    rnd, _ = C.load_checkpoint(ckpt, model)
    model.to(device)
    logger.info(f"Nap checkpoint {ckpt} (round {rnd})")
    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    m, y_true, y_pred = C.evaluate(model, loader, nn.CrossEntropyLoss(), device)
    logger.info(C.format_metrics(rnd, m))
    C.append_csv_row(os.path.join(args.out_dir, f"test_metrics_{args.arch}.csv"),
                     [rnd] + [round(m[k], 6) for k in C.METRIC_KEYS])
    tag = f"test_{args.arch}" + (f"_task{args.task}" if args.task is not None else "")
    C.save_confusion_matrix(y_true, y_pred, args.out_dir, tag,
                            C.load_class_names(args.data_dir))


def main():
    p = argparse.ArgumentParser(description="P4 SDN-FL IDS Flower server")
    p.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    p.add_argument("--arch", choices=["cnn", "rnn"], default="cnn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    # --- trong so tong hop ---
    p.add_argument("--weighting", choices=["trust", "state", "samples"], default="trust")
    p.add_argument("--alpha", type=float, default=1.0, help="Mu cua so mau")
    p.add_argument("--beta", type=float, default=1.0, help="Mu cua chat luong controller")
    p.add_argument("--gamma", type=float, default=1.0, help="Mu cua tin cay")
    p.add_argument("--trust-ema", type=float, default=0.7)
    # --- chung ---
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--address", type=str, default="0.0.0.0:8084")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--cm-every", type=int, default=0,
                   help="Ghi confusion matrix moi N round (0 = chi cuoi task)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, f"server_{args.arch}.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiet bi: {device} | che do: {args.mode} | arch: {args.arch} | "
                f"weighting: {args.weighting} | task: {args.task}")

    model = build_model(args.arch, NUM_GLOBAL_CLASSES, args.dropout,
                        args.hidden, args.layers).to(device)

    if args.mode == "test":
        run_test(args, model, device)
        return

    ckpt_dir = os.path.join(args.out_dir, f"checkpoints_{args.arch}")
    start_round = 0
    if args.mode == "resume":
        ckpt = args.ckpt or os.path.join(ckpt_dir, "latest.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Khong co checkpoint de resume: {ckpt}")
        start_round, _ = C.load_checkpoint(ckpt, model)
        model.to(device)
        logger.info(f"Resume tu round {start_round} ({ckpt})")

    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    class_names = C.load_class_names(args.data_dir)
    suffix = f"_{args.arch}" + (f"_task{args.task}" if args.task is not None else "")
    csv_file = os.path.join(args.out_dir, f"metrics{suffix}.csv")

    strategy = TrustWeightedFedAvg(
        model=model, ckpt_dir=ckpt_dir, start_round=start_round,
        weighting=args.weighting, alpha=args.alpha, beta=args.beta,
        gamma=args.gamma, trust_ema=args.trust_ema,
        weight_log=os.path.join(args.out_dir, f"client_weights{suffix}.csv"),
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(args.num_clients * args.fraction_fit)),
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=ndarrays_to_parameters(C.get_model_parameters(model)),
        on_fit_config_fn=fit_config_fn(args.local_epochs, args.lr),
        evaluate_fn=make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                     csv_file, args.out_dir, class_names,
                                     args.rounds, start_round, args.task, args.arch,
                                     args.cm_every),
    )

    logger.info(f"Server lang nghe {args.address} | {args.rounds} round | CSV -> {csv_file}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    logger.info(f"Xong. Ket qua trong {args.out_dir}")


if __name__ == "__main__":
    main()
