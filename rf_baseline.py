"""P4 - SDN-FL IDS: baseline Random Forest (khong chay qua Flower).

Bai bao so sanh Random Forest voi 1D-CNN va 1D-RNN. Random Forest khong the
tong hop bang FedAvg (khong co vector trong so), nen dung cach chuan cho
"federated random forest": moi controller train mot rung cuc bo, cloud GHEP
cac cay lai thanh mot rung chung.

So cay lay tu moi controller ti le voi trong so cua no -- dung cong thuc
trong so nhu server_iov.py:  n_k^alpha * q_k^beta * t_k^gamma. Nho vay ket qua
so sanh duoc voi hai kien truc mang.

  --weighting samples  : ghep theo so mau (doi chung)
  --weighting trust    : ghep theo so mau x chat luong controller x tin cay

Chay:
  python rf_baseline.py --clients 0 1 2 3 4 5 6 7 8 9 --trees-total 300
  python rf_baseline.py --clients 0 1 2 3 --weighting samples --centralized
"""
import argparse

import json
import logging
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = r"C:\FederatedLearning\Rebuild-IOV\P4-SDNFL-IDS\out"


def controller_state(cid, rng, simulate):
    if not simulate:
        return 50.0, 50.0, 1.0
    return (float(rng.uniform(10, 100)), float(rng.uniform(5, 150)),
            float(rng.uniform(0.3, 1.0)))


class FederatedForest:
    """Rung chung ghep tu nhieu rung cuc bo, moi cay mot phieu.

    KHONG the gan thang estimators_ cua rung nay sang rung kia: moi controller
    chi thay MOT PHAN trong 13 lop, nen predict_proba cua chung co so cot khac
    nhau (vd (N,12) va (N,13)) -> sklearn bao loi broadcast.

    Cach dung o day: lay phieu cua tung cay roi anh xa ve dung chi so lop toan
    cuc qua forest.classes_, cong don tren khong gian 13 lop. So cay lay tu moi
    controller (n_trees_each) chinh la trong so cua controller do.
    """

    def __init__(self, forests, n_trees_each, n_classes):
        self.forests = list(forests)
        self.n_trees_each = [max(1, int(k)) for k in n_trees_each]
        self.n_classes = int(n_classes)
        self.n_estimators = sum(self.n_trees_each)

    def predict_proba(self, x):
        out = np.zeros((len(x), self.n_classes), dtype=np.float64)
        votes = 0
        for f, k in zip(self.forests, self.n_trees_each):
            idx = np.asarray(f.classes_, dtype=int)
            for tree in f.estimators_[:k]:
                out[:, idx] += tree.predict_proba(x)
                votes += 1
        return out / max(votes, 1)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description="P4 - baseline Random Forest lien ket")
    p.add_argument("--clients", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--trees-total", type=int, default=300,
                   help="Tong so cay trong rung chung")
    p.add_argument("--trees-local", type=int, default=100,
                   help="So cay moi controller train (>= phan duoc lay)")
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=200_000, help="Moi client")
    p.add_argument("--weighting", choices=["trust", "samples"], default="trust")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--simulate-sdn", action="store_true")
    p.add_argument("--centralized", action="store_true",
                   help="Train 1 rung tren du lieu gop (gioi han tren, khong lien ket)")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, "rf_baseline.log"))
    rng = np.random.default_rng(args.seed)
    tag = "rf_centralized" if args.centralized else f"rf_{args.weighting}"

    # --- train ---
    if args.centralized:
        xs, ys = [], []
        for cid in args.clients:
            xi, yi = C.load_client_data(args.data_dir, cid, args.task, args.max_samples)
            xs.append(xi)
            ys.append(yi)
        x, y = np.concatenate(xs), np.concatenate(ys)
        del xs, ys
        logger.info(f"Tap trung: n={len(y)}, {args.trees_total} cay")
        model = RandomForestClassifier(n_estimators=args.trees_total,
                                       max_depth=args.max_depth,
                                       n_jobs=args.n_jobs, random_state=args.seed,
                                       class_weight="balanced_subsample")
        model.fit(x, y)
        del x, y
        info = {"mode": "centralized", "n_trees": args.trees_total}
    else:
        forests, n_samples, quality, trust = [], [], [], []
        for cid in args.clients:
            x, y = C.load_client_data(args.data_dir, cid, args.task, args.max_samples)
            thr, lat, node_trust = controller_state(cid, rng, args.simulate_sdn)
            f = RandomForestClassifier(n_estimators=args.trees_local,
                                       max_depth=args.max_depth,
                                       n_jobs=args.n_jobs, random_state=args.seed + cid,
                                       class_weight="balanced_subsample")
            f.fit(x, y)
            forests.append(f)
            n_samples.append(len(y))
            quality.append((thr, lat))
            trust.append(node_trust)
            logger.info(f"Controller {cid}: n={len(y)} thr={thr:.1f} lat={lat:.1f} "
                        f"trust={node_trust:.3f} -> {args.trees_local} cay")
            del x, y

        n = np.array(n_samples, float)
        if args.weighting == "samples":
            w = n.copy()
        else:
            thr = np.array([q[0] for q in quality], float)
            lat = np.array([q[1] for q in quality], float)
            q = (thr / thr.max()) * (lat.min() / np.clip(lat, 1e-6, None))
            t = np.clip(np.array(trust, float), 1e-6, None)
            w = (n ** args.alpha) * (q ** args.beta) * (t ** args.gamma)
        w = w / w.sum()
        k = np.maximum(1, np.round(w * args.trees_total).astype(int))
        k = np.minimum(k, args.trees_local)
        for cid, wi, ki in zip(args.clients, w, k):
            logger.info(f"  controller {cid}: trong so={wi:.4f} -> gop {ki} cay")
        model = FederatedForest(forests, k.tolist(), C.NUM_GLOBAL_CLASSES)
        info = {"mode": "federated", "weighting": args.weighting,
                "weights": {int(c): float(wi) for c, wi in zip(args.clients, w)},
                "trees_per_client": {int(c): int(ki) for c, ki in zip(args.clients, k)},
                "n_trees": int(k.sum())}
        logger.info(f"Rung chung: {model.n_estimators} cay")

    # --- danh gia ---
    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    xt = loader.dataset.tensors[0].numpy()
    yt = loader.dataset.tensors[1].numpy()
    yp = model.predict(xt)
    m = C.compute_metrics(yt, yp)
    logger.info(C.format_metrics(0, m))

    C.append_csv_row(os.path.join(args.out_dir, f"metrics_{tag}.csv"),
                     [0] + [round(m[k_], 6) for k_ in C.METRIC_KEYS])
    C.save_confusion_matrix(yt, yp, args.out_dir, tag, C.load_class_names(args.data_dir))
    info["metrics"] = {k_: round(m[k_], 6) for k_ in C.METRIC_KEYS}
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Xong. Ket qua trong {args.out_dir}")


if __name__ == "__main__":
    main()
