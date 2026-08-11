"""P4 - SDN-FL IDS: Flower client = SDN controller.

Bai bao: Hbaieb, Ayed, Chaari, ARES 2022.

Moi client dai dien mot SDN controller quan mot vung duong. Ngoai trong so
model, client con bao cao TRANG THAI CONTROLLER de server dung lam trong so
tong hop:

  throughput_mbps : bang thong controller do duoc
  latency_ms      : do tre control-plane
  node_trust      : diem tin cay tu node-properties (0..1)

Bai bao lay cac so nay tu Mininet-WiFi + Ryu + SUMO. O day KHONG dung ns nen
chung duoc mo phong: co dinh qua tham so dong lenh, hoac sinh ngau nhien co
seed (--simulate-sdn) voi dao dong nhe moi round. Muon noi voi Ryu that thi
thay ham sample_controller_state() bang mot lenh goi REST API cua controller.

Chay:
  python client_iov.py --client-id 0 --arch cnn
  python client_iov.py --client-id 3 --arch rnn --throughput 40 --latency 120 --node-trust 0.4
  python client_iov.py --client-id 5 --simulate-sdn        # sinh trang thai ngau nhien
"""
import argparse
import logging
import os
import sys

import flwr as fl
import numpy as np
import torch
import torch.optim as optim

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from models_sdn import build_model, FocalLoss, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"


class SDNControllerClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, device, max_samples, batch_size,
                 task, lr, dropout, arch, hidden, layers,
                 throughput, latency, node_trust, simulate, jitter, seed):
        self.cid = client_id
        self.device = device
        self.lr = lr
        self.jitter = jitter
        self.seed = seed
        self.rng = np.random.default_rng(seed + client_id)

        # --- trang thai controller ---
        if simulate:
            self.throughput = float(self.rng.uniform(10.0, 100.0))     # Mbps
            self.latency = float(self.rng.uniform(5.0, 150.0))         # ms
            self.node_trust = float(self.rng.uniform(0.3, 1.0))
        else:
            self.throughput, self.latency, self.node_trust = throughput, latency, node_trust
        logger.info(f"[Controller {self.cid}] throughput={self.throughput:.1f} Mbps "
                    f"latency={self.latency:.1f} ms trust={self.node_trust:.3f}")

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)

        self.model = build_model(arch, NUM_GLOBAL_CLASSES, dropout, hidden, layers).to(device)
        self.criterion = FocalLoss(alpha=C.make_focal_alpha(y).to(device), gamma=2.0)

    def sample_controller_state(self, rnd=0):
        """Trang thai o round hien tai (co dao dong nhe quanh gia tri co so).

        Phai gieo theo (client, ROUND). Trong che do simulation cua Flower,
        doi tuong client bi TAO LAI moi round voi cung seed, nen neu dung
        self.rng thi moi round deu boc ra dung mot so — jitter tro thanh VO
        TAC DUNG mot cach am tham (do bang thuc nghiem: trong so round 1 va
        round 2 giong nhau den 0.0%).
        """
        if self.jitter <= 0:
            return self.throughput, self.latency, self.node_trust
        r = np.random.default_rng(self.seed + self.cid * 100_003 + rnd)
        f = lambda v: float(max(1e-3, v * (1.0 + r.normal(0, self.jitter))))
        return f(self.throughput), f(self.latency), min(1.0, f(self.node_trust))

    # ---- Flower API -------------------------------------------------------
    def get_parameters(self, config):
        return C.get_model_parameters(self.model)

    def set_parameters(self, parameters):
        self.model.load_state_dict(C.ndarrays_to_state_dict(self.model, parameters))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        rnd = int(config.get("server_round", 0))
        lr = float(config.get("lr", self.lr))

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=lr)
        total_loss, n_batches, correct, seen = 0.0, 0, 0, 0
        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
                correct += (out.argmax(1) == yb).sum().item()
                seen += yb.numel()

        avg = total_loss / max(n_batches, 1)
        thr, lat, trust = self.sample_controller_state(rnd)
        logger.info(f"[Controller {self.cid}][Round {rnd}] n={self.n_samples} "
                    f"loss={avg:.4f} acc={correct / max(seen, 1):.4f} | "
                    f"thr={thr:.1f} lat={lat:.1f} trust={trust:.3f}")
        return C.get_model_parameters(self.model), self.n_samples, {
            "client_id": self.cid,          # de server ghi log theo id that,
                                            # khong phai UUID noi bo cua Flower
            "train_loss": avg,
            "train_acc": correct / max(seen, 1),
            "throughput_mbps": thr,
            "latency_ms": lat,
            "node_trust": trust,
        }

    def evaluate(self, parameters, config):
        return 0.0, self.n_samples, {}


def main():
    p = argparse.ArgumentParser(description="P4 SDN-FL IDS Flower client")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--server", type=str, default="127.0.0.1:8084")
    p.add_argument("--arch", choices=["cnn", "rnn"], default="cnn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    # --- trang thai SDN controller ---
    p.add_argument("--throughput", type=float, default=50.0, help="Mbps")
    p.add_argument("--latency", type=float, default=50.0, help="ms")
    p.add_argument("--node-trust", type=float, default=1.0, help="0..1")
    p.add_argument("--simulate-sdn", action="store_true",
                   help="Sinh throughput/latency/trust ngau nhien theo seed")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="Dao dong trang thai controller moi round. 0 = giu co dinh "
                        "(mac dinh): trong so on dinh, tai lap duoc")
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    C.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = SDNControllerClient(
        args.client_id, args.data_dir, device, args.max_samples, args.batch_size,
        args.task, args.lr, args.dropout, args.arch, args.hidden, args.layers,
        args.throughput, args.latency, args.node_trust, args.simulate_sdn,
        args.jitter, args.seed)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
