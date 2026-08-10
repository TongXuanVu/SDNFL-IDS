"""P4 - SDN-FL IDS: ba bo phan lop de doi chung (1D-CNN, 1D-RNN, Random Forest).

Bai bao: Hbaieb, Ayed, Chaari, "A federated learning based IDS approach for the
IoV", ARES 2022.

Bai so sanh Random Forest / 1D-CNN / 1D-RNN. CNN1D lay nguyen tu P1 de ba
baseline cua ta va bon bai bao dung chung backbone.

  --arch cnn   -> CNN1D_IDS   (import tu P1)
  --arch rnn   -> RNN1D_IDS   (GRU 2 lop, file nay)
  --arch rf    -> RandomForest (rf_baseline.py, khong qua Flower)
"""
import os
import sys

import torch
import torch.nn as nn

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

from model_cnn1d import CNN1D_IDS, FocalLoss, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402,F401


class RNN1D_IDS(nn.Module):
    """1D-RNN: coi 31 dac trung nhu chuoi dai 31, moi buoc 1 chieu.

    (B,31) -> (B,31,1) -> GRU(hidden, 2 lop, bidirectional) -> trang thai cuoi -> FC
    """

    def __init__(self, input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES,
                 hidden=64, layers=2, dropout=0.15, bidirectional=True):
        super().__init__()
        self.input_len = input_len
        self.num_classes = num_classes
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        feat = hidden * (2 if bidirectional else 1)
        self.feat_dim = feat
        self.classifier = nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    def embed(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)                  # (B, 31, 1)
        out, _ = self.rnn(x)
        return out[:, -1, :]                     # trang thai buoc cuoi

    def forward(self, x):
        return self.classifier(self.embed(x))


def build_model(arch: str, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15,
                hidden=64, layers=2):
    arch = arch.lower()
    if arch == "cnn":
        return CNN1D_IDS(INPUT_LEN, num_classes, dropout)
    if arch == "rnn":
        return RNN1D_IDS(INPUT_LEN, num_classes, hidden, layers, dropout)
    raise ValueError(f"arch phai la 'cnn' hoac 'rnn' (nhan duoc: {arch}). "
                     f"Random Forest chay bang rf_baseline.py")


if __name__ == "__main__":
    for a in ("cnn", "rnn"):
        m = build_model(a)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{a}: params={n:,} out={tuple(m(torch.randn(4, INPUT_LEN)).shape)}")
