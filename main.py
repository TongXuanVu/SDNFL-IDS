"""SDN-FL IDS — mot lenh chay het, dung CHE DO CUA BAI BAO.

Hbaieb, Ayed, Chaari, "Federated learning based IDS approach for the IoV",
ARES 2022.

MAC DINH = dung bai nhat:
  --weighting paper   Eq. (3):  GM(r+1) = 1/C * SUM_i W_i * LM_i
                      voi W_i quyet dinh boi "the current state of SDN
                      controllers (e.g, throughput and latency)".
                      KHONG dung so mau, KHONG dung tin cay hanh vi.
  --arch cnn          1-D CNN (bai so sanh Random Forest / 1-D CNN / 1-D RNN;
                      ta giu CNN1D lam backbone chung voi P1/P2/P3)
  full data           moi client dung het shard, danh gia tren het tap test

CHO LECH SO VOI BAI, phai ghi trong bao cao:
  1. Bai dung C = 2 SDN controller. Ta dung 100 client (yeu cau cua ban).
  2. Bang 1 cua bai: batch 32, 10 round, 7 epoch, lr 0.5. Mac dinh o day la
     batch 512, 30 round, 1 epoch, lr 1e-3 — de so sanh cong bang voi P1/P2/P3.
     Muon chay dung bang 1 thi them --paper_hparams.
  3. throughput/latency: bai do that bang Mininet-WiFi + SUMO. Ta khong co, nen
     --simulate-sdn sinh so NGAU NHIEN. Bieu do trong so ve tu do la NHIEU, dung
     rut ket luan tu no. Xem README.

Chay:
  python main.py --data_dir /kaggle/input/... --num_users 100
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser(description="SDN-FL IDS: mot lenh chay het")
    p.add_argument("--data_dir", required=True,
                   help="Thu muc chua federated_data/ va global_test_data.pt")
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--num_users", type=int, default=100)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--com_round", type=int, default=30, help="Round MOI task")
    p.add_argument("--local_ep", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=0, help="0 = full data")
    p.add_argument("--test_samples", type=int, default=0,
                   help="0 = danh gia tren HET tap test moi round")
    p.add_argument("--arch", choices=["cnn", "rnn"], default="cnn")
    p.add_argument("--weighting",
                   choices=["paper", "trust", "state", "samples"], default="paper",
                   help="paper = Eq.(3) cua bai (mac dinh)")
    p.add_argument("--paper_hparams", action="store_true",
                   help="Dung dung Bang 1 cua bai: batch 32, 7 epoch, lr 0.5, "
                        "10 round/task")
    p.add_argument("--cm_every", type=int, default=5)
    p.add_argument("--flat", action="store_true",
                   help="Gop 5 task lam mot (khong class-incremental)")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--fed_subdir", default="federated_data",
                   choices=["federated_data", "federated_data_fewshot",
                            "federated_data_10shot"])
    p.add_argument("--actor_gpus", type=float, default=-1.0)
    p.add_argument("--actor_cpus", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.paper_hparams:                      # Bang 1 cua bai
        a.batch_size, a.local_ep, a.com_round = 32, 7, 10
        a.lr = 0.5 if a.arch == "cnn" else 1.0

    argv = [
        "run_sim.py",
        "--data-dir", a.data_dir,
        "--out-dir", a.out_dir,
        "--clients", str(a.num_users),
        "--rounds", str(a.com_round),
        "--tasks", "none" if a.flat else ",".join(str(t) for t in range(a.tasks)),
        "--local-epochs", str(a.local_ep),
        "--batch-size", str(a.batch_size),
        "--lr", str(a.lr),
        "--max-samples", str(a.max_samples),
        "--test-samples", str(a.test_samples),
        "--cm-every", str(a.cm_every),
        "--seed", str(a.seed),
        "--arch", a.arch,
        "--weighting", a.weighting,
        "--simulate-sdn",
        "--actor-gpus", str(a.actor_gpus),
        "--actor-cpus", str(a.actor_cpus),
    ]
    if a.restart:
        argv.append("--restart")

    print("=" * 70)
    print("SDN-FL IDS | Hbaieb, Ayed, Chaari, ARES 2022")
    print(f"  du lieu   : {a.data_dir}")
    print(f"  ket qua   : {a.out_dir}")
    print(f"  cau hinh  : {a.num_users} client | {a.tasks} task x {a.com_round} "
          f"round = {a.tasks * a.com_round} round")
    print(f"  model     : {'1-D CNN' if a.arch == 'cnn' else '1-D RNN'}")
    mo_ta = ("Eq.(3) cua bai — W tu throughput/latency"
             if a.weighting == "paper" else a.weighting)
    print(f"  tong hop  : {mo_ta}")
    print(f"  sieu tham so: batch {a.batch_size}, {a.local_ep} epoch, lr {a.lr}"
          f"{'  (Bang 1 cua bai)' if a.paper_hparams else ''}")
    print("  LUU Y: throughput/latency la so MO PHONG ngau nhien, khong phai do that")
    print("=" * 70, flush=True)

    sys.argv = argv
    import run_sim
    run_sim.main()


if __name__ == "__main__":
    main()
