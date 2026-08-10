# SDN-FL IDS — tái hiện trên CICIoV

> Hbaieb, Ayed, Chaari, *A federated learning based IDS approach for the IoV*, ARES 2022

Tái hiện trên **CICIoV** (31 đặc trưng, 13 lớp) để so sánh được với ba bài còn
lại và với AFSIC-IoV / FedLiTeCAN trên cùng một backbone.

Ba repo anh em: [VANFED-IDS](https://github.com/TongXuanVu/VANFED-IDS) ·
[FEDIOV](https://github.com/TongXuanVu/FEDIOV) · [IOVFD](https://github.com/TongXuanVu/IOVFD) ·
[SDNFL-IDS](https://github.com/TongXuanVu/SDNFL-IDS)

---

## Ý tưởng được tái hiện

SDN controller = client, cloud = server. Trọng số tổng hợp không chỉ theo số mẫu mà là `w_k ∝ n_k^α · q_k^β · t_k^γ`, với `q_k` là chất lượng controller (throughput/latency) và `t_k` là độ tin cậy. So sánh 1D-CNN / 1D-RNN / Random Forest.

## Cài đặt

```bash
git clone https://github.com/TongXuanVu/SDNFL-IDS.git
cd SDNFL-IDS
pip install -r requirements.txt
```

## Chạy

Cần **1 server + N client**. Muốn chạy tay thì mở N+1 terminal, server trước:

```bash
python server_iov.py --rounds 30 --num-clients 10 --data-dir <DATA>
python client_iov.py --client-id 0 --data-dir <DATA>
python client_iov.py --client-id 1 --data-dir <DATA>
```

Trên Kaggle/Colab không mở được nhiều terminal — dùng `run_fl.py`, nó tự sinh
server + N client và chạy nối tiếp task 0→4 (class-incremental), resume giữa
các task nên số round liên tục:

```bash
python run_fl.py --data-dir <DATA> --clients 10 --rounds 20
python run_fl.py --data-dir <DATA> --tasks none      # FL thường, gộp cả 5 task
```

1D-RNN thay vì CNN:

```bash
python run_fl.py --data-dir <DATA> --server-extra="--arch rnn" --client-extra="--arch rnn --simulate-sdn"
```

FedAvg thường, để xem cơ chế trust có tác dụng không:

```bash
python run_fl.py --data-dir <DATA> --server-extra="--weighting samples" --client-extra="--simulate-sdn"
```

Random Forest liên kết (không qua Flower):

```bash
python rf_baseline.py --data-dir <DATA> --clients 0 1 2 3 4 5 6 7 8 9 --simulate-sdn
```


> `--server-extra` và `--client-extra` **bắt buộc viết dạng có dấu `=`**.
> Viết cách ra sẽ lỗi `expected one argument` vì argparse tưởng là option mới.

### Ba chế độ

```bash
python server_iov.py --mode train  --rounds 30
python server_iov.py --mode resume --rounds 50           # chạy tiếp từ latest.pth
python server_iov.py --mode test   --ckpt out/checkpoints/latest.pth
```

## Dữ liệu

Định dạng khớp AFSIC-IoV:

```
<DATA>/federated_data/client_<id>_task_<t>.pt    # t = 1..5, dict {'x','y'}
<DATA>/global_test_data.pt
<DATA>/class_mapping.json
```

Chia lớp theo task: `TASK_INCREMENTS = [3, 3, 3, 2, 2]` (13 lớp / 5 task).
`run_fl.py` tự bỏ qua client thiếu file của task đang chạy thay vì để server
treo chờ mãi.

## Kết quả

Đổ vào `--out-dir` (mặc định `out/`):

| File | Nội dung |
|---|---|
| `metrics_task*.csv` | 1 dòng/round, 12 cột: loss, accuracy, micro/macro/weighted P-R-F1 |
| `confusion_matrix_task*.csv` / `_normalized.csv` / `.png` | cuối mỗi task |
| `classification_report_task*.txt` | P/R/F1 từng lớp |
| `checkpoints/round_NNN.pth`, `latest.pth` | resume được |
| `client_weights_<arch>_task*.csv` | throughput / latency / trust / trọng số từng controller mỗi round |

Gộp nhiều lần chạy + đo mức độ quên:

```bash
python collect_results.py --runs A=out_a B=out_b --out-dir ket_qua
```

Sinh `comparison.csv`, ma trận quên từng lần chạy (`forgetting_*.csv`), và
`accuracy_curve.png`. Mức độ quên tính theo định nghĩa chuẩn class-incremental:
`forgetting(j) = max_{t<T} acc(j,t) − acc(j,T)`.

## Kiểm thử

```bash
python smoke_test.py
```

Tự sinh dữ liệu giả đúng định dạng, chạy 2 round, kiểm CSV đủ 12 cột, checkpoint
nạp lại được, confusion matrix có sinh ra, và cả `--mode test` lẫn `--mode resume`.

**Trạng thái:** **Đã chạy thật và đạt** trên dữ liệu giả, cả 3 kiến trúc: CNN và RNN chạy trọn 5 task nối tiếp, Random Forest ghép cây đúng qua không gian 13 lớp. Trọng số trust phân hoá đúng theo throughput/latency (0.07→0.50). Chưa chạy trên CICIoV thật.

## Khác gì so với bài báo

Không có Mininet-WiFi/Ryu/SUMO nên throughput/latency/trust được **mô phỏng** (`--simulate-sdn`). Muốn nối controller thật thì thay hàm `sample_controller_state()` trong `client_iov.py` bằng lời gọi REST API của Ryu. Bài mô tả trust metric bằng lời ("based on node properties") mà không cho công thức; ở đây `t_k = node_trust_k × behaviour_k` với `behaviour_k` là EMA của cosine giữa update client và update trung bình — **đây là lựa chọn cài đặt, không phải công thức của bài**. Random Forest không FedAvg được nên ghép cây theo trọng số.

Bài gốc **không công bố source code**. Mọi con số phải tự đo lại, không kỳ vọng
khớp bảng kết quả trong bài.

## Ghi chú về code dùng chung

`common.py` và `model_cnn1d.py` giống hệt ở cả 4 repo (cố ý — bốn bài phải dùng
chung backbone thì so sánh mới công bằng). Sửa một nơi thì phải đồng bộ cả 4;
`check_shared.py` đối chiếu hash giúp phát hiện lệch bản:

```bash
python check_shared.py --against ../VANFED-IDS
```
