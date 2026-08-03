# Reproducibility checklist

Mục tiêu file này: liệt kê chính xác những gì cần để (a) một máy khác clone
repo từ GitHub và tái lập được đúng môi trường, (b) một GPU server bắt đầu
train được ngay mà không phải đoán cấu trúc thư mục. Không lặp lại nội dung
đã có ở `README.md` (provenance P0) hay `docs/execution-plan.md`
(milestone) — chỉ tổng hợp phần "cần gì, ở đâu, khớp cái gì".

## 1. Môi trường

| Thành phần | Yêu cầu | Ghi chú |
|---|---|---|
| Python | `>=3.10` | Khớp `external/p0_leakage_audit/pyproject.toml` (`requires-python = ">=3.10"`); đã test trên 3.10.12 |
| DH2A-KT deps | `requirements.txt` (numpy/pandas/pyyaml/networkx/scikit-learn/pytest, khoảng version đã pin) | `pip install -r requirements.txt` |
| P0 deps | `external/p0_leakage_audit/requirements.txt` (thêm scipy/tqdm/matplotlib/seaborn/pyarrow/torch/hypothesis) | `pip install -r external/p0_leakage_audit/requirements.txt` |
| pykt-toolkit | `github.com/pykt-team/pykt-toolkit`, không vendor kèm P0 | Clone + `pip install -e` riêng — đã đưa vào `scripts/00_setup_p0_submodule.sh` |
| GPU | 1x GPU tối thiểu tương đương RTX 3090 24GB | Khớp ngân sách compute P0 đã kiểm chứng (`docs/idea-D-plan.md` mục 9); không giả định multi-GPU |
| torch / torch_geometric | **Chưa pin phiên bản cụ thể** — phụ thuộc CUDA của server | Cài theo đúng lệnh chính thức của PyTorch/PyG cho bản CUDA thật của máy (`nvidia-smi` trước), rồi uncomment 2 dòng Tier 1 trong `requirements.txt` để lần cài sau tái lập được đúng bản đã dùng |

Nguyên tắc: không tự do nâng version ngoài khoảng đã pin trong
`requirements.txt` mà không cập nhật file — nếu một milestone cần version
mới hơn, sửa file trước, không cài tay rồi quên ghi lại.

## 2. Provenance đã cố định (không tự ý đổi)

| Gì | Giá trị | Nguồn |
|---|---|---|
| P0 vendored commit | `5e3d6a01c32a649f1655ac8712f03359bddfe478` (2026-08-02T15:36:04Z) | `external/p0_leakage_audit/.p0_vendored_commit.txt` |
| Split protocol | learner-based, ratio 0.7/0.1/0.2, 3 fold, seed `42`/`43`/`44` | `docs/idea-D-plan.md` mục 6, khớp P0 Section 4.1 |
| Manipulation-check | `p=0.90`, operator mặc định `node_drop`, seed `42` | `configs/*.yaml` mục `manipulation_check` |
| Config checksum (xes3g5m) | sha256 `4f9f35d9…d980b2` | `python scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml` |
| Config checksum (assist2012) | sha256 `a98dd247…4e6c8f9` | tương tự, `configs/assist2012.yaml` |
| Config checksum (junyi) | sha256 `24aad059…fd6bc8` | tương tự, `configs/junyi.yaml` |

Nếu bất kỳ checksum nào ở trên khác đi sau khi clone lại `external/p0_leakage_audit`
(dù là submodule sống hay bản vendor tĩnh), **dừng lại và điều tra trước khi
train** — nghĩa là repo P0 đã thay đổi so với commit đã pin, và mọi so sánh
baseline sẽ không còn hợp lệ (`docs/idea-D-plan.md` mục 10, rủi ro cao nhất).

## 3. Dữ liệu — không nằm trong Git

**Không có file dữ liệu thô hay đã tiền xử lý nào được commit lên GitHub** —
`.gitignore` chặn `external/p0_leakage_audit/data/{raw,processed}/*` và
`data/`. Đây là chủ ý: (a) dữ liệu nặng (XES3G5M/ASSISTments2012/Junyi lên
tới hàng triệu tương tác), (b) tránh vi phạm điều khoản phân phối lại của
từng dataset gốc. Trên GPU server, sau khi clone repo:

1. Theo `external/p0_leakage_audit/README.md` mục 3 để tải raw data đúng
   nguồn (không tự ý dùng bản khác — sẽ phá vỡ checksum ở mục 2).
2. Đặt vào `external/p0_leakage_audit/data/raw/<dataset>/`.
3. Chạy tiền xử lý bằng chính code P0 (`python -m src.preprocess --config
   ...`) — không viết pipeline tiền xử lý riêng.
4. Chạy `python -m src.split_checker --config ...` — phải báo "no learner
   leakage" trước khi đi tiếp.

## 4. Cấu trúc thư mục (đã dựng sẵn khung, để trống nội dung)

```
DH2A_KT/
├── results/
│   ├── tables/      # M5 -> dh2_kt_vs_p0.csv, M7 -> ground-truth CV F1, ...
│   └── figures/      # ablation plots, DDR sweep curves, ...
├── checkpoints/       # trong .gitignore -- KHONG commit trong so len GitHub
├── logs/               # trong .gitignore -- log chay agent (M8), train (M5)
├── external/p0_leakage_audit/data/
│   ├── raw/           # trong .gitignore -- tu tai theo muc 3
│   └── processed/      # trong .gitignore -- sinh ra tu buoc preprocess
└── (xem README.md muc "Cau truc du an" cho phan con lai, khong doi)
```

`results/tables/.gitkeep`, `results/figures/.gitkeep`, `checkpoints/.gitkeep`,
`logs/.gitkeep` đã được commit để khung thư mục tồn tại ngay sau `git
clone`, dù nội dung bên trong không được track — GPU server không cần tự
`mkdir` trước khi chạy script.

## 5. Điều **không** nên push lên GitHub

- `external/p0_leakage_audit/data/**` (raw + processed) — dung lượng lớn,
  vấn đề bản quyền phân phối lại.
- `checkpoints/*.pt` / `*.pth` / `*.ckpt` — trọng số model, tái tạo được từ
  script train, không cần versioning trong Git (cân nhắc Git LFS hoặc lưu
  ngoài nếu cần chia sẻ).
- `.venv/` — môi trường ảo local.
- `results/tables/*.csv` sinh ra từ mỗi lần chạy thật (khác với
  `baselines/p0_reused/*.csv`, vốn là kết quả **đã công bố của P0**, cố
  tình commit làm cột so sánh cố định — hai loại file khác nhau, đừng gộp).

## 6. Chuyển `external/p0_leakage_audit` từ bản vendor tĩnh sang submodule sống

Môi trường dựng scaffold này (mounted filesystem của phiên Cowork) không hỗ
trợ `unlink` nên không tự chuyển được — đã kiểm chứng lại ở bước hiện tại
(xoá `.git/index.lock` xong nhưng `rm` một file thường vẫn báo "Operation
not permitted"). Trên máy/GPU server không bị giới hạn này:

```bash
rm -rf external/p0_leakage_audit
git submodule add https://github.com/edu-risk-lab/leakage-controlled-kt-audit.git external/p0_leakage_audit
cd external/p0_leakage_audit && git checkout 5e3d6a01c32a649f1655ac8712f03359bddfe478 && cd ../..
git submodule update --init --recursive
```

Sau bước này, `.gitmodules` sẽ xuất hiện và cần được commit; bản vendor
tĩnh hiện tại (4.0MB, chỉ có code, không có data) vẫn hợp lệ để push lên
GitHub nếu bạn không muốn chuyển ngay — không phải việc bắt buộc trước khi
push, chỉ là thực hành tốt hơn về lâu dài (tránh trùng lặp lịch sử với repo
P0 gốc).

## 7. Trình tự tái lập đầy đủ (map theo `docs/execution-plan.md`)

```bash
git clone <URL repo DH2A_KT sau khi push>
cd DH2A_KT
bash scripts/00_setup_p0_submodule.sh          # env + pykt-toolkit + thư mục output
# -- theo muc 3 o tren de co du lieu that --
python scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml
python -m pytest tests/ -v                      # phai 12/12 pass truoc khi train
# -- roi moi chay M1-M9 theo docs/execution-plan.md --
```

Nếu bất kỳ bước nào trong trình tự trên thất bại trên GPU server nhưng đã
pass trong môi trường dựng scaffold, nghi ngờ đầu tiên nên là: (1) version
package lệch khỏi `requirements.txt`, hoặc (2) checksum config ở mục 2 đã
đổi vì `external/p0_leakage_audit` không khớp đúng commit đã pin.
