# DH2A-KT

Leakage-audited knowledge attribution for knowledge tracing. The KBS
manuscript is `paper/main.tex`; table-to-script map is
[`docs/paper-artifacts.md`](docs/paper-artifacts.md). DH²A-KT is an
attribution pipeline (construct → audit → matched twins → credit gate),
not a hypergraph predictor. Headline evidence is on XES3G5M under a
clean $L{=}400$ protocol.

Historical scaffold notes below describe the early repository layout.
They are **not** the current submission status.

**Idea D**: Dynamic Heterogeneous Hypergraph + Agentic Knowledge Tracing, xây trực tiếp trên nền [P0](https://github.com/edu-risk-lab/leakage-controlled-kt-audit) (*Leakage-Controlled Concept Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing*, Dao Minh Tuan et al., APIN submission).

Kế hoạch đầy đủ (kiến trúc, câu hỏi nghiên cứu, timeline, bảng rủi ro) nằm ở [`docs/idea-D-plan.md`](docs/idea-D-plan.md).

## Historical scaffold notes (superseded by the KBS draft)

Đây là bộ khung dự án (project scaffold), được dựng trong một phiên làm việc — **không phải** kết quả của 15 tuần thực nghiệm mô tả trong `docs/idea-D-plan.md`. Cụ thể, những gì **đã thực sự chạy được**:

- `dh2a_kt/p0_bridge.py` — import trực tiếp, không sửa đổi, các hàm gốc của P0 (xây `E_pre`, 4 chỉ số leakage, DAG audit, DDR, cold-start strata, ground-truth cross-validation, split protocol). Đã có test (`tests/test_p0_bridge.py`) chạy qua bản vendor thật của P0.
- `dh2a_kt/hyperedge/construction.py::build_concept_prerequisite_hyperedges` — gom `E_pre` đã audit thành hyperedge chuỗi tiên quyết. Có test, chạy được trên dữ liệu toy.
- `dh2a_kt/hyperedge/audit.py` — mở rộng 4 chỉ số leakage của P0 sang hyperedge + 3 lớp leakage mới (group-membership, cross-modal, intervention-timing) theo đúng mục 2 của `docs/idea-D-plan.md`. `compute_group_membership_leakage` chạy thật, có test; `compute_cross_modal_leakage`/`compute_intervention_timing_leakage` chạy được nhưng cần input mà chưa có nguồn dữ liệu thật (Forum/Teacher).
- `dh2a_kt/baselines/loader.py` — nạp trực tiếp `baselines/p0_reused/*.csv` (69 dòng tổng hợp + 205 dòng theo fold, copy nguyên từ kết quả đã công bố của P0). Đây là phần **tiết kiệm compute lớn nhất**: không cần train lại BKT/DKT/AKT/simpleKT/GKT/GIKT/SKT/DyGKT/DGEKT trên RTX 3090. Chạy được, có test.
- `dh2a_kt/models/causal_layer.py::PropensityScoreATE` — ước lượng ATE bằng IPW, đúng như kế hoạch "bắt đầu bằng propensity-score/2-stage regression" (Pha 2). Implementation thật, không phải stub. Có test (`tests/test_causal_layer.py`, dữ liệu tổng hợp với ground-truth ATE đã biết) xác nhận ước lượng IPW sát effect thật hơn naive mean-difference, và cảnh báo overlap kém hoạt động đúng.
- `dh2a_kt/eval/manipulation_check.py` — tái dùng operator DDR của P0 (`apply_node_drop`/`apply_edge_drop`) để làm phép thử "model có thực sự đọc hyperedge không", theo đúng cách P0 tự áp dụng ở Section 4.7. Cần một `eval_auc_fn` (Tier 1 đã train) mới chạy hết được — phần còn thiếu chính là Tier 1.

Những gì **cố tình để là interface stub / `NotImplementedError`** (không giả vờ đã xong):

- `dh2a_kt/models/dh2_kt.py::DH2KT.forward()` — kiến trúc HypergraphConv/HeteroConv thật sự (Pha 2). Đây là công việc nhiều tuần theo đúng kế hoạch, không thể "tạo" trong một phiên.
- `dh2a_kt/agents/*` — 5 agent Tier 2 (Diagnostician, Tutor/Hint, Critic, Forum/Social, Curriculum Planner): interface (prompt, contract dữ liệu vào/ra) đã cố định, nhưng cần một `LLMClient` backend thật (local quantized hoặc API) mới chạy được (Pha 4).
- `dh2a_kt/hyperedge/construction.py::build_session_hyperedges` / `build_discussion_thread_hyperedges` / `build_teacher_intervention_hyperedges` — chặn bởi thiếu dữ liệu Hint/Video/Forum/Teacher đồng bộ, đúng như rủi ro đã nêu trong mọi phiên bản ý tưởng (A/B/C/D).
- `scripts/02_build_hyperedges.py`, `03_train_tier1.py`, `04_run_tier2_pilot.py` — driver script nối các phần trên lại; phần logic từng khối đã có sẵn trong `dh2a_kt/`, phần glue-code đọc dữ liệu thật từ P0 preprocessing để lại làm TODO tường minh (không fabricate đường dẫn dữ liệu).

## Provenance — P0 được đưa vào dự án như thế nào

`external/p0_leakage_audit/` là **bản sao tĩnh (static vendored snapshot)** của repo P0 gốc, pin đúng commit `5e3d6a01c32a649f1655ac8712f03359bddfe478` (xem `external/p0_leakage_audit/.p0_vendored_commit.txt`).

Ban đầu định làm `git submodule` thật (idiomatic hơn, giống cách chính P0 vendor `pykt-toolkit`), nhưng hệ thống file của thư mục dự án trong môi trường dựng scaffold này **không hỗ trợ unlink/relink** mà cơ chế submodule của git cần — thao tác `git submodule add` thất bại giữa chừng. Trên máy thật của bạn (Windows, không có giới hạn này), bạn có thể chuyển sang submodule sống:

```bash
rm -rf external/p0_leakage_audit
git submodule add https://github.com/edu-risk-lab/leakage-controlled-kt-audit.git external/p0_leakage_audit
git submodule update --init --recursive
```

Có 2 file rác còn sót lại từ lần thử submodule thất bại (`external/_failed_submodule_gitlink.txt`, `external/_writetest2.txt`) — vô hại, đã gitignore, không xoá được trong môi trường dựng scaffold nhưng bạn có thể xoá tay trên máy mình.

## Cách chạy

```bash
# 1. Cài dependency (DH2A-KT + P0 + pykt-toolkit) và dựng khung thư mục output
bash scripts/00_setup_p0_submodule.sh

# 2. Xác nhận config của D đang trỏ đúng vào P0 đã vendor (bắt buộc trước khi tin baseline reuse)
python scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml

# 3. Tải dữ liệu thô theo đúng hướng dẫn của P0 (không tự viết pipeline khác)
#    xem external/p0_leakage_audit/README.md muc 3 "Data download and preparation"
#    (bao gồm link Google Drive bundle P0 đã chia sẻ)

# 4. Tiền xử lý bằng chính code P0 (không phải bản sao khác)
cd external/p0_leakage_audit
python -m src.preprocess --config configs/xes3g5m.yaml
cd ../..

# 5. Chạy test để chắc scaffold hoạt động đúng
pytest tests/ -v
```

Checklist đầy đủ cho việc push lên GitHub và train trên GPU server khác máy
(môi trường cần gì, commit/checksum nào phải khớp, thư mục nào không được
commit) nằm ở [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
Bảng/hình trong bản thảo KBS map sang script ở [`docs/paper-artifacts.md`](docs/paper-artifacts.md).

## Cấu trúc dự án

```
DH2A_KT/
├── docs/idea-D-plan.md          # ke hoach day du (kien truc, RQ, timeline, rui ro)
├── external/p0_leakage_audit/   # P0 vendored (pinned commit, xem .p0_vendored_commit.txt)
├── baselines/p0_reused/         # baseline AUC/ACC/NLL da cong bo cua P0 (khong train lai)
├── configs/                     # config XES3G5M (primary) / ASSISTments2012 / Junyi cho D
├── dh2a_kt/
│   ├── p0_bridge.py             # import truc tiep code goc cua P0
│   ├── hyperedge/                # dong gop moi cua D: xay + audit hyperedge
│   ├── models/                  # Tier 1: DH2-KT (stub) + causal layer (that)
│   ├── agents/                  # Tier 2: AgentDG-KT pilot (interface stub)
│   ├── baselines/                # nap lai ket qua P0 (that, chay duoc)
│   └── eval/                    # manipulation-check kieu DDR
├── scripts/                     # driver theo tung Pha cua ke hoach
├── results/{tables,figures}/    # output that (gitignored, khung thu muc da co)
├── checkpoints/                  # trong so model (gitignored)
├── logs/                          # log chay (gitignored)
└── tests/
```

## Vì sao package tên `dh2a_kt` chứ không phải `src`

P0 dùng tên package top-level `src` trong toàn bộ import nội bộ (`from src.io_utils import ...`). Nếu DH2A-KT cũng đặt tên `src`, hai package sẽ đè lên nhau trên `sys.path` khi cùng được import trong một tiến trình Python. Đặt tên `dh2a_kt` để hai bên tồn tại song song an toàn — xem chi tiết trong docstring của `dh2a_kt/p0_bridge.py`.
