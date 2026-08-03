"""Train/evaluate neural baselines via pyKT with P0 protocol graphs (optional dependency)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
_PYKT_CPU_PATCHED = False


def _patch_pykt_cpu_tensors() -> None:
    """pyKT KTDataset imports LongTensor from torch.cuda; patch for CPU-only PyTorch."""
    global _PYKT_CPU_PATCHED
    if _PYKT_CPU_PATCHED:
        return
    import pykt.datasets.data_loader as dl

    dl.LongTensor = torch.LongTensor
    dl.FloatTensor = torch.FloatTensor
    _PYKT_CPU_PATCHED = True


def _mean_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-4) -> float:
    y = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _batch_to_device(dcur: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in dcur.items()
    }


def _dataloader_kwargs(batch_size: int, force_cpu: bool = False) -> dict:
    use_cuda = torch.cuda.is_available() and not force_cpu
    # FORCE num_workers=0 to prevent multiprocessing deadlock on Windows
    num_workers = 0
    kw: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": False,
        "multiprocessing_context": None,
    }
    return kw


def _bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE must run in full precision — unsafe under autocast (PyTorch 2.11+)."""
    from torch.nn.functional import binary_cross_entropy

    with torch.amp.autocast("cuda", enabled=False):
        return binary_cross_entropy(pred.float(), target.float())


def _model_forward_loss(model, batch: dict, model_name: str) -> torch.Tensor:
    from torch.nn.functional import one_hot

    device = next(model.parameters()).device
    dcur = _batch_to_device(batch, device)
    q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
    qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
    sm = dcur["smasks"]
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if model_name == "dkt":
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    if model_name == "akt":
        y, reg = model(cc.long(), cr.long(), cq.long())
        y = y[:, 1:]
        pred = torch.masked_select(y, sm)
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target) + reg
    if model_name == "gkt":
        y = model(cc.long(), cr.long())
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    if model_name == "simplekt":
        y, _y2, _y3 = model(dcur, train=True)
        y = y[:, 1:]
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    if model_name == "gikt":
        y = model(cq.long(), cc.long(), cr.long())
        y = y[:, 1:]
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    if model_name == "sakt":
        y = model(c.long(), r.long(), cshft.long())
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    if model_name in ("skt", "dygkt", "dgekt"):
        y = model(c.long(), r.long(), cshft.long())
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return _bce_loss(pred, target)
    raise ValueError(f"Unsupported model_name={model_name}")


def _evaluate_detailed(
    model,
    loader,
    model_name: str,
    uid_path: str | None = None,
    *,
    uid_fold: int = -1,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    model.eval()
    y_trues, y_scores = [], []
    uids_out = []
    kcs_out = []
    
    pt_device = "cpu" if (os.environ.get("FORCE_CPU", "0") == "1" or not torch.cuda.is_available()) else "cuda"
    dev = torch.device(pt_device)
    from torch.nn.functional import one_hot
    
    uids = None
    if uid_path is not None:
        import pandas as pd
        df_uid = pd.read_csv(uid_path)
        df_uid = df_uid[df_uid["fold"] == uid_fold]
        uids = df_uid["uid"].values
        
    batch_idx = 0

    with torch.no_grad():
        for data in loader:
            dcur = _batch_to_device(data, dev)
            q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
            qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
            sm = dcur["smasks"]

            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)

            if model_name == "dkt":
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "akt":
                y, _reg = model(cc.long(), cr.long(), cq.long())
                y = torch.clamp(y[:, 1:], 0.0, 1.0)
            elif model_name == "simplekt":
                preds = model(dcur, train=False)
                y = preds[:, 1:]
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
            elif model_name == "gikt":
                y = model(cq.long(), cc.long(), cr.long())
                y = y[:, 1:]
            elif model_name == "sakt":
                y = model(c.long(), r.long(), cshft.long())
            elif model_name in ("skt", "dygkt", "dgekt"):
                y = model(c.long(), r.long(), cshft.long())
            else:
                raise ValueError(f"Unsupported pyKT model_name={model_name}")

            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach().cpu()
            c_sel = torch.masked_select(cshft, sm).detach().cpu()
            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            kcs_out.append(c_sel.numpy())
            
            if uids is not None:
                batch_size = y.shape[0] if y.dim() > 0 else 1 # Not quite right for masked_select
                # Masked select flattens it.
                # sm shape is [batch_size, seq_len-1].
                batch_uids = uids[batch_idx * loader.batch_size : batch_idx * loader.batch_size + sm.shape[0]]
                batch_uids_tensor = torch.tensor(batch_uids, dtype=torch.int64).unsqueeze(1).expand(-1, sm.shape[1])
                uid_selected = torch.masked_select(batch_uids_tensor, sm.cpu())
                uids_out.append(uid_selected.numpy())
            batch_idx += 1

    ts = np.concatenate(y_trues, axis=0)
    ps = np.concatenate(y_scores, axis=0)
    cs = np.concatenate(kcs_out, axis=0)
    us = np.concatenate(uids_out, axis=0) if uids_out else None
    
    if len(np.unique(ts)) < 2:
        auc = float("nan")
    else:
        auc = float(metrics.roc_auc_score(ts, ps))
    acc = float(metrics.accuracy_score(ts, (ps >= 0.5).astype(int)))
    return auc, acc, ts, ps, us, cs


def _train_loop(model, train_loader, valid_loader, epochs: int, lr: float, patience: int = 3) -> None:
    # GKT has internal float32/float16 dtype conflicts with autocast — disable AMP for it
    # Also disable AMP on CPU (GradScaler requires CUDA)
    _force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    # Models excluded from AMP due to float16 overflow/dtype issues:
    # - gkt: float32/float16 conflict in _agg_neighbors scatter
    # - simplekt/gikt/akt: masked_fill(-1e32) overflows float16
    # - dygkt/dgekt: in-place index_put dtype mismatch under autocast
    _NO_AMP_MODELS = {'gkt', 'simplekt', 'gikt', 'akt', 'dygkt', 'dgekt'}
    use_amp = torch.cuda.is_available() and not _force_cpu and getattr(model, 'model_name', '') not in _NO_AMP_MODELS
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    torch.backends.cudnn.benchmark = True  # autotuning for faster kernels
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = -1.0
    stale = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for data in train_loader:
            opt.zero_grad(set_to_none=True)  # faster than zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = _model_forward_loss(model, data, model.model_name)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        tr_loss = float(np.mean(losses)) if losses else 0.0
        auc, acc, _, _, _, _ = _evaluate_detailed(model, valid_loader, model.model_name)
        logger.info("pyKT epoch %s train_loss=%.5f valid_auc=%.5f valid_acc=%.5f", ep, tr_loss, auc, acc)
        if auc > best_auc + 1e-4:
            best_auc = auc
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)


def run_pykt_fold(
    *,
    display_model: str,
    pykt_name: str,
    work_dir: Path,
    num_q: int,
    num_c: int,
    graph_npz: Path | None,
    graph_tag: str,
    hyperparams: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    max_seq_len: int,
    include_valid_predictions: bool = False,
) -> tuple[float, float, float, str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Train on fold 0 / validate on fold 1 rows inside ``train_valid_sequences.csv``; eval fold -1 test file."""
    import shutil

    _patch_pykt_cpu_tensors()
    from pykt.datasets.data_loader import KTDataset
    from pykt.models.init_model import init_model

    work_dir.mkdir(parents=True, exist_ok=True)
    for p in work_dir.glob("*.pkl"):
        try:
            p.unlink()
        except OSError:
            pass

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_type = ["questions", "concepts"]
    tv_path = str(work_dir / "train_valid_sequences.csv")
    te_path = str(work_dir / "test_sequences.csv")

    train_ds = KTDataset(tv_path, input_type, {0})
    valid_ds = KTDataset(tv_path, input_type, {1})
    eval_ds = KTDataset(te_path, input_type, {-1})

    loader_kw = _dataloader_kwargs(batch_size)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kw)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kw)

    emb_type = "qid"
    data_cfg = {
        "dpath": str(work_dir),
        "num_q": int(num_q),
        "num_c": int(num_c),
        "emb_path": "",
        "train_valid_original_file": "train_valid_sequences.csv",
        "test_original_file": "test_sequences.csv",
    }

    if pykt_name == "dkt":
        emb_size = int(hyperparams.get("emb_size", hyperparams.get("hidden_dim", 100)))
        dropout = float(hyperparams.get("dropout", 0.2))
        model_cfg = {"emb_size": emb_size, "dropout": dropout}
    elif pykt_name == "akt":
        model_cfg = {
            "d_model": int(hyperparams.get("d_model", 256)),
            "n_blocks": int(hyperparams.get("n_blocks", 1)),
            "dropout": float(hyperparams.get("dropout", 0.05)),
            "d_ff": int(hyperparams.get("d_ff", 1024)),
            "kq_same": int(hyperparams.get("kq_same", 1)),
            "final_fc_dim": int(hyperparams.get("final_fc_dim", 512)),
            "num_attn_heads": int(hyperparams.get("num_attn_heads", 8)),
            "separate_qa": bool(hyperparams.get("separate_qa", False)),
            "l2": float(hyperparams.get("l2", 1e-5)),
        }
    elif pykt_name == "gkt":
        if graph_npz is None:
            raise ValueError("GKT requires graph_npz")
        dest = work_dir / f"gkt_graph_{graph_tag}.npz"
        if graph_npz.resolve() != dest.resolve():
            shutil.copy(graph_npz, dest)
        model_cfg = {
            "hidden_dim": int(hyperparams.get("hidden_dim", 100)),
            "emb_size": int(hyperparams.get("emb_size", 100)),
            "dropout": float(hyperparams.get("dropout", 0.5)),
            "graph_type": graph_tag,
        }
    elif pykt_name == "simplekt":
        model_cfg = {
            "d_model": int(hyperparams.get("d_model", 256)),
            "n_blocks": int(hyperparams.get("n_blocks", 4)),
            "dropout": float(hyperparams.get("dropout", 0.05)),
            "d_ff": int(hyperparams.get("d_ff", 1024)),
            "num_attn_heads": int(hyperparams.get("num_attn_heads", 8)),
            "kq_same": int(hyperparams.get("kq_same", 1)),
            "l2": float(hyperparams.get("l2", 1e-4)),
            "seq_len": int(max_seq_len),
            "final_fc_dim": int(hyperparams.get("final_fc_dim", 512)),
            "final_fc_dim2": int(hyperparams.get("final_fc_dim2", 256)),
            "separate_qa": bool(hyperparams.get("separate_qa", False)),
        }
    elif pykt_name == "gikt":
        # Load bipartite edges from gikt_bipartite.csv
        bipartite_edges = []
        bip_path = work_dir / "gikt_bipartite.csv"
        if bip_path.exists():
            import csv
            with open(bip_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        bipartite_edges.append((int(row[0]), int(row[1])))
        emb_size = int(hyperparams.get("emb_size", 64))
        hidden_dim = int(hyperparams.get("hidden_dim", 128))
        from src.models.gikt import GIKTPyTorch
        model = GIKTPyTorch(
            num_q=int(num_q),
            num_c=int(num_c),
            emb_size=emb_size,
            hidden_dim=hidden_dim,
            bipartite_edges=bipartite_edges,
        )
        if torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
            model = model.cuda()
    elif pykt_name in ("skt", "dygkt", "dgekt"):
        if graph_npz is not None and os.path.exists(graph_npz):
            adj_matrix = torch.tensor(np.load(graph_npz, allow_pickle=True)['matrix']).float()
        else:
            adj_matrix = torch.eye(int(num_c)).float()
        if torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
            adj_matrix = adj_matrix.cuda()

        emb_size = int(hyperparams.get("emb_size", 64))
        hidden_dim = int(hyperparams.get("hidden_dim", 128))

        if pykt_name == "skt":
            from src.models.skt import SKTPyTorch
            beta = float(hyperparams.get("beta", 0.1))
            model = SKTPyTorch(
                num_c=int(num_c),
                emb_size=emb_size,
                hidden_dim=hidden_dim,
                adj_matrix=adj_matrix,
                beta=beta,
            )
        elif pykt_name == "dygkt":
            from src.models.dygkt import DyGKTPyTorch
            gamma = float(hyperparams.get("gamma", 0.1))
            model = DyGKTPyTorch(
                num_c=int(num_c),
                emb_size=emb_size,
                hidden_dim=hidden_dim,
                adj_matrix=adj_matrix,
                gamma=gamma,
            )
        elif pykt_name == "dgekt":
            from src.models.dgekt import DGEKTPyTorch
            beta = float(hyperparams.get("beta", 0.1))
            model = DGEKTPyTorch(
                num_c=int(num_c),
                emb_size=emb_size,
                hidden_dim=hidden_dim,
                adj_matrix=adj_matrix,
                beta=beta,
            )
        if torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
            model = model.cuda()
    elif pykt_name == "sakt":
        model_cfg = {
            "seq_len": int(max_seq_len),
            "emb_size": int(hyperparams.get("emb_size", 100)),
            "num_attn_heads": int(hyperparams.get("num_attn_heads", 5)),
            "dropout": float(hyperparams.get("dropout", 0.2)),
            "num_en": int(hyperparams.get("num_en", 2)),
        }
    else:
        raise ValueError(f"Unknown pyKT name={pykt_name}")

    if pykt_name not in ("gikt", "skt", "dygkt", "dgekt"):
        model = init_model(pykt_name, model_cfg, data_cfg, emb_type)
        if model is None:
            raise RuntimeError(f"pyKT init_model returned None for {pykt_name}")
        if torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1":
            model = model.cuda()

    note = (
        f"pyKT `{pykt_name}` trained on learner-split train users; metrics on valid+test sequence positions. "
        "GKT adjacency from P0 exported graphs."
    )
    if display_model != pykt_name:
        note += f" YAML alias `{display_model}` maps to `{pykt_name}` (GIKT not bundled in pyKT)."

    ckpt_path = work_dir / f"{pykt_name}_{graph_tag}_best.ckpt"
    if ckpt_path.exists():
        logger.info(f"Loading existing checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        patience = int(hyperparams.get("patience", 3))
        _train_loop(model, train_loader, valid_loader, epochs=max(1, int(epochs)), lr=float(lr), patience=patience)
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved PyKT checkpoint to {ckpt_path}")
        
    auc, acc, ts, ps, us, cs = _evaluate_detailed(
        model, eval_loader, model.model_name, uid_path=te_path, uid_fold=-1
    )
    if include_valid_predictions:
        _, _, ts_v, ps_v, us_v, cs_v = _evaluate_detailed(
            model, valid_loader, model.model_name, uid_path=tv_path, uid_fold=1
        )
        ts = np.concatenate([ts_v, ts])
        ps = np.concatenate([ps_v, ps])
        cs = np.concatenate([cs_v, cs])
        if us is not None and us_v is not None:
            us = np.concatenate([us_v, us])
    nll = _mean_nll(ts, ps)

    # Explicitly release GPU memory to prevent Out of Memory in sequential baseline runs
    import gc
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return auc, acc, nll, note, ts, ps, us, cs
