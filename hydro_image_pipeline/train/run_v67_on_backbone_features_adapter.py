from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / 'data_refs'
INTER = PACKAGE_ROOT / 'intermediate'
RESULT = PACKAGE_ROOT / 'results'
for p in [INTER, RESULT]:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _status_line(msg: str, done: bool = False) -> None:
    print(msg, flush=True)


def load_df(path: Path, apply_year_filter: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
    df = df[df['discharge'].notna()].copy()
    df = df[df['discharge'] != -999999].copy()
    if apply_year_filter:
        df = df[df['year'].isin([2024, 2025])].copy()
    df['discharge_log'] = np.log1p(df['discharge'])
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('emb_') and pd.api.types.is_numeric_dtype(df[c])]


def _season_from_month(month: pd.Series) -> pd.Series:
    m = month.astype(int)
    return (((m % 12) // 3) + 1).astype(int)


def build_metadata_array(df: pd.DataFrame, metadata_fields: list[str]) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    used_fields: list[str] = []
    if 'month' in metadata_fields:
        month = df['month'].fillna(-1).astype(int).clip(1, 12)
        month_oh = np.zeros((len(df), 12), dtype=np.float32)
        valid = (month >= 1) & (month <= 12)
        month_idx = month[valid].to_numpy() - 1
        month_oh[np.where(valid)[0], month_idx] = 1.0
        parts.append(month_oh)
        used_fields.append('month')
    if 'season' in metadata_fields:
        season = _season_from_month(df['month'].fillna(-1))
        season_oh = np.zeros((len(df), 4), dtype=np.float32)
        valid = (season >= 1) & (season <= 4)
        season_idx = season[valid].to_numpy() - 1
        season_oh[np.where(valid)[0], season_idx] = 1.0
        parts.append(season_oh)
        used_fields.append('season')
    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32), []
    return np.concatenate(parts, axis=1), used_fields


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'Bias': float(np.mean(y_pred - y_true)),
    }


def format_years(df: pd.DataFrame | None) -> str:
    if df is None or 'year' not in df.columns:
        return ''
    years = pd.to_numeric(df['year'], errors='coerce').dropna().astype(int).unique().tolist()
    return '|'.join(str(y) for y in sorted(years))


def build_ensemble_weights(val_r2s: list[float], mode: str, weight_power: float, topk: int) -> np.ndarray:
    arr = np.array(val_r2s, dtype=float)
    if len(arr) == 0:
        return arr
    if mode == 'uniform':
        return np.ones_like(arr, dtype=float) / float(len(arr))

    if mode == 'topk':
        k = max(1, min(int(topk), len(arr)))
        order = np.argsort(arr)[::-1]
        keep = order[:k]
        kept = np.array([max(arr[i], 0.0) + 1e-6 for i in keep], dtype=float)
        kept = np.power(kept, float(weight_power))
        kept = kept / np.sum(kept)
        w = np.zeros_like(arr, dtype=float)
        w[keep] = kept
        return w

    base = np.array([max(v, 0.0) + 1e-6 for v in arr], dtype=float)
    if mode == 'softmax':
        z = base - np.max(base)
        z = np.exp(float(weight_power) * z)
        return z / np.sum(z)

    base = np.power(base, float(weight_power))
    return base / np.sum(base)


class JointMoE(nn.Module):
    # Original v6.7 head with optional extra gate-only context.
    def __init__(self, in_dim: int, trunk: int = 256, head: int = 128, dropout: float = 0.1, gate_extra_dim: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk, head),
            nn.ReLU(),
        )
        self.low_head = nn.Sequential(nn.Linear(head, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        self.high_head = nn.Sequential(nn.Linear(head, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        gate_in = head + gate_extra_dim
        self.gate_head = nn.Sequential(nn.Linear(gate_in, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))

    def forward(self, x: torch.Tensor, gate_extra: torch.Tensor | None = None):
        h = self.trunk(x)
        low_raw = nn.functional.softplus(self.low_head(h).squeeze(1))
        high_raw = nn.functional.softplus(self.high_head(h).squeeze(1))
        gate_in = h if gate_extra is None else torch.cat([h, gate_extra], dim=1)
        gate_logit = self.gate_head(gate_in).squeeze(1)
        gate = torch.sigmoid(gate_logit)
        mix_raw = (1.0 - gate) * low_raw + gate * high_raw
        return low_raw, high_raw, gate_logit, gate, mix_raw


class DisentangledJointMoE(nn.Module):
    # Stronger core: low/high experts and gate learn separate representations from the same adapted input.
    def __init__(self, in_dim: int, trunk: int = 256, head: int = 128, dropout: float = 0.1, gate_extra_dim: int = 0):
        super().__init__()
        self.low_trunk = nn.Sequential(
            nn.Linear(in_dim, trunk),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk, head),
            nn.ReLU(),
        )
        self.high_trunk = nn.Sequential(
            nn.Linear(in_dim, trunk),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk, head),
            nn.ReLU(),
        )
        self.gate_trunk = nn.Sequential(
            nn.Linear(in_dim, trunk),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk, head),
            nn.ReLU(),
        )
        self.low_head = nn.Sequential(nn.Linear(head, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        self.high_head = nn.Sequential(nn.Linear(head, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        gate_in = head + gate_extra_dim
        self.gate_head = nn.Sequential(nn.Linear(gate_in, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))

    def forward(self, x: torch.Tensor, gate_extra: torch.Tensor | None = None):
        h_low = self.low_trunk(x)
        h_high = self.high_trunk(x)
        h_gate = self.gate_trunk(x)
        low_raw = nn.functional.softplus(self.low_head(h_low).squeeze(1))
        high_raw = nn.functional.softplus(self.high_head(h_high).squeeze(1))
        gate_in = h_gate if gate_extra is None else torch.cat([h_gate, gate_extra], dim=1)
        gate_logit = self.gate_head(gate_in).squeeze(1)
        gate = torch.sigmoid(gate_logit)
        mix_raw = (1.0 - gate) * low_raw + gate * high_raw
        return low_raw, high_raw, gate_logit, gate, mix_raw


class IdentityAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualMLPProjector(nn.Module):
    # Optional pre/post LN + residual scaling
    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout: float,
        ln_mode: str = 'prepost',
        alpha_init: float = 1.0,
        alpha_learnable: bool = False,
    ):
        super().__init__()
        self.ln_mode = ln_mode
        self.ln_in = nn.LayerNorm(dim) if ln_mode in ('pre', 'prepost') else nn.Identity()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln_out = nn.LayerNorm(dim) if ln_mode in ('post', 'prepost') else nn.Identity()
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln_in(x)
        z = self.fc2(self.drop(self.act(self.fc1(z))))
        return self.ln_out(x + self.alpha * z)


class GatedResidualMLPProjector(nn.Module):
    # LN -> shared input -> delta branch and sample-wise gate branch -> residual -> optional post LN
    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout: float,
        ln_mode: str = 'prepost',
        alpha_init: float = 1.0,
        alpha_learnable: bool = False,
    ):
        super().__init__()
        self.ln_mode = ln_mode
        self.ln_in = nn.LayerNorm(dim) if ln_mode in ('pre', 'prepost') else nn.Identity()
        self.delta_fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.delta_fc2 = nn.Linear(hidden, dim)
        self.gate_fc = nn.Linear(dim, dim)
        self.ln_out = nn.LayerNorm(dim) if ln_mode in ('post', 'prepost') else nn.Identity()
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_in(x)
        delta = self.delta_fc2(self.drop(self.act(self.delta_fc1(h))))
        gate = torch.sigmoid(self.gate_fc(h))
        return self.ln_out(x + self.alpha * gate * delta)


class DeeperResidualMLPProjector(nn.Module):
    # LN -> 2-layer hidden MLP -> residual -> optional post LN
    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout: float,
        ln_mode: str = 'prepost',
        alpha_init: float = 1.0,
        alpha_learnable: bool = False,
    ):
        super().__init__()
        self.ln_mode = ln_mode
        self.ln_in = nn.LayerNorm(dim) if ln_mode in ('pre', 'prepost') else nn.Identity()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(dim) if ln_mode in ('post', 'prepost') else nn.Identity()
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_in(x)
        z = self.drop(self.act(self.fc1(h)))
        z = self.drop(self.act(self.fc2(z)))
        z = self.fc3(z)
        return self.ln_out(x + self.alpha * z)


class BottleneckAdapter(nn.Module):
    # Linear -> GELU -> Dropout -> Linear -> residual (+ optional pre/post LN)
    def __init__(
        self,
        dim: int,
        bottleneck: int,
        dropout: float,
        ln_mode: str = 'post',
        alpha_init: float = 1.0,
        alpha_learnable: bool = False,
    ):
        super().__init__()
        self.ln_mode = ln_mode
        self.ln_in = nn.LayerNorm(dim) if ln_mode in ('pre', 'prepost') else nn.Identity()
        self.fc1 = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.ln_out = nn.LayerNorm(dim) if ln_mode in ('post', 'prepost') else nn.Identity()
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc2(self.drop(self.act(self.fc1(self.ln_in(x)))))
        return self.ln_out(x + self.alpha * z)


class AdapterMoE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        metadata_dim: int,
        metadata_hidden: int,
        trunk: int,
        head: int,
        dropout: float,
        adapter_type: str,
        adapter_hidden: int,
        adapter_bottleneck: int,
        adapter_dropout: float,
        adapter_ln_mode: str,
        adapter_alpha_init: float,
        adapter_alpha_learnable: bool,
        metadata_mode: str,
        core_type: str,
    ):
        super().__init__()
        fused_in_dim = in_dim
        self.metadata_encoder = None
        self.gate_metadata_encoder = None
        gate_extra_dim = 0
        if metadata_dim > 0:
            if metadata_mode == 'concat':
                self.metadata_encoder = nn.Sequential(
                    nn.Linear(metadata_dim, metadata_hidden),
                    nn.GELU(),
                )
                fused_in_dim += metadata_hidden
            elif metadata_mode == 'gate_only':
                self.gate_metadata_encoder = nn.Sequential(
                    nn.Linear(metadata_dim, metadata_hidden),
                    nn.GELU(),
                )
                gate_extra_dim = metadata_hidden
            else:
                raise ValueError(f'unsupported metadata_mode: {metadata_mode}')
        if adapter_type == 'none':
            self.adapter = IdentityAdapter()
        elif adapter_type == 'res_mlp':
            self.adapter = ResidualMLPProjector(
                dim=fused_in_dim,
                hidden=adapter_hidden,
                dropout=adapter_dropout,
                ln_mode=adapter_ln_mode,
                alpha_init=adapter_alpha_init,
                alpha_learnable=adapter_alpha_learnable,
            )
        elif adapter_type == 'gated_res_mlp':
            self.adapter = GatedResidualMLPProjector(
                dim=fused_in_dim,
                hidden=adapter_hidden,
                dropout=adapter_dropout,
                ln_mode=adapter_ln_mode,
                alpha_init=adapter_alpha_init,
                alpha_learnable=adapter_alpha_learnable,
            )
        elif adapter_type == 'deeper_res_mlp':
            self.adapter = DeeperResidualMLPProjector(
                dim=fused_in_dim,
                hidden=adapter_hidden,
                dropout=adapter_dropout,
                ln_mode=adapter_ln_mode,
                alpha_init=adapter_alpha_init,
                alpha_learnable=adapter_alpha_learnable,
            )
        elif adapter_type == 'bottleneck':
            self.adapter = BottleneckAdapter(
                dim=fused_in_dim,
                bottleneck=adapter_bottleneck,
                dropout=adapter_dropout,
                ln_mode=adapter_ln_mode,
                alpha_init=adapter_alpha_init,
                alpha_learnable=adapter_alpha_learnable,
            )
        else:
            raise ValueError(f'unsupported adapter_type: {adapter_type}')
        if core_type == 'shared':
            self.core = JointMoE(in_dim=fused_in_dim, trunk=trunk, head=head, dropout=dropout, gate_extra_dim=gate_extra_dim)
        elif core_type == 'disentangled':
            self.core = DisentangledJointMoE(
                in_dim=fused_in_dim,
                trunk=trunk,
                head=head,
                dropout=dropout,
                gate_extra_dim=gate_extra_dim,
            )
        else:
            raise ValueError(f'unsupported core_type: {core_type}')

    def forward(self, x: torch.Tensor, x_meta: torch.Tensor | None = None):
        gate_extra = None
        if self.metadata_encoder is not None and x_meta is not None:
            meta = self.metadata_encoder(x_meta)
            x = torch.cat([x, meta], dim=1)
        if self.gate_metadata_encoder is not None and x_meta is not None:
            gate_extra = self.gate_metadata_encoder(x_meta)
        x2 = self.adapter(x)
        return self.core(x2, gate_extra=gate_extra)


def clamp_raw_pred(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=20000.0, neginf=0.0)
    return torch.clamp(x, min=0.0, max=20000.0)


@dataclass
class TrainOutput:
    model: AdapterMoE
    best_epoch: int
    best_val_r2: float


def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    X_meta: np.ndarray | None = None,
) -> DataLoader:
    meta = np.zeros((len(X), 0), dtype=np.float32) if X_meta is None else X_meta
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(meta, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    X_fit: np.ndarray,
    X_meta_fit: np.ndarray | None,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    X_meta_val: np.ndarray | None,
    y_val: np.ndarray,
    thr: float,
    seed: int,
    lr: float,
    wd: float,
    batch_size: int,
    epochs: int,
    patience: int,
    gate_aux: float,
    expert_aux: float,
    gstar_aux: float,
    high_weight: float,
    trunk_dim: int,
    head_dim: int,
    dropout: float,
    adapter_type: str,
    adapter_hidden: int,
    adapter_bottleneck: int,
    adapter_dropout: float,
    adapter_ln_mode: str,
    adapter_alpha_init: float,
    adapter_alpha_learnable: bool,
    metadata_hidden: int,
    metadata_mode: str,
    core_type: str,
    device: torch.device,
    dataset_name: str,
    seed_idx: int,
    n_seeds: int,
) -> TrainOutput:
    set_seed(seed)
    _status_line(
        f'[{dataset_name}] seed {seed_idx}/{n_seeds} ({seed}) | '
        f'initializing | n_fit={len(X_fit)} n_val={len(X_val)}'
    )
    model = AdapterMoE(
        in_dim=X_fit.shape[1],
        metadata_dim=0 if X_meta_fit is None else X_meta_fit.shape[1],
        metadata_hidden=metadata_hidden,
        trunk=trunk_dim,
        head=head_dim,
        dropout=dropout,
        adapter_type=adapter_type,
        adapter_hidden=adapter_hidden,
        adapter_bottleneck=adapter_bottleneck,
        adapter_dropout=adapter_dropout,
        adapter_ln_mode=adapter_ln_mode,
        adapter_alpha_init=adapter_alpha_init,
        adapter_alpha_learnable=adapter_alpha_learnable,
        metadata_mode=metadata_mode,
        core_type=core_type,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    fit_ld = _to_loader(X_fit, y_fit, batch_size=batch_size, shuffle=True, X_meta=X_meta_fit)
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
    val_meta = None if X_meta_val is None else torch.tensor(X_meta_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)

    pos = float(np.sum(y_fit > thr))
    neg = float(np.sum(y_fit <= thr))
    pos_weight = torch.tensor([(neg + 1.0) / (pos + 1.0)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {'r2': -1e9, 'epoch': 0, 'state': None}
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_seen = 0
        for xb, xb_meta, yb in fit_ld:
            xb = xb.to(device)
            xb_meta = xb_meta.to(device)
            yb = yb.to(device)
            yb_log = torch.log1p(torch.clamp(yb, min=0.0))
            seg = (yb > thr).float()

            low_raw, high_raw, gate_logit, gate, mix_raw = model(xb, xb_meta)

            sample_w = 1.0 + high_weight * seg
            reg = nn.functional.smooth_l1_loss(torch.log1p(clamp_raw_pred(mix_raw)), yb_log, reduction='none')
            reg = (reg * sample_w).mean()
            low_mask = seg < 0.5
            high_mask = seg > 0.5
            low_loss = torch.tensor(0.0, device=device)
            high_loss = torch.tensor(0.0, device=device)
            if torch.any(low_mask):
                low_loss = nn.functional.smooth_l1_loss(torch.log1p(clamp_raw_pred(low_raw[low_mask])), yb_log[low_mask])
            if torch.any(high_mask):
                high_loss = nn.functional.smooth_l1_loss(torch.log1p(clamp_raw_pred(high_raw[high_mask])), yb_log[high_mask])
            gate_loss = bce(gate_logit, seg)
            denom = high_raw - low_raw
            g_star = torch.where(
                torch.abs(denom) < 1e-5,
                torch.full_like(denom, 0.5),
                (yb - low_raw) / torch.clamp(denom, min=-1e6, max=1e6),
            )
            g_star = torch.clamp(g_star, 0.0, 1.0)
            gstar_per = nn.functional.smooth_l1_loss(gate, g_star, reduction='none')
            gstar_loss = torch.sum(gstar_per * sample_w) / torch.clamp(sample_w.sum(), min=1e-12)
            loss = reg + gate_aux * gate_loss + expert_aux * (low_loss + high_loss) + gstar_aux * gstar_loss
            if not torch.isfinite(loss):
                continue
            batch_n = int(yb.numel())
            epoch_loss_sum += float(loss.detach().cpu()) * batch_n
            epoch_seen += batch_n

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            _, _, _, _, val_mix_raw = model(val_x, val_meta)
            val_pred = clamp_raw_pred(val_mix_raw).cpu().numpy()
            val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=20000.0, neginf=0.0)
            val_r2 = float(r2_score(val_y.cpu().numpy(), val_pred))

        best_r2_show = val_r2 if best['r2'] < -1e8 else best['r2']
        epoch_pct = 100.0 * ep / max(1, epochs)
        overall_pct = 100.0 * ((seed_idx - 1) + (ep / max(1, epochs))) / max(1, n_seeds)
        avg_loss = epoch_loss_sum / max(1, epoch_seen)
        _status_line(
            f'[{dataset_name}] seed {seed_idx}/{n_seeds} ({seed}) | '
            f'epoch {ep}/{epochs} ({epoch_pct:5.1f}%, overall {overall_pct:5.1f}%) | '
            f'loss={avg_loss:.6f} | val_r2={val_r2:.6f} | '
            f'best={best_r2_show:.6f} | no_improve={bad}/{patience}'
        )

        if val_r2 > best['r2']:
            best = {
                'r2': val_r2,
                'epoch': ep,
                'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                overall_pct = 100.0 * ((seed_idx - 1) + (ep / max(1, epochs))) / max(1, n_seeds)
                _status_line(
                    f'[{dataset_name}] seed {seed_idx}/{n_seeds} ({seed}) | '
                    f'early stop at epoch {ep}/{epochs} (overall {overall_pct:5.1f}%) | '
                    f'best_epoch={best["epoch"]} | '
                    f'best_val_r2={best["r2"]:.6f}'
                )
                break

    model.load_state_dict(best['state'])
    _status_line(
        f'[{dataset_name}] seed {seed_idx}/{n_seeds} ({seed}) done | '
        f'best_epoch={best["epoch"]} | best_val_r2={best["r2"]:.6f}',
        done=True,
    )
    return TrainOutput(model=model, best_epoch=int(best['epoch']), best_val_r2=float(best['r2']))


def predict_model(
    model: AdapterMoE,
    X: np.ndarray,
    X_meta: np.ndarray | None,
    device: torch.device,
    batch_size: int = 1024,
):
    model.eval()
    meta = np.zeros((len(X), 0), dtype=np.float32) if X_meta is None else X_meta
    ld = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    low, high, p, soft = [], [], [], []
    with torch.no_grad():
        for xb, xb_meta in ld:
            xb = xb.to(device)
            xb_meta = xb_meta.to(device)
            low_raw, high_raw, _, gate, mix_raw = model(xb, xb_meta)
            low.append(clamp_raw_pred(low_raw).cpu().numpy())
            high.append(clamp_raw_pred(high_raw).cpu().numpy())
            p.append(gate.cpu().numpy())
            soft.append(clamp_raw_pred(mix_raw).cpu().numpy())
    return (
        np.concatenate(low),
        np.concatenate(high),
        np.clip(np.concatenate(p), 0.0, 1.0),
        np.concatenate(soft),
    )


def evaluate_dataset(name: str, df: pd.DataFrame, args, device: torch.device):
    feats = feature_cols(df)
    metadata_fields = [f for f in args.metadata_fields if f in {'month', 'season'}] if args.use_metadata else []
    explicit_split = bool(getattr(args, 'explicit_split', False))

    if explicit_split:
        if 'split' not in df.columns:
            raise ValueError("explicit_split=True but input dataframe has no split column")
        split_norm = df['split'].astype(str).str.lower().str.strip()
        tr_all = df[split_norm == 'train'].copy().reset_index(drop=True)
        val_all = df[split_norm == 'val'].copy().reset_index(drop=True)
        te = df[split_norm == 'test'].copy().reset_index(drop=True)
        if tr_all.empty:
            raise ValueError("explicit split mode requires at least one train row")
        if val_all.empty:
            raise ValueError("explicit split mode requires at least one val row")
        if te.empty:
            raise ValueError("explicit split mode requires at least one test row")
    else:
        tr_all = df[df['year'] == 2024].copy().reset_index(drop=True)
        te = df[df['year'] == 2025].copy().reset_index(drop=True)
        val_all = None

    X_tr_full = tr_all[feats].values
    y_tr_full = tr_all['discharge'].values.astype(float)
    X_te_raw = te[feats].values
    y_te = te['discharge'].values.astype(float)
    X_te_meta, used_metadata_fields = build_metadata_array(te, metadata_fields)

    thr_tr = float(np.percentile(y_tr_full, args.threshold_pct))

    print(
        f'[dataset {name}] start | n_train={len(tr_all)} n_test={len(te)} '
        f'n_features={len(feats)} threshold_pct={args.threshold_pct} thr_train={thr_tr:.4f}',
        flush=True,
    )

    model_preds = []
    val_r2s = []
    seed_soft_r2s = []
    fit_logs = []

    for seed_idx, seed in enumerate(args.seeds, start=1):
        if explicit_split:
            tr_fit = tr_all.copy().reset_index(drop=True)
            tr_val = val_all.copy().reset_index(drop=True)
        else:
            tr_fit, tr_val = train_test_split(tr_all, test_size=args.val_ratio, random_state=int(seed), shuffle=True)
        X_fit = tr_fit[feats].values
        y_fit = tr_fit['discharge'].values.astype(float)
        X_val = tr_val[feats].values
        y_val = tr_val['discharge'].values.astype(float)
        X_fit_meta, _ = build_metadata_array(tr_fit, metadata_fields)
        X_val_meta, _ = build_metadata_array(tr_val, metadata_fields)
        thr_fit = float(np.percentile(y_fit, args.threshold_pct))

        sc_fit = RobustScaler()
        X_fit_s = sc_fit.fit_transform(X_fit)
        X_val_s = sc_fit.transform(X_val)

        out = train_model(
            X_fit_s,
            X_fit_meta if metadata_fields else None,
            y_fit,
            X_val_s,
            X_val_meta if metadata_fields else None,
            y_val,
            thr_fit,
            seed=int(seed),
            lr=args.lr,
            wd=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            gate_aux=args.gate_aux,
            expert_aux=args.expert_aux,
            gstar_aux=args.gstar_aux,
            high_weight=args.high_weight,
            trunk_dim=args.trunk_dim,
            head_dim=args.head_dim,
            dropout=args.dropout,
            adapter_type=args.adapter_type,
            adapter_hidden=args.adapter_hidden,
            adapter_bottleneck=args.adapter_bottleneck,
            adapter_dropout=args.adapter_dropout,
            adapter_ln_mode=args.adapter_ln_mode,
            adapter_alpha_init=args.adapter_alpha_init,
            adapter_alpha_learnable=args.adapter_alpha_learnable,
            metadata_hidden=args.metadata_hidden,
            metadata_mode=args.metadata_mode,
            core_type=args.core_type,
            device=device,
            dataset_name=name,
            seed_idx=seed_idx,
            n_seeds=len(args.seeds),
        )
        val_r2s.append(out.best_val_r2)

        X_te_s = sc_fit.transform(X_te_raw)
        low_te, high_te, p_te, soft_te = predict_model(
            out.model,
            X_te_s,
            X_te_meta if metadata_fields else None,
            device=device,
        )
        model_preds.append({'low': low_te, 'high': high_te, 'p': p_te, 'soft': soft_te})
        seed_soft_r2 = float(r2_score(y_te, soft_te))
        seed_soft_r2s.append(seed_soft_r2)
        print(
            f'[dataset {name}] seed {seed_idx}/{len(args.seeds)} ({seed}) | '
            f'test_soft_r2={seed_soft_r2:.6f}',
            flush=True,
        )

        fit_logs.append(
            {
                'dataset': name,
                'seed': int(seed),
                'val_r2_from_fit': float(out.best_val_r2),
                'best_epoch': int(out.best_epoch),
                'test_soft_r2_seed': seed_soft_r2,
            }
        )

    weights = build_ensemble_weights(
        val_r2s=val_r2s,
        mode=args.ensemble_mode,
        weight_power=args.ensemble_weight_power,
        topk=args.ensemble_topk,
    )

    pred_low = np.sum(np.column_stack([m['low'] for m in model_preds]) * weights, axis=1)
    pred_high = np.sum(np.column_stack([m['high'] for m in model_preds]) * weights, axis=1)
    p_high = np.sum(np.column_stack([m['p'] for m in model_preds]) * weights, axis=1)
    pred_soft = np.sum(np.column_stack([m['soft'] for m in model_preds]) * weights, axis=1)

    seg_pred = (p_high >= 0.5).astype(int)
    seg_true = (y_te > thr_tr).astype(int)
    pred_hard = np.where(seg_pred == 1, pred_high, pred_low)
    pred_seg_oracle = np.where(seg_true == 1, pred_high, pred_low)
    pred_expert_oracle = np.where(np.abs(pred_high - y_te) <= np.abs(pred_low - y_te), pred_high, pred_low)

    hard_m = metrics(y_te, pred_hard)
    soft_m = metrics(y_te, pred_soft)
    seg_o = metrics(y_te, pred_seg_oracle)
    exp_o = metrics(y_te, pred_expert_oracle)

    summary = {
        'dataset': name,
        'mode': 'joint_moe_gstar_adapter',
        'core_type': args.core_type,
        'adapter_type': args.adapter_type,
        'adapter_hidden': int(args.adapter_hidden),
        'adapter_bottleneck': int(args.adapter_bottleneck),
        'adapter_dropout': float(args.adapter_dropout),
        'adapter_ln_mode': args.adapter_ln_mode,
        'adapter_alpha_init': float(args.adapter_alpha_init),
        'adapter_alpha_learnable': bool(args.adapter_alpha_learnable),
        'feature_type': (
            'image_plus_metadata_gate_only'
            if metadata_fields and args.metadata_mode == 'gate_only'
            else ('image_plus_metadata' if metadata_fields else 'image_only')
        ),
        'split_mode': 'explicit_train_val_test' if explicit_split else 'year_train_test_with_random_val',
        'metadata_fields': '|'.join(used_metadata_fields) if used_metadata_fields else '',
        'metadata_hidden': int(args.metadata_hidden if metadata_fields else 0),
        'gate': 'nn_joint_gate_gstar',
        'train_year': '' if explicit_split else 2024,
        'test_year': '' if explicit_split else 2025,
        'train_years': format_years(tr_all),
        'val_years': format_years(val_all if explicit_split else None),
        'test_years': format_years(te),
        'threshold_pct': int(args.threshold_pct),
        'threshold_discharge_train': float(thr_tr),
        'n_train': int(len(tr_all)),
        'n_test': int(len(te)),
        'n_features': int(len(feats) + (X_te_meta.shape[1] if metadata_fields else 0)),
        'n_models': int(len(args.seeds)),
        'ensemble_mode': args.ensemble_mode,
        'ensemble_weight_power': float(args.ensemble_weight_power),
        'ensemble_topk': int(args.ensemble_topk),
        'val_soft_R2': float(np.average(np.array(val_r2s), weights=weights)),
        'seed_test_soft_R2_mean': float(np.mean(seed_soft_r2s)),
        'seed_test_soft_R2_std': float(np.std(seed_soft_r2s)),
        'seed_test_soft_R2_best': float(np.max(seed_soft_r2s)),
        'hard_MAE': hard_m['MAE'],
        'hard_RMSE': hard_m['RMSE'],
        'hard_R2': hard_m['R2'],
        'soft_MAE': soft_m['MAE'],
        'soft_RMSE': soft_m['RMSE'],
        'soft_R2': soft_m['R2'],
        'seg_oracle_R2': seg_o['R2'],
        'seg_oracle_RMSE': seg_o['RMSE'],
        'expert_oracle_R2': exp_o['R2'],
        'expert_oracle_RMSE': exp_o['RMSE'],
    }

    print(
        f'[dataset {name}] complete | soft_R2={summary["soft_R2"]:.6f} '
        f'hard_R2={summary["hard_R2"]:.6f} '
        f'seed_mean={summary["seed_test_soft_R2_mean"]:.6f} '
        f'seed_best={summary["seed_test_soft_R2_best"]:.6f}',
        flush=True,
    )

    pred_df = pd.DataFrame(
        {
            'dataset': name,
            'mode': 'joint_moe_gstar_adapter',
            'image_path': te['image_path'].values,
            'discharge_true': y_te,
            'seg_true': seg_true,
            'seg_pred': seg_pred,
            'p_high': p_high,
            'pred_low': pred_low,
            'pred_high': pred_high,
            'pred_hard': pred_hard,
            'pred_soft': pred_soft,
            'pred_seg_oracle': pred_seg_oracle,
            'pred_expert_oracle': pred_expert_oracle,
        }
    )
    return summary, pred_df, fit_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['RAW'], choices=['RAW', 'CE'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument('--threshold-pct', type=int, default=50)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight-decay', type=float, default=8e-4)
    parser.add_argument('--gate-aux', type=float, default=0.20)
    parser.add_argument('--expert-aux', type=float, default=0.60)
    parser.add_argument('--gstar-aux', type=float, default=0.25)
    parser.add_argument('--high-weight', type=float, default=1.0)
    parser.add_argument('--trunk-dim', type=int, default=256)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.10)

    parser.add_argument('--adapter-type', type=str, default='res_mlp', choices=['none', 'res_mlp', 'gated_res_mlp', 'deeper_res_mlp', 'bottleneck'])
    parser.add_argument('--adapter-hidden', type=int, default=192)
    parser.add_argument('--adapter-bottleneck', type=int, default=64)
    parser.add_argument('--adapter-dropout', type=float, default=0.10)
    parser.add_argument('--adapter-ln-mode', type=str, default='prepost', choices=['prepost', 'pre', 'post'])
    parser.add_argument('--adapter-alpha-init', type=float, default=1.0)
    parser.add_argument('--adapter-alpha-learnable', action='store_true')
    parser.add_argument('--use-metadata', action='store_true')
    parser.add_argument('--metadata-fields', nargs='+', default=['month', 'season'])
    parser.add_argument('--metadata-hidden', type=int, default=16)
    parser.add_argument('--metadata-mode', type=str, default='concat', choices=['concat', 'gate_only'])
    parser.add_argument('--core-type', type=str, default='shared', choices=['shared', 'disentangled'])

    parser.add_argument('--raw-feature-csv', type=str, default='')
    parser.add_argument('--ce-feature-csv', type=str, default='')
    parser.add_argument('--ensemble-mode', type=str, default='weighted', choices=['weighted', 'uniform', 'topk', 'softmax'])
    parser.add_argument('--ensemble-weight-power', type=float, default=1.0)
    parser.add_argument('--ensemble-topk', type=int, default=5)
    parser.add_argument('--tag', type=str, default='adapter_a1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    print(f'tag={args.tag}', flush=True)
    print(f'datasets={args.datasets}', flush=True)
    print(f'seeds={args.seeds}', flush=True)

    ds_map = {}
    raw_path = Path(args.raw_feature_csv) if args.raw_feature_csv else (DATA_DIR / 'features_raw_convnext_tiny_exp2.csv')
    ce_path = Path(args.ce_feature_csv) if args.ce_feature_csv else (DATA_DIR / 'features_ce_cnn_v6.csv')
    if 'RAW' in args.datasets:
        print(f'loading RAW features from: {raw_path}', flush=True)
        ds_map['RAW'] = load_df(raw_path)
    if 'CE' in args.datasets:
        if ce_path.exists():
            print(f'loading CE features from: {ce_path}', flush=True)
            ds_map['CE'] = load_df(ce_path)
        else:
            print(f'skip CE: file not found: {ce_path}', flush=True)

    summaries = []
    preds = []
    logs = []
    for ds_name, ds in ds_map.items():
        s, p, l = evaluate_dataset(ds_name, ds, args=args, device=device)
        summaries.append(s)
        preds.append(p)
        logs.extend(l)

    mdf = pd.DataFrame(summaries)
    pdf = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    ldf = pd.DataFrame(logs)

    m_path = RESULT / f'mlp_v67_adapter_metrics_{args.tag}.csv'
    p_path = RESULT / f'mlp_v67_adapter_predictions_{args.tag}.csv'
    l_path = INTER / f'mlp_v67_adapter_seed_logs_{args.tag}.csv'
    c_path = INTER / f'mlp_v67_adapter_config_{args.tag}.json'

    mdf.to_csv(m_path, index=False)
    pdf.to_csv(p_path, index=False)
    ldf.to_csv(l_path, index=False)
    c_path.write_text(
        json.dumps(
            {
                'datasets': args.datasets,
                'seeds': args.seeds,
                'threshold_pct': args.threshold_pct,
                'val_ratio': args.val_ratio,
                'epochs': args.epochs,
                'patience': args.patience,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'gate_aux': args.gate_aux,
                'expert_aux': args.expert_aux,
                'gstar_aux': args.gstar_aux,
                'high_weight': args.high_weight,
                'trunk_dim': args.trunk_dim,
                'head_dim': args.head_dim,
                'dropout': args.dropout,
                'adapter_type': args.adapter_type,
                'adapter_hidden': args.adapter_hidden,
                'adapter_bottleneck': args.adapter_bottleneck,
                'adapter_dropout': args.adapter_dropout,
                'adapter_ln_mode': args.adapter_ln_mode,
                'adapter_alpha_init': args.adapter_alpha_init,
                'adapter_alpha_learnable': args.adapter_alpha_learnable,
                'use_metadata': args.use_metadata,
                'metadata_fields': args.metadata_fields,
                'metadata_hidden': args.metadata_hidden,
                'metadata_mode': args.metadata_mode,
                'core_type': args.core_type,
                'raw_feature_csv': str(raw_path),
                'ce_feature_csv': str(ce_path),
                'ensemble_mode': args.ensemble_mode,
                'ensemble_weight_power': args.ensemble_weight_power,
                'ensemble_topk': args.ensemble_topk,
                'tag': args.tag,
                'device': str(device),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print('Saved:')
    print(m_path)
    print(p_path)
    print(l_path)
    print(c_path)
    if len(mdf) > 0:
        print('\nMetrics:')
        print(mdf.to_string(index=False))


if __name__ == '__main__':
    main()
