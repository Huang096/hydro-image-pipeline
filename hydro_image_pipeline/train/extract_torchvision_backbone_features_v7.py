from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
)

SEED = 42
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / 'data_refs'
INTER = PACKAGE_ROOT / 'intermediate'
RESULT = PACKAGE_ROOT / 'results'
TORCH_CACHE = INTER / 'torch_cache'
TORCH_CACHE.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_CACHE))

np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class RoiSpec:
    image_width: int
    image_height: int
    polygon: list[tuple[float, float]]


def load_roi(path: Path) -> RoiSpec:
    obj = json.loads(path.read_text(encoding='utf-8'))
    shapes = obj.get('shapes') or []
    shape = next((s for s in shapes if s.get('label') == 'roi'), None)
    if shape is None:
        shape = next((s for s in shapes if s.get('points')), None)
    if shape is None:
        raise ValueError(f"ROI file has no polygon points: {path}")
    pts = [(float(x), float(y)) for x, y in shape['points']]
    return RoiSpec(image_width=int(obj['imageWidth']), image_height=int(obj['imageHeight']), polygon=pts)


def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
    df = df[df['discharge'].notna()].copy()
    df = df[df['discharge'] != -999999].copy()
    df = df[df['year'].isin([2024, 2025])].copy().reset_index(drop=True)
    keep = ['image_path', 'image_time', 'year', 'month', 'hydro_time', 'discharge', 'gauge_height']
    return df[keep].copy()


def make_mask_and_bbox(w: int, h: int, roi: RoiSpec):
    sx = w / float(roi.image_width)
    sy = h / float(roi.image_height)
    pts = [(x * sx, y * sy) for x, y in roi.polygon]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, int(np.floor(min(xs))))
    y1 = max(0, int(np.floor(min(ys))))
    x2 = min(w, int(np.ceil(max(xs))))
    y2 = min(h, int(np.ceil(max(ys))))
    mask_img = Image.new('L', (w, h), 0)
    dr = ImageDraw.Draw(mask_img)
    dr.polygon(pts, fill=255)
    return np.array(mask_img, dtype=np.uint8), (x1, y1, x2, y2)


def refine_roi_water_mask(image_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Remove high-confidence vegetation and bank pixels inside the hand-drawn ROI."""
    roi_u8 = (roi_mask > 0).astype(np.uint8)
    roi = roi_u8 > 0
    roi_area = int(np.count_nonzero(roi))
    if roi_area == 0:
        return roi_mask

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_ch = hsv[..., 0].astype(np.float32)
    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)
    rgb = image_rgb.astype(np.float32)
    r_ch, g_ch, b_ch = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    excess_green = 2.0 * g_ch - r_ch - b_ch
    excess_red = 1.4 * r_ch - g_ch
    dist_to_edge = cv2.distanceTransform(roi_u8, cv2.DIST_L2, 3)
    edge_band_px = max(8.0, min(image_rgb.shape[:2]) * 0.045)
    near_roi_edge = (dist_to_edge > 0) & (dist_to_edge <= edge_band_px)

    vegetation = (
        (h_ch >= 34)
        & (h_ch <= 92)
        & (s_ch >= 62)
        & (v_ch >= 45)
        & (excess_green >= 34)
        & (g_ch >= r_ch * 1.10)
        & (g_ch >= b_ch * 1.00)
    )
    bright_grass = (
        (h_ch >= 24)
        & (h_ch <= 48)
        & (s_ch >= 82)
        & (v_ch >= 85)
        & (excess_green >= 20)
        & (g_ch >= r_ch * 0.95)
        & (g_ch >= b_ch * 1.08)
    )
    warm_bank = (
        near_roi_edge
        & (h_ch <= 30)
        & (s_ch >= 72)
        & (v_ch >= 68)
        & (excess_red >= 30)
        & (r_ch >= g_ch * 1.05)
        & (r_ch >= b_ch * 1.18)
    )
    yellow_bank = (
        near_roi_edge
        & (h_ch >= 20)
        & (h_ch <= 42)
        & (s_ch >= 88)
        & (v_ch >= 88)
        & (r_ch >= b_ch * 1.22)
        & (g_ch >= b_ch * 1.12)
    )

    nonwater = ((vegetation | bright_grass | warm_bank | yellow_bank) & roi).astype(np.uint8) * 255
    kernel_size = 3 if min(nonwater.shape[:2]) < 500 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    nonwater = cv2.morphologyEx(nonwater, cv2.MORPH_OPEN, kernel)
    nonwater = cv2.dilate(nonwater, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((nonwater > 0).astype(np.uint8), 8)
    filtered_nonwater = np.zeros_like(nonwater)
    min_component_area = max(20, int(roi_area * 0.0008))
    max_component_area = int(roi_area * 0.60)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if min_component_area <= area <= max_component_area:
            filtered_nonwater[labels == label_idx] = 255

    refined = roi & (filtered_nonwater == 0)
    keep_ratio = float(np.count_nonzero(refined)) / float(roi_area)
    if keep_ratio < 0.40:
        return roi_mask

    return refined.astype(np.uint8) * 255


def load_roi_pil(path: str, roi: RoiSpec, refine_water_mask: bool = True) -> Image.Image:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    mask, (x1, y1, x2, y2) = make_mask_and_bbox(w, h, roi)
    if refine_water_mask:
        mask = refine_roi_water_mask(arr, mask)
    arr = arr.copy()
    arr[mask == 0] = 0
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = arr
    
    # Apply CLAHE for optical normalization
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(crop)


class RoiDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, roi: RoiSpec, preprocess, refine_water_mask: bool = True):
        self.meta = meta.reset_index(drop=True)
        self.roi = roi
        self.preprocess = preprocess
        self.refine_water_mask = refine_water_mask

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        p = str(self.meta.iloc[idx]['image_path'])
        img = load_roi_pil(p, self.roi, refine_water_mask=self.refine_water_mask)
        return self.preprocess(img)


def build_backbone(name: str):
    if name == 'vit_b16':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        preprocess = weights.transforms()
        emb_dim = 768
        return model, preprocess, emb_dim
    if name == 'vit_l16':
        weights = ViT_L_16_Weights.IMAGENET1K_V1
        model = vit_l_16(weights=weights)
        model.heads = nn.Identity()
        preprocess = weights.transforms()
        emb_dim = 1024
        return model, preprocess, emb_dim
    if name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)
        model.classifier = nn.Identity()
        preprocess = weights.transforms()
        emb_dim = 768
        return model, preprocess, emb_dim
    if name == 'convnext_small':
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = convnext_small(weights=weights)
        model.classifier = nn.Identity()
        preprocess = weights.transforms()
        emb_dim = 768
        return model, preprocess, emb_dim
    if name == 'convnext_base':
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=weights)
        model.classifier = nn.Identity()
        preprocess = weights.transforms()
        emb_dim = 1024
        return model, preprocess, emb_dim
    raise ValueError(f'unsupported backbone: {name}')


def extract(
    meta: pd.DataFrame,
    roi: RoiSpec,
    backbone: str,
    batch_size: int,
    progress_label: str = 'feature extraction',
    refine_water_mask: bool = True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_rows = int(len(meta))
    print(
        f'[{progress_label}] start | device={device} | rows={n_rows} '
        f'| batch_size={batch_size} | refine_water_mask={refine_water_mask}',
        flush=True,
    )
    model, preprocess, emb_dim = build_backbone(backbone)
    model = model.to(device)
    model.eval()

    ds = RoiDataset(meta, roi, preprocess, refine_water_mask=refine_water_mask)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_batches = len(ld)

    feats = []
    with torch.no_grad():
        for i, x in enumerate(ld, start=1):
            x = x.to(device)
            f = model(x)
            f = f.float().view(f.size(0), -1)
            feats.append(f.cpu().numpy())
            if i == 1 or i % 20 == 0 or i == n_batches:
                rows_done = min(i * batch_size, n_rows)
                pct = 100.0 * rows_done / max(1, n_rows)
                print(
                    f'[{progress_label}] batch {i}/{n_batches} ({pct:5.1f}%) | rows {rows_done}/{n_rows}',
                    flush=True,
                )

    arr = np.concatenate(feats, axis=0) if feats else np.zeros((0, emb_dim), dtype=np.float32)
    print(f'[{progress_label}] done | rows={arr.shape[0]} | emb_dim={emb_dim}', flush=True)
    return arr, emb_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='RAW', choices=['RAW', 'CE'])
    ap.add_argument('--backbone', type=str, required=True, choices=['vit_b16', 'vit_l16', 'convnext_tiny', 'convnext_small', 'convnext_base'])
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--disable-water-mask', action='store_true', help='Use the hand-drawn ROI only, without filtering vegetation/bank pixels inside it.')
    args = ap.parse_args()

    csv_name = 'features_raw.csv' if args.dataset == 'RAW' else 'features_ce.csv'
    meta = load_meta(DATA_DIR / csv_name)
    roi = load_roi(DATA_DIR / 'roi.json')

    arr, emb_dim = extract(
        meta,
        roi,
        args.backbone,
        args.batch_size,
        progress_label=f'features {args.dataset} {args.backbone}',
        refine_water_mask=not args.disable_water_mask,
    )

    emb_df = pd.DataFrame(arr, columns=[f'emb_{i:03d}' for i in range(arr.shape[1])])
    out_df = pd.concat([meta.reset_index(drop=True), emb_df], axis=1)

    out_csv = RESULT / f'features_{args.dataset.lower()}_{args.backbone}_{args.tag}.csv'
    out_df.to_csv(out_csv, index=False)

    cfg = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'refine_water_mask': not args.disable_water_mask,
        'tag': args.tag,
        'emb_dim': emb_dim,
        'n_rows': int(len(out_df)),
        'output_csv': str(out_csv),
    }
    cfg_path = INTER / f'torchvision_extract_config_{args.backbone}_{args.tag}.json'
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    print('Saved:')
    print(out_csv)
    print(cfg_path)


if __name__ == '__main__':
    main()
