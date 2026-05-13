from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NIMS_LIST_FILES = "https://api.waterdata.usgs.gov/nims/v0/listFiles"
NIMS_CAMERAS = "https://api.waterdata.usgs.gov/nims/v0/cameras"
NIMS_IMAGES = "https://usgs-nims-images.s3.amazonaws.com"
NWIS_IV = "https://waterservices.usgs.gov/nwis/iv/"
HYDRO_PARAMS = ("00060",)
PARAM_RENAME = {"00060": "discharge"}
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PIPELINE_ROOT / "data"


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_nims_timestamp(filename: str) -> pd.Timestamp | None:
    try:
        stamp = filename.split("___", 1)[1].rsplit(".", 1)[0].replace("Z", "")
        return pd.to_datetime(stamp, format="%Y-%m-%dT%H-%M-%S", utc=True)
    except Exception:
        return None


def local_image_name(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H-%M-%S") + ".jpg"


def request_json(url: str, params: dict, retries: int = 3) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error)


def camera_metadata(cam_id: str) -> dict:
    payload = request_json(NIMS_CAMERAS, {"camId": cam_id})
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("items") or payload.get("files") or payload.get("value") or []
    return items[0] if items and isinstance(items[0], dict) else {}


def parse_nims_items(cam_id: str, payload) -> list[dict]:
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("items") or payload.get("files") or payload.get("value") or []
    rows = []
    for item in items:
        if isinstance(item, str):
            filename = item.split("/")[-1]
        elif isinstance(item, dict):
            filename = (
                item.get("name")
                or item.get("filename")
                or item.get("fileName")
                or item.get("key")
                or item.get("s3Key")
                or ""
            )
            filename = str(filename).split("/")[-1]
        else:
            continue
        if not filename.lower().endswith((".jpg", ".jpeg")):
            continue
        ts = parse_nims_timestamp(filename)
        if ts is None:
            continue
        rows.append(
            {
                "cam_id": cam_id,
                "timestamp_utc": ts,
                "source_filename": filename,
                "url": f"{NIMS_IMAGES}/720/{cam_id}/{filename}",
            }
        )
    return rows


def rows_to_manifest(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["cam_id", "timestamp_utc", "source_filename", "url"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates("source_filename")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def month_floor(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp("2022-01-01T00:00:00Z")
    return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=1, tz="UTC")


def month_window_rows(cam_id: str, limit: int, metadata: dict) -> list[dict]:
    start = month_floor(metadata.get("createdDate"))
    end = pd.to_datetime(metadata.get("newestImageDT"), utc=True, errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp.now(tz="UTC")

    all_rows: list[dict] = []
    cur = start
    while cur <= end + pd.Timedelta(days=31):
        nxt = cur + pd.DateOffset(months=1)
        payload = request_json(
            NIMS_LIST_FILES,
            {
                "camId": cam_id,
                "rawItem": "true",
                "limit": str(limit),
                "recent": "false",
                "after": cur.strftime("%Y-%m-%dT%H-%M-%SZ"),
                "before": nxt.strftime("%Y-%m-%dT%H-%M-%SZ"),
            },
        )
        all_rows.extend(parse_nims_items(cam_id, payload))
        cur = nxt
        time.sleep(0.05)
    return all_rows


def list_nims_files(cam_id: str, limit: int) -> pd.DataFrame:
    metadata = camera_metadata(cam_id)
    rows: list[dict] = []
    bulk_error = None
    try:
        payload = request_json(
            NIMS_LIST_FILES,
            {"camId": cam_id, "rawItem": "true", "limit": str(limit), "recent": "false"},
        )
        rows = parse_nims_items(cam_id, payload)
    except Exception as exc:
        bulk_error = repr(exc)

    if rows and len(rows) < int(limit):
        return rows_to_manifest(rows)

    # Some large cameras either hit the NIMS listFiles limit or return a transient
    # 502 for bulk listing. Windowed monthly listing is slower but avoids truncation.
    try:
        rows = month_window_rows(cam_id=cam_id, limit=limit, metadata=metadata)
    except Exception:
        if bulk_error:
            raise RuntimeError(f"bulk listFiles failed for {cam_id}: {bulk_error}")
        raise
    return rows_to_manifest(rows)


def download_image(url: str, out_path: Path, retries: int = 3) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with out_path.open("wb") as handle:
                    for chunk in response.iter_content(1 << 15):
                        if chunk:
                            handle.write(chunk)
            return True
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return False


def fetch_images(cam_id: str, site_dir: Path, limit: int, skip_download: bool) -> pd.DataFrame:
    listed = list_nims_files(cam_id, limit=limit)
    image_dir = site_dir / "images_all"
    rows = []
    failed_rows = []
    for row in listed.to_dict(orient="records"):
        ts = pd.Timestamp(row["timestamp_utc"])
        filename = local_image_name(ts)
        image_path = image_dir / filename
        downloaded = False
        if not skip_download:
            try:
                downloaded = download_image(str(row["url"]), image_path)
            except Exception as exc:
                failed_rows.append(
                    {
                        "cam_id": cam_id,
                        "source_filename": row["source_filename"],
                        "url": row["url"],
                        "filename": filename,
                        "image_path": str(image_path.resolve()),
                        "image_time": ts.isoformat(),
                        "error": repr(exc),
                    }
                )
                continue
        rows.append(
            {
                "cam_id": cam_id,
                "source_filename": row["source_filename"],
                "url": row["url"],
                "filename": filename,
                "image_path": str(image_path.resolve()),
                "image_time": ts.isoformat(),
                "downloaded_now": downloaded,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(site_dir / "image_manifest.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(site_dir / "failed_downloads.csv", index=False)
    return out


def normalize_qualifiers(raw) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, list):
        raw = [raw]
    values = []
    for item in raw:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            value = item.get("qualifierCode") or item.get("value") or item.get("qualifierDescription")
            if value:
                values.append(str(value))
        elif item is not None:
            values.append(str(item))
    return ",".join(values)


def fetch_hydro(nwis_id: str, start_utc: datetime, end_utc: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = request_json(
        NWIS_IV,
        {
            "sites": nwis_id,
            "parameterCd": ",".join(HYDRO_PARAMS),
            "startDT": iso_z(start_utc),
            "endDT": iso_z(end_utc),
            "format": "json",
            "siteStatus": "all",
        },
    )
    rows = []
    for series in payload.get("value", {}).get("timeSeries", []):
        variable = series.get("variable", {})
        var_code = variable.get("variableCode", [{}])[0].get("value")
        var_name = variable.get("variableName")
        unit = variable.get("unit", {}).get("unitCode")
        source_info = series.get("sourceInfo", {})
        site_no = source_info.get("siteCode", [{}])[0].get("value", nwis_id)
        for block in series.get("values", []):
            qualifiers = normalize_qualifiers(block.get("qualifier", []))
            method = block.get("method", [{}])[0].get("methodDescription")
            for value_item in block.get("value", []):
                rows.append(
                    {
                        "datetime_utc": value_item.get("dateTime"),
                        "site_no": site_no,
                        "variable_cd": var_code,
                        "variable_name": var_name,
                        "unit": unit,
                        "value": value_item.get("value"),
                        "qualifiers": qualifiers,
                        "method": method,
                        "source": "nwis_iv",
                    }
                )
    origin = pd.DataFrame(rows)
    if origin.empty:
        return origin, pd.DataFrame(columns=["time", *PARAM_RENAME.values()])
    origin["datetime_utc"] = pd.to_datetime(origin["datetime_utc"], utc=True, errors="coerce")
    origin["value"] = pd.to_numeric(origin["value"], errors="coerce")
    origin = origin.dropna(subset=["datetime_utc"]).sort_values(["datetime_utc", "variable_cd"]).reset_index(drop=True)
    long_table = (
        origin.pivot_table(index="datetime_utc", columns="variable_cd", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"datetime_utc": "time", **PARAM_RENAME})
        .sort_values("time")
        .reset_index(drop=True)
    )
    for col in PARAM_RENAME.values():
        if col not in long_table.columns:
            long_table[col] = np.nan
    return origin, long_table


def interpolate_at_targets(hydro: pd.DataFrame, image_times: pd.DatetimeIndex, max_gap_minutes: int) -> pd.DataFrame:
    if hydro.empty:
        out = pd.DataFrame(index=image_times)
        for col in PARAM_RENAME.values():
            out[col] = np.nan
        out["nearest_obs_time"] = pd.NaT
        out["gap_prev_min"] = np.inf
        out["gap_next_min"] = np.inf
        out["valid_interpolation"] = False
        return out

    hydro = hydro.copy()
    hydro["time"] = pd.to_datetime(hydro["time"], utc=True, errors="coerce")
    hydro = hydro.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").set_index("time")
    obs = hydro.index
    union = obs.union(image_times).unique().sort_values()
    interp = hydro.reindex(union).interpolate(method="time", limit_direction="both").reindex(image_times)
    outside = (interp.index < obs.min()) | (interp.index > obs.max())
    interp.loc[outside, :] = np.nan

    pos = np.searchsorted(obs.values.astype("datetime64[ns]"), image_times.values.astype("datetime64[ns]"))
    prev_gap = []
    next_gap = []
    nearest = []
    for idx, p in enumerate(pos):
        target = image_times[idx]
        prev_t = obs[p - 1] if p > 0 else None
        next_t = obs[p] if p < len(obs) else None
        pg = (target - prev_t).total_seconds() / 60.0 if prev_t is not None else np.inf
        ng = (next_t - target).total_seconds() / 60.0 if next_t is not None else np.inf
        prev_gap.append(pg)
        next_gap.append(ng)
        nearest.append(prev_t if pg <= ng and prev_t is not None else next_t if next_t is not None else pd.NaT)
    interp["nearest_obs_time"] = nearest
    interp["gap_prev_min"] = prev_gap
    interp["gap_next_min"] = next_gap
    interp["valid_interpolation"] = interp[["gap_prev_min", "gap_next_min"]].min(axis=1) <= max_gap_minutes
    return interp


def align_images_with_hydro(images: pd.DataFrame, hydro: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    if images.empty:
        return pd.DataFrame()
    image_times = pd.DatetimeIndex(pd.to_datetime(images["image_time"], utc=True))
    interp = interpolate_at_targets(hydro, image_times, max_gap_minutes=max_gap_minutes)
    out = images.copy()
    out["hydro_time"] = image_times.map(lambda x: x.isoformat())
    out["nearest_obs_time"] = pd.to_datetime(interp["nearest_obs_time"], utc=True, errors="coerce").map(
        lambda x: x.isoformat() if pd.notna(x) else ""
    )
    for col in PARAM_RENAME.values():
        out[col] = interp[col].values if col in interp.columns else np.nan
    out["gap_prev_min"] = interp["gap_prev_min"].values
    out["gap_next_min"] = interp["gap_next_min"].values
    out["valid_interpolation"] = interp["valid_interpolation"].values
    return out


def write_hydro_outputs(
    hydro_dir: Path,
    nwis_id: str,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hydro_dir.mkdir(parents=True, exist_ok=True)
    origin, hydro = fetch_hydro(str(nwis_id), start_utc=start_utc, end_utc=end_utc)
    origin.to_csv(hydro_dir / "origin.csv", index=False)
    hydro_out = hydro.copy()
    if "time" in hydro_out.columns:
        hydro_out["time"] = pd.to_datetime(hydro_out["time"], utc=True, errors="coerce").map(
            lambda x: x.isoformat() if pd.notna(x) else ""
        )
    hydro_out.to_csv(hydro_dir / "long_table.csv", index=False)
    hydro_counts = origin.groupby("variable_cd").size().astype(int).to_dict() if not origin.empty else {}
    meta = {
        "nwis_id": str(nwis_id),
        "start_utc": iso_z(start_utc),
        "end_utc": iso_z(end_utc),
        "hydro_counts": hydro_counts,
        "has_discharge": int(hydro_counts.get("00060", 0)) > 0,
        "origin_csv": str((hydro_dir / "origin.csv").resolve()),
        "long_table_csv": str((hydro_dir / "long_table.csv").resolve()),
    }
    (hydro_dir / "hydro_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return origin, hydro, meta


def prepare_one(
    row: dict,
    output_root: Path,
    limit: int,
    hydro_buffer_days: int,
    max_gap_minutes: int,
    skip_download: bool,
) -> dict:
    site_dir = output_root / str(row["river_group"]) / str(row["site_folder"])
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "site_meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    images = fetch_images(str(row["cam_id"]), site_dir=site_dir, limit=limit, skip_download=skip_download)
    if images.empty:
        raise ValueError(f"no images listed for {row['cam_id']}")

    image_times = pd.to_datetime(images["image_time"], utc=True)
    hydro_dir = output_root / str(row["river_group"]) / "discharge" / str(row["nwis_id"])
    start_utc = image_times.min().to_pydatetime() - timedelta(days=hydro_buffer_days)
    end_utc = image_times.max().to_pydatetime() + timedelta(days=hydro_buffer_days)
    _, hydro, hydro_meta = write_hydro_outputs(hydro_dir, str(row["nwis_id"]), start_utc, end_utc)

    matched = align_images_with_hydro(images, hydro, max_gap_minutes=max_gap_minutes)
    matched["river_group"] = row["river_group"]
    matched["site_folder"] = row["site_folder"]
    matched["nwis_id"] = str(row["nwis_id"])
    matched.to_csv(site_dir / "matched_all.csv", index=False)
    matched_valid = matched[matched["valid_interpolation"].fillna(False)].copy()
    matched_valid.to_csv(site_dir / "matched.csv", index=False)

    summary = {
        "river_group": row["river_group"],
        "site_folder": row["site_folder"],
        "cam_id": row["cam_id"],
        "nwis_id": str(row["nwis_id"]),
        "site_dir": str(site_dir.resolve()),
        "n_images": int(len(images)),
        "n_matched_valid": int(len(matched_valid)),
        "hydro_counts": hydro_meta["hydro_counts"],
        "has_discharge": bool(hydro_meta["has_discharge"]),
        "shared_hydro": hydro_meta,
    }
    (site_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_river_group(
    group_df: pd.DataFrame,
    output_root: Path,
    limit: int,
    hydro_buffer_days: int,
    max_gap_minutes: int,
    skip_download: bool,
) -> list[dict]:
    if group_df.empty:
        return []
    river_group = str(group_df.iloc[0]["river_group"])
    nwis_ids = sorted(group_df["nwis_id"].astype(str).unique().tolist())
    if len(nwis_ids) != 1:
        raise ValueError(f"{river_group}: expected one nwis_id per river group, got {nwis_ids}")
    nwis_id = nwis_ids[0]
    river_dir = output_root / river_group
    river_dir.mkdir(parents=True, exist_ok=True)

    image_frames: list[tuple[dict, Path, pd.DataFrame]] = []
    all_times = []
    for row in group_df.to_dict(orient="records"):
        site_dir = river_dir / str(row["site_folder"])
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "site_meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        images = fetch_images(str(row["cam_id"]), site_dir=site_dir, limit=limit, skip_download=skip_download)
        if images.empty:
            raise ValueError(f"no images listed for {row['cam_id']}")
        image_times = pd.to_datetime(images["image_time"], utc=True)
        all_times.append(image_times)
        image_frames.append((row, site_dir, images))

    combined_times = pd.concat([pd.Series(t) for t in all_times], ignore_index=True)
    start_utc = combined_times.min().to_pydatetime() - timedelta(days=hydro_buffer_days)
    end_utc = combined_times.max().to_pydatetime() + timedelta(days=hydro_buffer_days)
    hydro_dir = river_dir / "discharge" / str(nwis_id)
    _, hydro, hydro_meta = write_hydro_outputs(hydro_dir, nwis_id, start_utc, end_utc)

    summaries = []
    for row, site_dir, images in image_frames:
        matched = align_images_with_hydro(images, hydro, max_gap_minutes=max_gap_minutes)
        matched["river_group"] = row["river_group"]
        matched["site_folder"] = row["site_folder"]
        matched["nwis_id"] = str(row["nwis_id"])
        matched.to_csv(site_dir / "matched_all.csv", index=False)
        matched_valid = matched[matched["valid_interpolation"].fillna(False)].copy()
        matched_valid.to_csv(site_dir / "matched.csv", index=False)
        summary = {
            "river_group": row["river_group"],
            "site_folder": row["site_folder"],
            "cam_id": row["cam_id"],
            "nwis_id": str(row["nwis_id"]),
            "site_dir": str(site_dir.resolve()),
            "n_images": int(len(images)),
            "n_matched_valid": int(len(matched_valid)),
            "hydro_counts": hydro_meta["hydro_counts"],
            "has_discharge": bool(hydro_meta["has_discharge"]),
            "shared_hydro": hydro_meta,
        }
        (site_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(river_dir / "river_prepare_summary.csv", index=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Download camera images and NWIS discharge data.")
    parser.add_argument("--site-csv", default=str(Path(__file__).resolve().parents[1] / "configs" / "site_list.csv"))
    parser.add_argument("--output-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--hydro-buffer-days", type=int, default=1)
    parser.add_argument("--max-gap-minutes", type=int, default=90)
    parser.add_argument("--only-cam-id", default="")
    parser.add_argument("--only-river-group", default="")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    sites = pd.read_csv(args.site_csv, dtype=str).fillna("")
    if args.only_cam_id:
        sites = sites[sites["cam_id"] == args.only_cam_id].copy()
    if args.only_river_group:
        sites = sites[sites["river_group"] == args.only_river_group].copy()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    if args.only_river_group:
        print(f"[group] {args.only_river_group} | cameras={len(sites)}", flush=True)
        summaries.extend(
            prepare_river_group(
                group_df=sites,
                output_root=output_root,
                limit=args.limit,
                hydro_buffer_days=args.hydro_buffer_days,
                max_gap_minutes=args.max_gap_minutes,
                skip_download=args.skip_download,
            )
        )
    else:
        for idx, row in enumerate(sites.to_dict(orient="records"), start=1):
            print(f"[{idx}/{len(sites)}] {row['river_group']} / {row['site_folder']}", flush=True)
            summaries.append(
                prepare_one(
                    row=row,
                    output_root=output_root,
                    limit=args.limit,
                    hydro_buffer_days=args.hydro_buffer_days,
                    max_gap_minutes=args.max_gap_minutes,
                    skip_download=args.skip_download,
                )
            )
    summary_name = "data_prepare_summary.csv"
    if args.only_cam_id:
        safe_cam_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.only_cam_id)
        summary_name = f"data_prepare_summary__{safe_cam_id}.csv"
    if args.only_river_group:
        safe_group = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.only_river_group)
        summary_name = f"data_prepare_summary__{safe_group}.csv"
    pd.DataFrame(summaries).to_csv(output_root / summary_name, index=False)
    print(json.dumps({"output_root": str(output_root), "n_sites": int(len(summaries))}, indent=2))


if __name__ == "__main__":
    main()
