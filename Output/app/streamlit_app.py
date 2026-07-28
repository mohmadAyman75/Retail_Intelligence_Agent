from __future__ import annotations

import html
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from llm_query import OLLAMA_MODEL, run_llm_query

# ──────────────────────────────────── PATHS ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "Output" / "tables"
VIDEOS_DIR = PROJECT_ROOT / "Output" / "videos"
RAW_DIR = PROJECT_ROOT / "Data" / "raw"
DB_PATH = PROJECT_ROOT / "Output" / "database" / "retail_intelligence.duckdb"
ZONES_PATH = PROJECT_ROOT / "Data" / "config" / "store_zones.json"
LOCAL_TRACKING_RUN_PATH = TABLES_DIR / "local_tracking_run.json"
LOCAL_TRACKS_PATH = TABLES_DIR / "local_tracks.csv"
FRESHNESS_EXEMPT_CSVS = frozenset({"local_tracks.csv", "video_metadata.csv"})

# ──────────────────────────────────── CONFIG ───────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────── CUSTOM CSS ────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
.stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #1a1f36 0%, #252b48 50%, #1e2740 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4f6ef7, #6c5ce7, #a855f7);
}
.main-header h1 {
    color: #e8eaed;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: rgba(255,255,255,0.5);
    font-size: 0.85rem;
    margin: 0.3rem 0 0 0;
    font-weight: 400;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(25, 30, 52, 0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 0.8rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 0.5rem;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.12);
}
.metric-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8eaed;
    line-height: 1.2;
}
.metric-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.75rem;
    margin-top: 0.25rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-card.blue   { border-top: 2px solid #4f6ef7; }
.metric-card.blue .metric-value { color: #7b93fa; }
.metric-card.purple { border-top: 2px solid #8b5cf6; }
.metric-card.purple .metric-value { color: #a78bfa; }
.metric-card.green  { border-top: 2px solid #10b981; }
.metric-card.green .metric-value { color: #34d399; }
.metric-card.orange { border-top: 2px solid #f59e0b; }
.metric-card.orange .metric-value { color: #fbbf24; }
.metric-card.red    { border-top: 2px solid #ef4444; }
.metric-card.red .metric-value { color: #f87171; }
.metric-card.pink   { border-top: 2px solid #ec4899; }
.metric-card.pink .metric-value { color: #f472b6; }

/* ── Insight Cards ── */
.insight-card {
    background: rgba(25, 30, 52, 0.7);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid #4f6ef7;
    transition: border-color 0.2s ease;
}
.insight-card:hover { border-left-color: #6c5ce7; }
.insight-card .insight-title {
    color: #a78bfa;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}
.insight-card .insight-text {
    color: rgba(255,255,255,0.65);
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.section-header h3 {
    color: #e8eaed;
    font-weight: 600;
    margin: 0;
    font-size: 1rem;
}

/* ── Video Info Bar ── */
.video-info {
    background: rgba(25, 30, 52, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
}
.video-info-item { text-align: center; }
.video-info-item .label { color: rgba(255,255,255,0.4); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
.video-info-item .value { color: #7b93fa; font-weight: 600; font-size: 0.95rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 18, 35, 0.98), rgba(10, 12, 25, 0.99));
    border-right: 1px solid rgba(255,255,255,0.04);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(20, 25, 45, 0.4);
    border-radius: 8px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* ── Heatmap Legend ── */
.heatmap-legend {
    background: rgba(25, 30, 52, 0.6);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.6rem;
    text-align: center;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.55);
}

/* ── Chat examples ── */
.chat-examples {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), rgba(16, 185, 129, 0.02));
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.chat-examples .ex-title {
    color: #10b981;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
}
.chat-examples .ex-list {
    color: rgba(255,255,255,0.55);
    font-size: 0.8rem;
    line-height: 1.8;
}

/* ── View mode controls ── */
.view-mode-bar {
    background: rgba(25, 30, 52, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* ── Speed indicator ── */
.speed-badge {
    display: inline-block;
    background: rgba(79, 110, 247, 0.15);
    border: 1px solid rgba(79, 110, 247, 0.3);
    color: #7b93fa;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* ── Cam label overlay ── */
.cam-label {
    background: rgba(0,0,0,0.65);
    color: #e8eaed;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    position: absolute;
    top: 6px;
    left: 6px;
}

/* ── ID note ── */
.id-note {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.5rem;
}
.id-note strong { color: #fbbf24; }

/* ── Plotly chart container ── */
.stPlotlyChart { border-radius: 8px; overflow: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────── DATA LOADING ──────────────────────────────────


def tracking_reference_mtime_ns() -> int | None:
    """Return the latest local-tracking marker, falling back to local tracks."""
    timestamps: list[int] = []
    for path in (LOCAL_TRACKING_RUN_PATH, LOCAL_TRACKS_PATH):
        try:
            if path.exists() and path.stat().st_size > 0:
                timestamps.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return max(timestamps) if timestamps else None


def artifact_is_current(
    path: Path,
    reference_mtime_ns: int | None = None,
) -> bool:
    """Reject derived artifacts created before the latest local-tracking run."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if (
            path.suffix.casefold() == ".csv"
            and path.name.casefold() in FRESHNESS_EXEMPT_CSVS
        ):
            return True
        reference = (
            tracking_reference_mtime_ns()
            if reference_mtime_ns is None
            else reference_mtime_ns
        )
        return reference is None or path.stat().st_mtime_ns >= reference
    except OSError:
        return False


def read_csv(
    name: str,
    columns: list[str] | None = None,
    reference_mtime_ns: int | None = None,
) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not artifact_is_current(path, reference_mtime_ns):
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ):
        return pd.DataFrame(columns=columns or [])


@st.cache_data(ttl=20)
def load_data(reference_mtime_ns: int | None) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    return (
        read_csv("zone_traffic.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("queue_history.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("agent_actions.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("zone_events.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("video_quarter_counts.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("movement_metrics.csv", reference_mtime_ns=reference_mtime_ns),
        read_csv("zone_transitions.csv", reference_mtime_ns=reference_mtime_ns),
    )


@st.cache_data(ttl=20)
def csv_has_rows(name: str, reference_mtime_ns: int | None = None) -> bool:
    path = TABLES_DIR / name
    if not artifact_is_current(path, reference_mtime_ns):
        return False
    try:
        return not pd.read_csv(path, nrows=1).empty
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ):
        return False


# ─────────────────────────────── HELPER FUNCTIONS ──────────────────────────────


def camera_id_from_video(video_path: Path) -> str:
    stem = video_path.stem
    for prefix in ("annotated_",):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return stem


def infer_store_id(camera_id: str) -> str:
    match = re.search(r"(place_\d+)", camera_id, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "default_store"


def list_annotated_videos(selected_store: str | None = None) -> list[Path]:
    if not VIDEOS_DIR.exists():
        return []

    videos_by_camera = {
        camera_id_from_video(video): video
        for video in sorted(VIDEOS_DIR.glob("annotated_*.mp4"))
    }

    videos = [
        videos_by_camera[camera_id]
        for camera_id in sorted(videos_by_camera, key=str.casefold)
    ]
    if selected_store:
        videos = [
            v for v in videos
            if infer_store_id(camera_id_from_video(v)) == selected_store.lower()
        ]
    return videos


def extract_frame(video_path: Path, frame_index: int = 0) -> np.ndarray | None:
    """Extract a single frame from a video file and return as RGB."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


def get_video_info(video_path: Path) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_synced_frames(
    video_paths: list[Path], frame_index: int
) -> list[np.ndarray | None]:
    """Extract the same frame index from multiple videos."""
    frames = []
    for vp in video_paths:
        frames.append(extract_frame(vp, frame_index))
    return frames


def add_camera_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Overlay a small camera label on the top-left of a frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 800)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (4, 4), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.rectangle(frame, (4, 4), (tw + 14, th + 14), (79, 110, 247), 1)
    cv2.putText(frame, label, (9, th + 9), font, scale, (232, 234, 237), thickness, cv2.LINE_AA)
    return frame


def stitch_grid(frames: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Combine frames into a grid with `cols` columns."""
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Determine uniform size from first frame
    target_h, target_w = frames[0].shape[:2]
    resized = []
    for f in frames:
        if f is not None:
            resized.append(cv2.resize(f, (target_w, target_h)))
        else:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Pad to fill the grid
    rows_needed = (len(resized) + cols - 1) // cols
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    row_images = []
    for r in range(rows_needed):
        row_frames = resized[r * cols: (r + 1) * cols]
        row_images.append(np.hstack(row_frames))
    return np.vstack(row_images)


def camera_short_name(camera_id: str) -> str:
    """Extract a short camera name like 'Cam 17' from a full camera ID."""
    match = re.search(r"camera_(\d+)", camera_id)
    if match:
        return f"Cam {int(match.group(1))}"
    return camera_id


# ─── Heatmap generation ───


@st.cache_data(ttl=300, show_spinner=False)
def generate_camera_heatmap(
    camera_id: str,
    _zone_events_bytes: bytes,
    _raw_dir: str,
    _videos_dir: str,
) -> np.ndarray | None:
    """Build a heatmap overlay on a camera's reference frame.

    Parameters use underscore-prefix or bytes so Streamlit skips expensive
    hashing of large DataFrames.
    """
    zone_events = pd.read_csv(pd.io.common.BytesIO(_zone_events_bytes))

    # Resolve video path (prefer raw over annotated)
    raw_path = Path(_raw_dir) / f"{camera_id}.mp4"
    ann_path = Path(_videos_dir) / f"annotated_{camera_id}.mp4"
    video_path = raw_path if raw_path.exists() else (ann_path if ann_path.exists() else None)
    if video_path is None:
        return None

    # Grab a mid-video frame as background
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]

    # Collect foot-point coordinates for this camera
    cam = zone_events[zone_events["camera_id"] == camera_id]
    if cam.empty or "foot_x" not in cam.columns or "foot_y" not in cam.columns:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fx = pd.to_numeric(cam["foot_x"], errors="coerce").dropna().values
    fy = pd.to_numeric(cam["foot_y"], errors="coerce").dropna().values
    if len(fx) == 0:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Accumulate a density surface
    heatmap = np.zeros((h, w), dtype=np.float32)
    for x, y in zip(fx, fy):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            heatmap[iy, ix] += 1.0

    # Smooth with a large Gaussian
    ksize = max(w, h) // 8
    if ksize % 2 == 0:
        ksize += 1
    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)

    # Normalise to [0, 1]
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx

    # Colour-map (JET: blue → green → yellow → red)
    heatmap_coloured = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Alpha-blend (stronger where density is higher, max 70 %)
    alpha = np.clip(heatmap * 0.7, 0, 0.7)
    alpha3 = np.stack([alpha] * 3, axis=-1)
    blended = frame.astype(np.float32) * (1 - alpha3) + heatmap_coloured.astype(np.float32) * alpha3
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


@st.cache_data(ttl=300, show_spinner=False)
def generate_camera_trajectory_overlay(
    camera_id: str,
    _zone_events_bytes: bytes,
    _raw_dir: str,
    _videos_dir: str,
) -> np.ndarray | None:
    """Draw camera-local customer trajectories from floor points on a reference frame.

    With the current identity calibration, floor_x/floor_y are camera pixels.
    The calculation is intentionally scoped to one camera and never joins paths
    across cameras.
    """
    zone_events = pd.read_csv(pd.io.common.BytesIO(_zone_events_bytes))
    required_columns = {
        "camera_id",
        "camera_track_uid",
        "floor_x",
        "floor_y",
        "timestamp_sec",
        "frame_index",
    }
    if not required_columns.issubset(zone_events.columns):
        return None

    raw_path = Path(_raw_dir) / f"{camera_id}.mp4"
    ann_path = Path(_videos_dir) / f"annotated_{camera_id}.mp4"
    video_path = raw_path if raw_path.exists() else (ann_path if ann_path.exists() else None)
    if video_path is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    camera_events = zone_events[zone_events["camera_id"].astype(str).eq(camera_id)].copy()
    if "is_customer" in camera_events.columns:
        camera_events = camera_events[
            camera_events["is_customer"].astype(str).str.lower().isin({"true", "1"})
        ]
    for column in ("floor_x", "floor_y", "timestamp_sec", "frame_index"):
        camera_events[column] = pd.to_numeric(camera_events[column], errors="coerce")
    camera_events = camera_events.dropna(
        subset=["floor_x", "floor_y", "timestamp_sec", "frame_index"]
    ).sort_values(["camera_track_uid", "timestamp_sec", "frame_index"])
    if camera_events.empty:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width = frame.shape[:2]
    for color_index, (track_id, track) in enumerate(
        camera_events.groupby("camera_track_uid", sort=True)
    ):
        points = track[["floor_x", "floor_y"]].to_numpy(dtype=np.float32)
        in_frame = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        points = points[in_frame]
        if len(points) == 0:
            continue

        polyline = np.round(points).astype(np.int32).reshape((-1, 1, 2))
        color = (
            int(80 + (color_index * 73) % 160),
            int(80 + (color_index * 137) % 160),
            int(80 + (color_index * 41) % 160),
        )
        if len(polyline) > 1:
            cv2.polylines(frame, [polyline], False, color, 2, cv2.LINE_AA)
        start = tuple(polyline[0, 0])
        end = tuple(polyline[-1, 0])
        cv2.circle(frame, start, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, end, 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (min(start[0] + 8, width - 70), max(start[1] - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_transition_sankey(transitions: pd.DataFrame) -> go.Figure | None:
    """Create a Sankey chart from the saved zone-transition table."""
    required_columns = {"from_zone_id", "to_zone_id", "transition_count"}
    if transitions.empty or not required_columns.issubset(transitions.columns):
        return None

    flow = transitions.copy()
    flow["transition_count"] = pd.to_numeric(flow["transition_count"], errors="coerce")
    flow = flow.dropna(subset=["from_zone_id", "to_zone_id", "transition_count"])
    flow = flow[flow["transition_count"] > 0]
    if flow.empty:
        return None

    node_labels = sorted(
        set(flow["from_zone_id"].astype(str)) | set(flow["to_zone_id"].astype(str))
    )
    node_index = {label: index for index, label in enumerate(node_labels)}
    figure = go.Figure(
        go.Sankey(
            node={"label": node_labels, "pad": 18, "thickness": 22},
            link={
                "source": [node_index[value] for value in flow["from_zone_id"].astype(str)],
                "target": [node_index[value] for value in flow["to_zone_id"].astype(str)],
                "value": flow["transition_count"].astype(int).tolist(),
            },
        )
    )
    apply_dark_layout(figure, "Customer Flow Between Zones")
    figure.update_layout(height=430)
    return figure


# ─── UI component helpers ───


def metric_card_html(icon: str, value: str, label: str, color: str = "blue") -> str:
    return (
        f'<div class="metric-card {color}">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f"</div>"
    )


def insight_card_html(title: str, text: str, border_color: str = "#4f6ef7") -> str:
    safe_title = html.escape(str(title))
    safe_text = html.escape(str(text)).replace("\n", "<br>")
    return (
        f'<div class="insight-card" style="border-left-color:{border_color};">'
        f'<div class="insight-title">{safe_title}</div>'
        f'<div class="insight-text">{safe_text}</div>'
        f"</div>"
    )


def generate_insights(
    traffic: pd.DataFrame,
    queues: pd.DataFrame,
    alerts: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Auto-generate smart textual insights from the data."""
    insights: list[tuple[str, str]] = []

    if not traffic.empty and "customer_count" in traffic.columns:
        # Peak
        peak = traffic.sort_values("customer_count", ascending=False).iloc[0]
        zone_label = peak.get("zone_label_ar", peak.get("zone_id", "Unknown"))
        insights.append((
            "Peak Congestion",
            f"Highest traffic recorded in '{zone_label}' at "
            f"{peak['window_start']} with {int(peak['customer_count'])} customers.",
        ))
        # Average per zone
        label_col = "zone_label_ar" if "zone_label_ar" in traffic.columns else "zone_id"
        avg = traffic.groupby(label_col)["customer_count"].mean()
        top_zone = avg.idxmax()
        insights.append((
            "Busiest Zone",
            f"'{top_zone}' is the busiest zone with an average of "
            f"{avg[top_zone]:.1f} customers per time window.",
        ))
        # Trend
        if "window_start_sec" in traffic.columns:
            by_time = traffic.groupby("window_start_sec")["customer_count"].sum().sort_index()
            if len(by_time) >= 4:
                first = by_time.iloc[: len(by_time) // 2].mean()
                second = by_time.iloc[len(by_time) // 2 :].mean()
                if second > first * 1.15:
                    pct = (second - first) / first * 100
                    insights.append((
                        "Traffic Trend — Increasing",
                        f"Traffic is rising. Increased by {pct:.0f}% in the second half of the recording.",
                    ))
                elif first > second * 1.15:
                    pct = (first - second) / first * 100
                    insights.append((
                        "Traffic Trend — Decreasing",
                        f"Traffic is declining. Dropped by {pct:.0f}% in the second half.",
                    ))
                else:
                    insights.append((
                        "Traffic Trend — Stable",
                        "Traffic remains relatively stable throughout the recording period.",
                    ))

    if not queues.empty and "queue_length" in queues.columns:
        mx = int(queues["queue_length"].max())
        av = round(queues["queue_length"].mean(), 1)
        tip = "Consider opening an additional checkout during peak." if mx > 5 else "Queue is under control."
        insights.append((
            "Queue Analysis",
            f"Max length: {mx} people — Average: {av}. {tip}",
        ))

    if not alerts.empty:
        n_high = (
            int(alerts["severity"].eq("high").sum())
            if "severity" in alerts.columns
            else len(alerts)
        )
        if n_high > 0:
            insights.append((
                "High-Severity Alerts",
                f"{n_high} high-severity alerts detected. Check the Alerts tab.",
            ))

    return insights


def apply_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent premium dark theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 18, 35, 0.8)",
        title=dict(text=title, font=dict(size=15, color="#e8eaed", family="Inter")),
        font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    return fig


def build_chart_from_decision(
    query_result: pd.DataFrame,
    decision: dict,
) -> go.Figure | None:
    """Build Plotly code locally from the model's validated JSON decision."""
    if query_result.empty:
        return None
    chart_type = decision.get("chart_type", "none")
    x_column = decision.get("x")
    y_column = decision.get("y")
    if (
        chart_type == "none"
        or x_column not in query_result.columns
        or y_column not in query_result.columns
    ):
        return None

    if chart_type == "bar":
        figure = px.bar(query_result, x=x_column, y=y_column)
    elif chart_type == "line":
        figure = px.line(query_result, x=x_column, y=y_column, markers=True)
    elif chart_type == "scatter":
        figure = px.scatter(query_result, x=x_column, y=y_column)
    else:
        return None

    return apply_dark_layout(figure, "Query Result")


def rule_based_query(
    question: str,
    traffic: pd.DataFrame,
    queues: pd.DataFrame,
) -> tuple[str, pd.DataFrame | None] | None:
    """Fast path for common keyword questions (Arabic + English)."""
    q = question.strip().lower()
    if not q:
        return "Type your question about traffic, queues, or peak times.", None
    if traffic.empty:
        return (
            "No data yet. Run the processing notebooks and place camera videos in Data/raw.",
            None,
        )
    # Arabic keywords (kept for backwards compatibility)
    if any(w in q for w in ["ذروة", "زحمة", "ازدحام", "أكثر", "peak", "busiest", "crowded"]):
        top = traffic.sort_values("customer_count", ascending=False).head(10)
        row = top.iloc[0]
        zone_label = row.get("zone_label_ar", row.get("zone_id", "Unknown"))
        return (
            f"Peak traffic was at '{zone_label}' at {row['window_start']} "
            f"with {int(row['customer_count'])} customers.",
            top,
        )
    if any(w in q for w in ["طابور", "كاشير", "صف", "queue", "checkout", "line"]):
        if queues.empty:
            return "No queue zone configured in the store data.", None
        last = queues.sort_values("window_start_sec").tail(1).iloc[0]
        return (
            f"Latest queue reading: {int(last['queue_length'])} customers. "
            f"Predicted next window: {int(last['predicted_queue_next_window'])}.",
            queues.tail(12),
        )
    if any(w in q for w in ["مدخل", "دخل", "زوار", "entrance", "visitors", "entered"]):
        entrance = traffic[traffic["zone_kind"].eq("entrance")]
        total = int(entrance["customer_count"].sum()) if not entrance.empty else 0
        return f"{total} entrance readings recorded in the available data.", entrance
    return None


@st.cache_data(ttl=300, show_spinner=False)
def cached_llm_query(question: str, database_mtime_ns: int) -> dict:
    """Cache a query only while the DuckDB file has not changed."""
    return run_llm_query(question)


# ──────────────────────────────── LOAD DATA ────────────────────────────────────
tracking_reference_ns = tracking_reference_mtime_ns()
(
    traffic,
    queues,
    alerts,
    zone_events,
    video_quarter_counts,
    movement_metrics,
    zone_transitions,
) = load_data(tracking_reference_ns)
local_tracks_ready = csv_has_rows("local_tracks.csv", tracking_reference_ns)
database_ready = artifact_is_current(DB_PATH, tracking_reference_ns)

# ──────────────────────────────── HEADER ───────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>Retail Intelligence Agent</h1>"
    "<p>Customer traffic analysis, queue monitoring, and store analytics</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────── SIDEBAR ──────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")

    store_options = sorted(
        {
            sid
            for df in (traffic, zone_events)
            if not df.empty and "store_id" in df.columns
            for sid in df["store_id"].dropna().astype(str).unique()
        }
        | {
            infer_store_id(camera_id_from_video(video))
            for video in list_annotated_videos()
        }
    )
    selected_store = (
        st.selectbox("Store / Location", store_options, help="Select a store to view its data")
        if store_options
        else None
    )

    camera_options = sorted(
        {
            camera_id
            for df in (traffic, zone_events)
            if not df.empty and "camera_id" in df.columns
            for camera_id in df["camera_id"].dropna().astype(str).unique()
        }
        | {
            camera_id_from_video(video)
            for video in list_annotated_videos(selected_store)
        }
    )
    selected_camera = (
        st.selectbox(
            "Camera",
            camera_options,
            help="All metrics are scoped to one camera because identities are not merged across cameras.",
        )
        if camera_options
        else None
    )

    st.markdown("---")

    n_videos = len(list_annotated_videos(selected_store))

    st.markdown(
        f"**Database:** {'Ready' if database_ready else 'Not found or stale'}"
    )
    st.markdown(f"**Tracking data:** {'Available' if local_tracks_ready else 'Not available'}")
    st.markdown(f"**Annotated videos:** {n_videos}")
    st.markdown(f"**Local model:** `{OLLAMA_MODEL}`")

    st.markdown("**Identity scope:** Local IDs inside each camera only")

    st.markdown("---")

    if st.button("Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────── STORE FILTER ─────────────────────────────────
if selected_store:
    def filter_store(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "store_id" not in frame.columns:
            return frame
        return frame[frame["store_id"].astype(str).eq(selected_store)].copy()

    traffic = filter_store(traffic)
    queues = filter_store(queues)
    alerts = filter_store(alerts)
    zone_events = filter_store(zone_events)
    video_quarter_counts = filter_store(video_quarter_counts)
    movement_metrics = filter_store(movement_metrics)
    zone_transitions = filter_store(zone_transitions)

if selected_camera:
    def filter_camera(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "camera_id" not in frame.columns:
            return frame
        return frame[frame["camera_id"].astype(str).eq(selected_camera)].copy()

    traffic = filter_camera(traffic)
    queues = filter_camera(queues)
    alerts = filter_camera(alerts)
    zone_events = filter_camera(zone_events)
    video_quarter_counts = filter_camera(video_quarter_counts)
    movement_metrics = filter_camera(movement_metrics)
    zone_transitions = filter_camera(zone_transitions)

# ──────────────────────────────── TABS ─────────────────────────────────────────
(
    tab_live,
    tab_camera,
    tab_heatmap,
    tab_trajectories,
    tab_queue,
    tab_analytics,
    tab_alerts,
    tab_chat,
) = st.tabs(
    [
        "Dashboard",
        "Camera Review",
        "Density Heatmap",
        "Trajectories",
        "Queue Analysis",
        "Analytics",
        "Alerts",
        "Query Data",
    ]
)

# ══════════════════════════════ TAB 1 — DASHBOARD ══════════════════════════════
with tab_live:
    latest = (
        traffic.sort_values("window_start_sec").groupby("zone_id", as_index=False).tail(1)
        if not traffic.empty
        else traffic
    )

    customers = int(latest["customer_count"].sum()) if not latest.empty else 0
    peak_zone = (
        latest.sort_values("customer_count", ascending=False).iloc[0].get(
            "zone_label_ar", latest.sort_values("customer_count", ascending=False).iloc[0].get("zone_id", "—")
        )
        if not latest.empty
        else "—"
    )
    n_alerts = len(alerts)
    avg_cust = round(traffic["customer_count"].mean(), 1) if not traffic.empty else 0
    n_zones = traffic["zone_id"].nunique() if not traffic.empty else 0
    anomaly_n = (
        int(traffic["is_anomaly"].sum())
        if not traffic.empty and "is_anomaly" in traffic.columns
        else 0
    )

    # ── KPI row ──
    cols = st.columns(6)
    kpis = [
        ("👥", str(customers), "Current Customers", "blue"),
        ("📍", str(peak_zone), "Busiest Zone", "purple"),
        ("🔔", str(n_alerts), "Alerts", "orange"),
        ("📊", str(avg_cust), "Avg. Customers", "green"),
        ("🗂️", str(n_zones), "Active Zones", "pink"),
        ("⚠️", str(anomaly_n), "Anomalies", "red"),
    ]
    for col, (ic, val, lbl, clr) in zip(cols, kpis):
        col.markdown(metric_card_html(ic, val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if latest.empty:
        st.info("Run notebooks 01 → 02 → 03 after adding camera videos to Data/raw.")
    else:
        col_chart, col_insights = st.columns([3, 2])

        with col_chart:
            st.markdown(
                '<div class="section-header"><h3>Customer Distribution by Zone</h3></div>',
                unsafe_allow_html=True,
            )
            zone_colors = {
                "queue": "#ef4444",
                "sales": "#3b82f6",
                "entrance": "#10b981",
                "outside": "#6b7280",
            }
            label_col = "zone_label_ar" if "zone_label_ar" in latest.columns else "zone_id"
            fig_bar = px.bar(
                latest,
                x=label_col,
                y="customer_count",
                color="zone_kind",
                color_discrete_map=zone_colors,
                labels={
                    label_col: "Zone",
                    "customer_count": "Camera-local Track Count",
                    "zone_kind": "Zone Type",
                },
            )
            apply_dark_layout(fig_bar, "Traffic by Zone")
            fig_bar.update_traces(marker_line_width=0, opacity=0.9)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_insights:
            st.markdown(
                '<div class="section-header"><h3>Key Insights</h3></div>',
                unsafe_allow_html=True,
            )
            insights = generate_insights(traffic, queues, alerts)
            if insights:
                for title, text in insights:
                    st.markdown(insight_card_html(title, text), unsafe_allow_html=True)
            else:
                st.info("Not enough data for insights yet.")

# ══════════════════════════════ TAB 2 — CAMERA ═════════════════════════════════
with tab_camera:
    st.markdown(
        '<div class="section-header"><h3>Detection & Tracking Review</h3></div>',
        unsafe_allow_html=True,
    )

    if not local_tracks_ready:
        st.info("local_tracks.csv not found or empty. Run notebook 01 first.")
    else:
        annotated_videos = list_annotated_videos(selected_store)
        if not annotated_videos:
            st.info("No annotated videos for the selected store. Re-run Notebook 01.")
        else:
            # ── View mode selector ──
            view_mode = st.radio(
                "View Mode",
                ["Single", "Dual", "Quad (2×2)"],
                horizontal=True,
                key="view_mode",
                help="Single: one camera. Dual: two side-by-side. Quad: all cameras in a 2×2 grid.",
            )

            # ── Camera selection based on view mode ──
            if view_mode == "Single":
                sel_vid = st.selectbox(
                    "Camera",
                    annotated_videos,
                    format_func=lambda p: camera_id_from_video(p),
                    key="cam_select_single",
                )
                active_videos = [sel_vid]
            elif view_mode == "Dual":
                col_a, col_b = st.columns(2)
                with col_a:
                    vid_a = st.selectbox(
                        "Camera A",
                        annotated_videos,
                        index=0,
                        format_func=lambda p: camera_id_from_video(p),
                        key="cam_select_dual_a",
                    )
                with col_b:
                    default_b = min(1, len(annotated_videos) - 1)
                    vid_b = st.selectbox(
                        "Camera B",
                        annotated_videos,
                        index=default_b,
                        format_func=lambda p: camera_id_from_video(p),
                        key="cam_select_dual_b",
                    )
                active_videos = [vid_a, vid_b]
            else:  # Quad
                active_videos = annotated_videos[:4]
                cam_names = [camera_short_name(camera_id_from_video(v)) for v in active_videos]
                st.caption(f"Showing {len(active_videos)} cameras: {', '.join(cam_names)}")

            id_label_mode = "Local per-camera IDs"

            # ── Video info from first active video ──
            vinfo = get_video_info(active_videos[0])
            total_frames = vinfo["total_frames"]
            fps = vinfo["fps"]
            dur = vinfo["duration_sec"]

            # For multi-view, use the minimum total_frames across all videos
            if len(active_videos) > 1:
                all_infos = [get_video_info(v) for v in active_videos]
                total_frames = min(vi["total_frames"] for vi in all_infos)
                fps = all_infos[0]["fps"]  # Assume same FPS

            # Info bar
            st.markdown(
                '<div class="video-info">'
                f'<div class="video-info-item"><div class="label">Resolution</div>'
                f'<div class="value">{vinfo["width"]}×{vinfo["height"]}</div></div>'
                f'<div class="video-info-item"><div class="label">FPS</div>'
                f'<div class="value">{fps:.0f}</div></div>'
                f'<div class="video-info-item"><div class="label">Duration</div>'
                f'<div class="value">{int(dur // 60)}:{int(dur % 60):02d}</div></div>'
                f'<div class="video-info-item"><div class="label">Total Frames</div>'
                f'<div class="value">{total_frames:,}</div></div>'
                f'<div class="video-info-item"><div class="label">Cameras</div>'
                f'<div class="value">{len(active_videos)}</div></div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("", unsafe_allow_html=True)

            # ── Session state ──
            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0
            if "playing" not in st.session_state:
                st.session_state.playing = False

            # Clamp after a camera switch
            if st.session_state.frame_idx >= total_frames:
                st.session_state.frame_idx = 0

            # ── Controls row: slider + speed ──
            ctrl_col1, ctrl_col2 = st.columns([5, 1])
            with ctrl_col1:
                frame_idx = st.slider(
                    "Frame",
                    0,
                    max(0, total_frames - 1),
                    st.session_state.frame_idx,
                    key="frame_slider",
                )
                st.session_state.frame_idx = frame_idx
            with ctrl_col2:
                speed_options = {
                    "0.25×": 0.25,
                    "0.5×": 0.5,
                    "1×": 1.0,
                    "1.5×": 1.5,
                    "2×": 2.0,
                    "4×": 4.0,
                }
                selected_speed = st.selectbox(
                    "Speed",
                    list(speed_options.keys()),
                    index=2,  # default 1×
                    key="speed_select",
                )
                speed_multiplier = speed_options[selected_speed]

            # Navigation buttons
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                if st.button("⏮ Start", use_container_width=True):
                    st.session_state.frame_idx = 0
                    st.rerun()
            with b2:
                if st.button("◀ −50", use_container_width=True):
                    st.session_state.frame_idx = max(0, st.session_state.frame_idx - 50)
                    st.rerun()
            with b3:
                lbl = "⏸ Pause" if st.session_state.playing else "▶ Play"
                if st.button(lbl, use_container_width=True, type="primary"):
                    st.session_state.playing = not st.session_state.playing
                    st.rerun()
            with b4:
                if st.button("▶ +50", use_container_width=True):
                    st.session_state.frame_idx = min(total_frames - 1, st.session_state.frame_idx + 50)
                    st.rerun()
            with b5:
                if st.button("⏭ End", use_container_width=True):
                    st.session_state.frame_idx = total_frames - 1
                    st.rerun()

            # ── Display frame(s) ──
            frame_ph = st.empty()

            def render_current_frame(fidx: int) -> np.ndarray | None:
                """Render the current frame(s) based on view mode."""
                if len(active_videos) == 1:
                    return extract_frame(active_videos[0], fidx)
                else:
                    raw_frames = extract_synced_frames(active_videos, fidx)
                    labeled = []
                    for i, (f, v) in enumerate(zip(raw_frames, active_videos)):
                        if f is not None:
                            cam_name = camera_short_name(camera_id_from_video(v))
                            labeled.append(add_camera_label(f, cam_name))
                        else:
                            labeled.append(None)
                    grid_cols = 2
                    return stitch_grid(labeled, cols=grid_cols)

            if st.session_state.playing:
                # Multi-camera synchronized playback
                caps = []
                for v in active_videos:
                    cap = cv2.VideoCapture(str(v))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
                    caps.append(cap)

                delay = (1.0 / fps / speed_multiplier) if fps > 0 else 0.2
                batch = min(150, total_frames - st.session_state.frame_idx)

                for _ in range(batch):
                    bgr_frames = []
                    all_ok = True
                    for cap in caps:
                        ret, bgr = cap.read()
                        if not ret:
                            all_ok = False
                            break
                        bgr_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                    if not all_ok:
                        st.session_state.playing = False
                        break

                    cur = int(caps[0].get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    t = cur / fps if fps > 0 else 0

                    if len(bgr_frames) == 1:
                        display_img = bgr_frames[0]
                    else:
                        labeled = []
                        for i, (f, v) in enumerate(zip(bgr_frames, active_videos)):
                            cam_name = camera_short_name(camera_id_from_video(v))
                            labeled.append(add_camera_label(f, cam_name))
                        display_img = stitch_grid(labeled, cols=2)

                    frame_ph.image(
                        display_img,
                        caption=(
                            f"{int(t // 60)}:{int(t % 60):02d} — "
                            f"Frame {cur}/{total_frames} — {selected_speed} — {id_label_mode}"
                        ),
                        use_container_width=True,
                    )
                    st.session_state.frame_idx = cur
                    time.sleep(delay)

                for cap in caps:
                    cap.release()
                st.session_state.playing = False
                st.rerun()
            else:
                rgb = render_current_frame(st.session_state.frame_idx)
                if rgb is not None:
                    t = st.session_state.frame_idx / fps if fps > 0 else 0
                    frame_ph.image(
                        rgb,
                        caption=(
                            f"{int(t // 60)}:{int(t % 60):02d} — "
                            f"Frame {st.session_state.frame_idx}/{total_frames} — "
                            f"{id_label_mode}"
                        ),
                        use_container_width=True,
                    )
                else:
                    st.error("Could not extract the selected frame.")

            id_note = (
                "Each video displays local ByteTrack IDs. The same physical person may "
                "receive a different ID in another camera."
            )
            st.markdown(
                f'<div class="id-note"><strong>ID labels:</strong> {id_note}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════ TAB 3 — HEATMAP ════════════════════════════════
with tab_heatmap:
    st.markdown(
        '<div class="section-header"><h3>Traffic Density Heatmap</h3></div>',
        unsafe_allow_html=True,
    )

    if zone_events.empty:
        st.info("zone_events.csv not found or empty. Run notebook 02 first.")
    else:
        cam_ev = zone_events.copy()
        if selected_store and "store_id" in cam_ev.columns:
            cam_ev = cam_ev[cam_ev["store_id"].astype(str).eq(selected_store)]

        cams = sorted(cam_ev["camera_id"].unique()) if "camera_id" in cam_ev.columns else []

        if not cams:
            st.warning("No cameras available for the selected store.")
        else:
            # Hide dropdown when only one camera is available
            if len(cams) == 1:
                sel_cam = cams[0]
                st.caption(f"Camera: {sel_cam}")
            else:
                sel_cam = st.selectbox(
                    "Select camera for heatmap",
                    cams,
                    key="heatmap_cam",
                )

            if sel_cam:
                with st.spinner("Generating density heatmap..."):
                    # Convert filtered events to bytes for caching
                    ev_bytes = cam_ev.to_csv(index=False).encode()
                    heatmap_img = generate_camera_heatmap(
                        sel_cam, ev_bytes, str(RAW_DIR), str(VIDEOS_DIR)
                    )

                if heatmap_img is not None:
                    cam_rows = cam_ev[cam_ev["camera_id"] == sel_cam]
                    total_pts = len(cam_rows)
                    cust_pts = (
                        int(cam_rows["is_customer"].astype(str).str.lower().isin({"true", "1"}).sum())
                        if "is_customer" in cam_rows.columns
                        else total_pts
                    )
                    cam_num = camera_short_name(sel_cam)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.markdown(
                        metric_card_html("📌", f"{total_pts:,}", "Total Track Points", "blue"),
                        unsafe_allow_html=True,
                    )
                    mc2.markdown(
                        metric_card_html("👥", f"{cust_pts:,}", "Customer Points", "green"),
                        unsafe_allow_html=True,
                    )
                    mc3.markdown(
                        metric_card_html("📹", cam_num, "Camera", "purple"),
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.image(
                        heatmap_img,
                        caption=f"Density Heatmap — {sel_cam}",
                        use_container_width=True,
                    )

                    st.markdown(
                        '<div class="heatmap-legend">'
                        "🔴 Red = Very high traffic &nbsp;&nbsp;|&nbsp;&nbsp; "
                        "🟡 Yellow = Medium traffic &nbsp;&nbsp;|&nbsp;&nbsp; "
                        "🔵 Blue = Low traffic &nbsp;&nbsp;|&nbsp;&nbsp; "
                        "No color = No activity"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Video file not found for this camera.")

# ════════════════════════════ TAB 4 — TRAJECTORIES ═════════════════════════════
with tab_trajectories:
    st.markdown(
        '<div class="section-header"><h3>Camera-Local Movement Trajectories</h3></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Trajectory overlays are displayed one camera at a time. IDs are local to the "
        "selected camera and do not identify the same person across different cameras."
    )

    if zone_events.empty:
        st.info("zone_events.csv not found or empty. Run notebooks 02 and 03 first.")
    else:
        trajectory_events = zone_events.copy()
        if selected_store and "store_id" in trajectory_events.columns:
            trajectory_events = trajectory_events[
                trajectory_events["store_id"].astype(str).eq(selected_store)
            ].copy()
        required_trajectory_columns = {
            "camera_id",
            "camera_track_uid",
            "floor_x",
            "floor_y",
            "timestamp_sec",
            "frame_index",
        }
        if not required_trajectory_columns.issubset(trajectory_events.columns):
            missing = sorted(required_trajectory_columns - set(trajectory_events.columns))
            st.error(f"Trajectory data incomplete. Missing columns: {', '.join(missing)}")
        else:
            trajectory_cameras = sorted(
                trajectory_events["camera_id"].dropna().astype(str).unique()
            )
            if not trajectory_cameras:
                st.warning("No cameras available for the selected store.")
            else:
                selected_trajectory_camera = (
                    st.selectbox(
                        "Select camera for trajectories",
                        trajectory_cameras,
                        key="trajectory_camera",
                    )
                    if len(trajectory_cameras) > 1
                    else trajectory_cameras[0]
                )
                camera_trajectory_events = trajectory_events[
                    trajectory_events["camera_id"].astype(str).eq(selected_trajectory_camera)
                ].copy()
                if "is_customer" in camera_trajectory_events.columns:
                    camera_trajectory_events = camera_trajectory_events[
                        camera_trajectory_events["is_customer"]
                        .astype(str)
                        .str.lower()
                        .isin({"true", "1"})
                    ].copy()

                available_tracks = sorted(
                    camera_trajectory_events["camera_track_uid"].dropna().unique(),
                    key=lambda value: str(value),
                )
                if not available_tracks:
                    st.info("No valid customer trajectories for this camera.")
                else:
                    selected_tracks = st.multiselect(
                        "Select local tracks to display",
                        available_tracks,
                        default=available_tracks[: min(20, len(available_tracks))],
                        key=f"trajectory_tracks_{selected_trajectory_camera}",
                        help="Selecting fewer IDs makes the overlay clearer.",
                    )
                    displayed_events = camera_trajectory_events[
                        camera_trajectory_events["camera_track_uid"].isin(selected_tracks)
                    ].copy()
                    if displayed_events.empty:
                        st.info("Select at least one person to display their trajectory.")
                    else:
                        with st.spinner("Drawing trajectories..."):
                            trajectory_img = generate_camera_trajectory_overlay(
                                selected_trajectory_camera,
                                displayed_events.to_csv(index=False).encode(),
                                str(RAW_DIR),
                                str(VIDEOS_DIR),
                            )
                        if trajectory_img is None:
                            st.warning("Reference video not found for this camera.")
                        else:
                            st.image(
                                trajectory_img,
                                caption=(
                                    f"Trajectories of {len(selected_tracks)} local tracks — "
                                    f"{selected_trajectory_camera}"
                                ),
                                use_container_width=True,
                            )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header"><h3>Movement Metrics</h3></div>',
                    unsafe_allow_html=True,
                )
                if movement_metrics.empty:
                    st.info("No movement_metrics.csv yet. Run notebook 03 to generate it.")
                else:
                    camera_metrics = movement_metrics[
                        movement_metrics["camera_id"].astype(str).eq(selected_trajectory_camera)
                    ].copy()
                    if camera_metrics.empty:
                        st.info("No movement metrics for this camera.")
                    else:
                        display_cols = [
                            c for c in [
                                "camera_track_uid",
                                "zone_id",
                                "zone_label_ar",
                                "total_distance",
                                "dwell_time_per_zone",
                                "point_count",
                            ] if c in camera_metrics.columns
                        ]
                        display_metrics = camera_metrics[display_cols].sort_values(
                            ["camera_track_uid"] + (["zone_id"] if "zone_id" in display_cols else [])
                        )
                        rename_map = {
                            "camera_track_uid": "Camera-local Track ID",
                            "zone_id": "Zone ID",
                            "zone_label_ar": "Zone",
                            "total_distance": "Total Distance",
                            "dwell_time_per_zone": "Dwell Time (s)",
                            "point_count": "Track Points",
                        }
                        st.dataframe(
                            display_metrics.rename(columns=rename_map),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header"><h3>Zone Transition Flow</h3></div>',
                    unsafe_allow_html=True,
                )
                transition_figure = build_transition_sankey(zone_transitions)
                if transition_figure is None:
                    st.info("No cross-zone transitions found in the current data.")
                else:
                    st.plotly_chart(transition_figure, use_container_width=True)
                    st.dataframe(
                        zone_transitions.sort_values(
                            "transition_count", ascending=False
                        ).rename(
                            columns={
                                "from_zone_id": "From Zone",
                                "to_zone_id": "To Zone",
                                "transition_count": "People Count",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

# ══════════════════════════════ TAB 5 — QUEUES ═════════════════════════════════
with tab_queue:
    st.markdown(
        '<div class="section-header"><h3>Queue Analysis</h3></div>',
        unsafe_allow_html=True,
    )

    if queues.empty:
        st.info("No queue data available yet.")
    else:
        cur_q = int(queues.sort_values("window_start_sec").iloc[-1]["queue_length"])
        max_q = int(queues["queue_length"].max())
        avg_q = round(queues["queue_length"].mean(), 1)
        pred_q = int(queues.sort_values("window_start_sec").iloc[-1]["predicted_queue_next_window"])

        qc = st.columns(4)
        qc[0].markdown(metric_card_html("🚶", str(cur_q), "Current Queue", "blue"), unsafe_allow_html=True)
        qc[1].markdown(metric_card_html("📈", str(max_q), "Peak Length", "red"), unsafe_allow_html=True)
        qc[2].markdown(metric_card_html("📊", str(avg_q), "Average", "green"), unsafe_allow_html=True)
        qc[3].markdown(metric_card_html("🔮", str(pred_q), "Next Prediction", "purple"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_q = go.Figure()
        fig_q.add_trace(
            go.Scatter(
                x=queues["window_start"],
                y=queues["queue_length"],
                mode="lines+markers",
                name="Queue Length",
                line=dict(color="#4f6ef7", width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(79,110,247,0.08)",
            )
        )
        fig_q.add_trace(
            go.Scatter(
                x=queues["window_start"],
                y=queues["predicted_queue_next_window"],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
        apply_dark_layout(fig_q, "Actual Queue vs. Prediction")
        fig_q.update_layout(xaxis_title="Time", yaxis_title="People")
        st.plotly_chart(fig_q, use_container_width=True)

        with st.expander("Detailed Data"):
            st.dataframe(
                queues.sort_values("window_start_sec", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

# ══════════════════════════════ TAB 6 — ANALYTICS ══════════════════════════════
with tab_analytics:
    st.markdown(
        '<div class="section-header"><h3>Detailed Analytics</h3></div>',
        unsafe_allow_html=True,
    )

    if traffic.empty:
        st.info("No analytics data available yet.")
    else:
        label_col = "zone_label_ar" if "zone_label_ar" in traffic.columns else "zone_id"
        fig_line = px.line(
            traffic,
            x="window_start",
            y="customer_count",
            color=label_col,
            markers=True,
            color_discrete_sequence=["#4f6ef7", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"],
            labels={
                "window_start": "Time",
                "customer_count": "Camera-local Track Count",
                label_col: "Zone",
            },
        )
        apply_dark_layout(fig_line, "Traffic Over Time")
        fig_line.update_traces(line_width=2)
        st.plotly_chart(fig_line, use_container_width=True)

        # Zone comparison
        if traffic["zone_id"].nunique() > 1:
            zs = traffic.groupby(label_col).agg(
                avg=("customer_count", "mean"),
                peak=("customer_count", "max"),
            ).round(1)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(
                go.Bar(x=zs.index, y=zs["avg"], name="Average", marker_color="#4f6ef7")
            )
            fig_cmp.add_trace(
                go.Bar(x=zs.index, y=zs["peak"], name="Peak", marker_color="#ef4444")
            )
            apply_dark_layout(fig_cmp, "Zone Comparison — Average vs. Peak")
            fig_cmp.update_layout(
                barmode="group",
                xaxis_title="Zone",
                yaxis_title="Camera-local Track Count",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><h3>People Count by Video Quarter</h3></div>',
            unsafe_allow_html=True,
        )
        if video_quarter_counts.empty:
            st.info(
                "No video quarter data yet. Run notebook 01 then notebook 03 "
                "to compute people counts per quarter."
            )
        else:
            quarter_cameras = sorted(
                video_quarter_counts["camera_id"].dropna().astype(str).unique()
            )
            selected_quarter_camera = (
                st.selectbox(
                    "Camera for video quarters",
                    quarter_cameras,
                    key="quarter_camera",
                )
                if len(quarter_cameras) > 1
                else quarter_cameras[0]
            )
            camera_quarters = (
                video_quarter_counts[
                    video_quarter_counts["camera_id"].astype(str).eq(selected_quarter_camera)
                ]
                .sort_values("video_quarter")
                .copy()
            )

            quarter_label_col = "quarter_label_ar" if "quarter_label_ar" in camera_quarters.columns else "video_quarter"
            # Add English labels
            quarter_en = {
                "الربع الأول": "Q1",
                "الربع الثاني": "Q2",
                "الربع الثالث": "Q3",
                "الربع الرابع": "Q4",
            }
            if quarter_label_col == "quarter_label_ar":
                camera_quarters["quarter_label"] = camera_quarters[quarter_label_col].map(
                    lambda x: quarter_en.get(x, x)
                )
                ql_col = "quarter_label"
            else:
                ql_col = quarter_label_col

            fig_quarters = px.bar(
                camera_quarters,
                x=ql_col,
                y="unique_camera_tracks",
                text="unique_camera_tracks",
                hover_data={
                    c: True for c in ["frame_range", "time_range", "quarter_start_frame",
                                       "quarter_end_frame", "observations"]
                    if c in camera_quarters.columns
                },
                labels={
                    ql_col: "Video Quarter",
                    "unique_camera_tracks": "Camera-local Tracks",
                },
                color_discrete_sequence=["#8b5cf6"],
            )
            apply_dark_layout(fig_quarters, "People Count per Video Quarter")
            fig_quarters.update_traces(textposition="outside")
            fig_quarters.update_layout(yaxis_title="People", xaxis_title="Quarter")
            st.plotly_chart(fig_quarters, use_container_width=True)
            st.caption(
                "Count = unique customer IDs within each quarter. "
                "frame_range shows the start/end frame of each quarter."
            )
            with st.expander("Quarter Frame Details"):
                detail_cols = [c for c in [
                    ql_col, "frame_range", "time_range",
                    "unique_camera_tracks", "observations"
                ] if c in camera_quarters.columns]
                st.dataframe(
                    camera_quarters[detail_cols],
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Detailed Traffic Data"):
            st.dataframe(
                traffic.sort_values("window_start_sec", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

# ══════════════════════════════ TAB 7 — ALERTS ═════════════════════════════════
with tab_alerts:
    st.markdown(
        '<div class="section-header"><h3>Alert Log</h3></div>',
        unsafe_allow_html=True,
    )

    if alerts.empty:
        st.success("No critical alerts recorded. Everything is operating normally.")
    else:
        if "severity" in alerts.columns:
            sev = alerts["severity"].value_counts()
            ac = st.columns(3)
            ac[0].markdown(
                metric_card_html("🔴", str(sev.get("high", 0)), "High Alerts", "red"),
                unsafe_allow_html=True,
            )
            ac[1].markdown(
                metric_card_html("🟡", str(sev.get("medium", 0)), "Medium Alerts", "orange"),
                unsafe_allow_html=True,
            )
            ac[2].markdown(
                metric_card_html("🟢", str(sev.get("low", 0)), "Low Alerts", "green"),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            alerts.sort_values("created_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════ TAB 8 — CHAT ═══════════════════════════════════
with tab_chat:
    st.markdown(
        '<div class="section-header"><h3>Query Your Store Data</h3></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chat-examples">'
        '<div class="ex-title">Example Questions</div>'
        '<div class="ex-list">'
        "• When was the peak traffic?<br>"
        "• How many people are at the checkout?<br>"
        "• Tell me about the queue<br>"
        "• How many visitors at the entrance?<br>"
        "• Compare average customers between zones"
        "</div></div>",
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. When was the busiest time?",
        key="chat_input",
    )

    if question:
        fast_path_result = rule_based_query(question, traffic, queues)

        if fast_path_result is not None:
            answer, result = fast_path_result
            st.caption("Fast path — no LLM needed.")
            st.markdown(
                insight_card_html("Answer", answer, "#10b981"),
                unsafe_allow_html=True,
            )
            if result is not None and not result.empty:
                st.dataframe(result, use_container_width=True, hide_index=True)
        elif not database_ready:
            st.error(
                "DuckDB database not found or stale. Run notebook 03 after the "
                "latest local-tracking run."
            )
        else:
            with st.spinner("Analyzing with local LLM..."):
                outcome = cached_llm_query(
                    question,
                    DB_PATH.stat().st_mtime_ns,
                )

            st.caption(f"LLM fallback using `{OLLAMA_MODEL}`.")
            retry_count = int(outcome.get("retry_count", 0))
            if retry_count:
                st.caption(f"SQL correction attempts: {retry_count}")

            if not outcome.get("success"):
                st.error(outcome.get("error", "An unknown error occurred."))
            else:
                st.markdown(
                    insight_card_html(
                        "Answer",
                        outcome["answer"],
                        "#4f6ef7",
                    ),
                    unsafe_allow_html=True,
                )

                result = outcome["result"]
                chart = build_chart_from_decision(
                    result,
                    outcome["chart_decision"],
                )
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)

                if not result.empty:
                    with st.expander("Query Result"):
                        st.dataframe(
                            result,
                            use_container_width=True,
                            hide_index=True,
                        )

            if outcome.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(outcome["sql"], language="sql")
