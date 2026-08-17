"""
Smart Crop Growth Tracking System
==================================
A beginner-friendly, self-contained Streamlit application that detects plants
in field images/video, assigns persistent IDs, tracks them across frames,
estimates growth stage & canopy area, stores historical measurements, and
displays a per-plant growth dashboard.

No API keys. No paid services. Runs fully offline.

--------------------------------------------------------------------------
ARCHITECTURE (designed so pieces can be swapped later without a rewrite)
--------------------------------------------------------------------------
1. Detector            -> abstract interface: detect(frame) -> list[Box]
     - OpenCVGreenDetector : real, works out of the box, no downloads (DEFAULT)
     - YOLODetector        : real YOLO path, auto-used ONLY if the optional
                              `ultralytics` package AND a local .pt weights
                              file are available. Falls back to OpenCV
                              otherwise. This is how you "add YOLO later"
                              without touching any other code.
2. PlantTracker         -> IoU + centroid-distance greedy tracker.
                            Assigns and persists integer plant IDs across
                            frames/uploads within a session.
3. GrowthEstimator      -> maps canopy pixel area -> growth stage using
                            simple, tunable thresholds (swap for a trained
                            regressor/classifier later).
4. HistoryStore         -> append-only in-memory table of measurements.
                            Structured as a list of dict rows, exactly the
                            shape a SQL table would have, so swapping this
                            for SQLite/Postgres later is a drop-in change
                            (see the `HistoryStore` docstring for the plan).
5. Camera / video input -> reads uploaded images/video today. The same
                            `iter_frames()` interface can point at
                            `cv2.VideoCapture(0)` for a real webcam/field
                            camera later with no change to the rest of the
                            pipeline.
6. DemoFieldSimulator   -> generates a synthetic but realistic growing
                            field video in real time so the app has a
                            genuine live demo the instant it starts,
                            with no uploads required.
"""

import io
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================================
# PAGE CONFIG (must be first Streamlit call)
# =========================================================================
st.set_page_config(
    page_title="Smart Crop Growth Tracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# TRY TO IMPORT YOLO (OPTIONAL — app works perfectly fine without it)
# =========================================================================
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # noqa: F401
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# =========================================================================
# 1. DATA STRUCTURES
# =========================================================================
@dataclass
class Box:
    """A single detected plant bounding box for one frame."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    area_px: int

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class TrackedPlant:
    """Persistent state for one plant across frames."""
    plant_id: int
    box: Box
    first_seen: int          # frame index
    last_seen: int           # frame index
    misses: int = 0
    color: tuple = field(default_factory=lambda: tuple(
        int(c) for c in np.random.randint(60, 255, size=3)
    ))


# =========================================================================
# 2. DETECTORS
# =========================================================================
class BaseDetector:
    """Common interface every detector must implement."""

    name = "base"

    def detect(self, frame_bgr: np.ndarray):
        """Return a list[Box] for one BGR frame."""
        raise NotImplementedError


class OpenCVGreenDetector(BaseDetector):
    """
    Real, dependency-free plant detector using classic computer vision:
    HSV color thresholding (isolate green vegetation) + contour extraction.

    This is the DEFAULT detector because it requires no model download and
    no API key, so the app has a genuine working detector the instant it
    starts. It is intentionally simple so beginners can read and tune it.
    """

    name = "OpenCV Green-Mask Detector"

    def __init__(self, min_area=180, lower_green=(25, 35, 35), upper_green=(95, 255, 255)):
        self.min_area = min_area
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)

    def detect(self, frame_bgr: np.ndarray):
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # Clean up noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)

                # Confidence heuristic: how "filled" the box is by green
                # pixels (solidity) blended with a size prior. This stands
                # in for a model's softmax confidence. Swappable later.
                rect_area = w * h
                solidity = float(area) / float(rect_area + 1e-6)
                size_score = min(area / 4000.0, 1.0)
                confidence = float(np.clip(0.35 * size_score + 0.65 * solidity, 0.05, 0.99))

                boxes.append(Box(x, y, w, h, confidence, int(area)))
            return boxes
        except Exception as e:
            st.session_state.setdefault("errors", []).append(f"Detector error: {e}")
            return []


class YOLODetector(BaseDetector):
    """
    Real YOLO detection path. Only activates if `ultralytics` is installed
    AND a local weights file path is supplied. This lets you drop in a
    custom-trained crop/plant YOLO model later (milestone 9) by pointing
    `weights_path` at your .pt file — no other code changes needed.

    NOTE: no weights ship with this app and none are downloaded
    automatically, so no network access or API key is ever required.
    """

    name = "YOLO Detector"

    def __init__(self, weights_path: str, conf=0.25):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray):
        try:
            results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    w, h = x2 - x1, y2 - y1
                    boxes.append(Box(x1, y1, w, h, conf, w * h))
            return boxes
        except Exception as e:
            st.session_state.setdefault("errors", []).append(f"YOLO detector error: {e}")
            return []


def get_detector() -> BaseDetector:
    """
    Factory that decides which detector to use.
    Defaults to the dependency-free OpenCV detector. Will only try YOLO if
    the user has both installed `ultralytics` and provided weights in the
    sidebar — otherwise it silently and safely falls back.
    """
    weights_path = st.session_state.get("yolo_weights_path")
    if YOLO_AVAILABLE and weights_path:
        try:
            return YOLODetector(weights_path, conf=st.session_state.get("conf_threshold", 0.25))
        except Exception as e:
            st.session_state.setdefault("errors", []).append(
                f"Could not load YOLO weights ({e}). Falling back to OpenCV detector."
            )
    return OpenCVGreenDetector(min_area=st.session_state.get("min_area", 180))


# =========================================================================
# 3. TRACKER — assigns & persists plant IDs across frames
# =========================================================================
class PlantTracker:
    """
    Lightweight multi-object tracker using IoU + centroid distance greedy
    matching. Good enough for slow-moving / static field footage where
    plants don't jump far between frames. Swap for SORT/DeepSORT/ByteTrack
    later without touching the rest of the app — only this class changes.
    """

    def __init__(self, iou_threshold=0.15, max_distance=140, max_misses=8):
        self.tracks: dict[int, TrackedPlant] = {}
        self._next_id = 1
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.frame_idx = 0

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a.xyxy
        bx1, by1, bx2, by2 = b.xyxy
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list) -> dict:
        """Match new detections to existing tracks, return current tracks."""
        self.frame_idx += 1
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Build a score for every (track, detection) pair
        pairs = []
        for tid, track in self.tracks.items():
            for di in unmatched_dets:
                det = detections[di]
                iou = self._iou(track.box, det)
                dist = np.hypot(track.box.cx - det.cx, track.box.cy - det.cy)
                if iou >= self.iou_threshold or dist <= self.max_distance:
                    score = iou - (dist / 10000.0)  # prefer high IoU, then close distance
                    pairs.append((score, tid, di))

        pairs.sort(key=lambda p: p[0], reverse=True)
        used_dets = set()
        for score, tid, di in pairs:
            if tid in matched_track_ids or di in used_dets:
                continue
            det = detections[di]
            track = self.tracks[tid]
            track.box = det
            track.last_seen = self.frame_idx
            track.misses = 0
            matched_track_ids.add(tid)
            used_dets.add(di)

        # New tracks for unmatched detections
        for di in range(len(detections)):
            if di in used_dets:
                continue
            det = detections[di]
            new_id = self._next_id
            self._next_id += 1
            self.tracks[new_id] = TrackedPlant(
                plant_id=new_id, box=det,
                first_seen=self.frame_idx, last_seen=self.frame_idx,
            )

        # Age out tracks that were not matched this frame
        dead = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids and track.last_seen != self.frame_idx:
                track.misses += 1
                if track.misses > self.max_misses:
                    dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        return self.tracks


# =========================================================================
# 4. GROWTH ESTIMATOR
# =========================================================================
class GrowthEstimator:
    """
    Maps canopy pixel area to a human-readable growth stage using simple
    thresholds. Deliberately simple & tunable — swap for a trained
    regression model later (milestone 9) by replacing `stage_for_area`.
    """

    STAGES = ["Seedling", "Vegetative", "Flowering", "Mature"]

    def __init__(self, thresholds=(600, 2200, 5000)):
        # area < t0 -> Seedling, < t1 -> Vegetative, < t2 -> Flowering, else Mature
        self.t0, self.t1, self.t2 = thresholds

    def stage_for_area(self, area_px: int) -> str:
        if area_px < self.t0:
            return "Seedling"
        elif area_px < self.t1:
            return "Vegetative"
        elif area_px < self.t2:
            return "Flowering"
        return "Mature"

    def stage_progress(self, area_px: int) -> float:
        """0-100% progress through the full seedling->mature range, for a progress bar."""
        return float(np.clip(area_px / self.t2 * 100, 0, 100))


# =========================================================================
# 5. HISTORY STORE
# =========================================================================
class HistoryStore:
    """
    Append-only measurement log kept as a list of dict rows — exactly the
    shape a SQL table (`plant_measurements`) would have:

        plant_id | timestamp | frame | canopy_area_px | growth_stage |
        confidence | source

    TO ADD A REAL DATABASE LATER:
      1. Create a table with the columns above (SQLite/Postgres/etc).
      2. Replace `self.rows.append(row)` in `add()` with an INSERT.
      3. Replace `to_dataframe()` with a `SELECT * FROM plant_measurements`
         read into pandas via `pd.read_sql`.
      Nothing else in the app needs to change because everything downstream
      only ever talks to `to_dataframe()`.
    """

    def __init__(self):
        self.rows = []

    def add(self, plant_id, area_px, stage, confidence, source, frame_idx):
        self.rows.append({
            "plant_id": plant_id,
            "timestamp": datetime.now(),
            "frame": frame_idx,
            "canopy_area_px": area_px,
            "growth_stage": stage,
            "confidence": round(confidence, 3),
            "source": source,
        })

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=[
                "plant_id", "timestamp", "frame", "canopy_area_px",
                "growth_stage", "confidence", "source"
            ])
        return pd.DataFrame(self.rows)

    def latest_snapshot(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.sort_values("timestamp").groupby("plant_id").tail(1).reset_index(drop=True)


# =========================================================================
# 6. DEMO FIELD SIMULATOR — makes a real, live, growing synthetic field
# =========================================================================
class DemoFieldSimulator:
    """
    Generates synthetic-but-realistic top-down field frames with plants
    that visibly grow over time (radius increases frame by frame, with a
    little jitter and per-plant variation), so the OpenCV detector has
    real green blobs to detect the instant the app opens. This gives a
    genuine live, interactive demo with zero uploads and zero setup.
    """

    def __init__(self, n_plants=8, width=640, height=420, seed=42):
        rng = np.random.default_rng(seed)
        self.width, self.height = width, height
        self.plants = []
        margin = 60
        cols = 4
        rows = int(np.ceil(n_plants / cols))
        xs = np.linspace(margin, width - margin, cols)
        ys = np.linspace(margin, height - margin, rows)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n_plants:
                    break
                cx = xs[c] + rng.integers(-15, 15)
                cy = ys[r] + rng.integers(-15, 15)
                growth_rate = rng.uniform(0.55, 1.35)
                start_radius = rng.uniform(4, 10)
                self.plants.append({
                    "cx": cx, "cy": cy,
                    "growth_rate": growth_rate,
                    "radius": start_radius,
                    "wobble_seed": rng.uniform(0, 6.28),
                })
                idx += 1
        self.t = 0

    def _soil_background(self):
        base = np.full((self.height, self.width, 3), (45, 65, 95), dtype=np.uint8)  # brownish BGR
        noise = np.random.randint(-8, 8, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # subtle furrow lines
        for y in range(0, self.height, 26):
            cv2.line(img, (0, y), (self.width, y), (35, 52, 78), 1)
        return img

    def next_frame(self) -> np.ndarray:
        self.t += 1
        img = self._soil_background()
        for p in self.plants:
            # grow slowly, with a light breathing wobble so it's visibly "live"
            p["radius"] = min(p["radius"] + p["growth_rate"] * 0.10, 55)
            wobble = 1.0 + 0.03 * np.sin(self.t * 0.15 + p["wobble_seed"])
            r = max(3, int(p["radius"] * wobble))
            green = (40, int(np.clip(120 + p["radius"] * 1.3, 120, 235)), 55)
            cv2.circle(img, (int(p["cx"]), int(p["cy"])), r, green, -1, lineType=cv2.LINE_AA)
            # a few leaf-like blobs around center for texture at larger sizes
            if r > 14:
                for k in range(3):
                    ang = self.t * 0.05 + k * 2.1 + p["wobble_seed"]
                    lx = int(p["cx"] + np.cos(ang) * r * 0.6)
                    ly = int(p["cy"] + np.sin(ang) * r * 0.6)
                    cv2.circle(img, (lx, ly), max(2, r // 4), green, -1, lineType=cv2.LINE_AA)
        return img


# =========================================================================
# 7. VIDEO / IMAGE INPUT HELPERS
# =========================================================================
def read_image_upload(uploaded_file) -> np.ndarray:
    """Decode a Streamlit UploadedFile (image) into a BGR numpy array."""
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupted or an unsupported format.")
    return img


def iter_video_frames(uploaded_file, max_frames=120, stride=3):
    """
    Yield BGR frames from an uploaded video file.
    `stride` skips frames for speed on long clips (beginner-friendly default).
    Designed so a live camera (`cv2.VideoCapture(0)`) can be substituted
    later with the exact same downstream loop.
    """
    tmp_path = f"/tmp/_upload_{uuid.uuid4().hex}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try re-exporting as MP4 (H.264).")

    count, yielded = 0, 0
    try:
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if count % stride == 0:
                yield frame
                yielded += 1
            count += 1
    finally:
        cap.release()


# =========================================================================
# 8. DRAWING HELPERS
# =========================================================================
def draw_tracks(frame_bgr, tracks: dict, estimator: GrowthEstimator):
    out = frame_bgr.copy()
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t.box.xyxy
        color = t.color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        stage = estimator.stage_for_area(t.box.area_px)
        label = f"ID {tid} | {stage} | {t.box.confidence*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================================================================
# 9. SESSION STATE INITIALIZATION
# =========================================================================
def init_state():
    defaults = {
        "tracker": PlantTracker(),
        "estimator": GrowthEstimator(),
        "history": HistoryStore(),
        "simulator": DemoFieldSimulator(),
        "errors": [],
        "demo_running": True,        # live demo starts automatically
        "min_area": 180,
        "conf_threshold": 0.25,
        "yolo_weights_path": None,
        "last_frame_rgb": None,
        "last_tracks": {},
        "mode": "🎬 Live Demo",
        "frames_processed": 0,
        "demo_seeded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def seed_demo_history():
    """
    Pre-populates several frames of simulated growth history immediately on
    first load, so charts and tables are populated with realistic data
    right away instead of a blank dashboard.
    """
    if st.session_state.demo_seeded:
        return
    sim = st.session_state.simulator
    tracker = st.session_state.tracker
    estimator = st.session_state.estimator
    history = st.session_state.history
    detector = OpenCVGreenDetector(min_area=st.session_state.min_area)

    frame = None
    for _ in range(35):  # fast-forward growth so there's a real history trail
        frame = sim.next_frame()
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for tid, t in tracks.items():
            stage = estimator.stage_for_area(t.box.area_px)
            history.add(tid, t.box.area_px, stage, t.box.confidence, "Live Demo", tracker.frame_idx)

    st.session_state.last_tracks = tracks
    st.session_state.last_frame_rgb = bgr_to_rgb(draw_tracks(frame, tracks, estimator))
    st.session_state.frames_processed = tracker.frame_idx
    st.session_state.demo_seeded = True


init_state()
seed_demo_history()

# =========================================================================
# 10. SIDEBAR — CONTROLS
# =========================================================================
with st.sidebar:
    st.markdown("## 🌱 Smart Crop Tracker")
    st.caption("Beginner-friendly plant detection & growth tracking — 100% offline, no API keys.")

    st.session_state.mode = st.radio(
        "Input source",
        ["🎬 Live Demo", "🖼️ Upload Image", "📹 Upload Video"],
        index=["🎬 Live Demo", "🖼️ Upload Image", "📹 Upload Video"].index(st.session_state.mode),
    )

    st.divider()
    st.markdown("### ⚙️ Detection settings")
    st.session_state.min_area = st.slider(
        "Minimum plant size (pixels²)", 50, 1500, st.session_state.min_area, step=10,
        help="Smaller values detect tinier seedlings but pick up more noise."
    )

    st.markdown("### 🧠 Detector engine")
    if YOLO_AVAILABLE:
        st.success("`ultralytics` is installed — you can plug in a YOLO .pt weights file below.")
        weights = st.text_input("Path to YOLO weights (.pt) — optional", value="")
        st.session_state.yolo_weights_path = weights.strip() or None
        st.session_state.conf_threshold = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25)
    else:
        st.info(
            "Using the built-in **OpenCV detector** (no downloads needed).\n\n"
            "Install `ultralytics` and provide a `.pt` weights file to switch to a "
            "real YOLO model later — the rest of the app needs no changes."
        )

    st.divider()
    st.markdown("### 📈 Growth stage thresholds (px² canopy area)")
    t0 = st.number_input("Seedling → Vegetative", value=600, step=50)
    t1 = st.number_input("Vegetative → Flowering", value=2200, step=50)
    t2 = st.number_input("Flowering → Mature", value=5000, step=100)
    st.session_state.estimator = GrowthEstimator(thresholds=(t0, t1, t2))

    st.divider()
    if st.session_state.errors:
        with st.expander(f"⚠️ {len(st.session_state.errors)} warning(s)/error(s)"):
            for e in st.session_state.errors[-10:]:
                st.warning(e)
            if st.button("Clear errors"):
                st.session_state.errors = []
                st.rerun()

    st.divider()
    if st.button("🔄 Reset session (clear tracks & history)", use_container_width=True):
        st.session_state.tracker = PlantTracker()
        st.session_state.history = HistoryStore()
        st.session_state.simulator = DemoFieldSimulator()
        st.session_state.demo_seeded = False
        st.session_state.frames_processed = 0
        st.rerun()

# =========================================================================
# 11. HEADER
# =========================================================================
st.title("🌱 Smart Crop Growth Tracking System")
st.caption(
    "Upload field images/video — or watch the live demo below — to detect plants, "
    "track them individually across frames, and monitor canopy growth over time."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
current_tracks = st.session_state.last_tracks
kpi1.metric("Plants currently tracked", len(current_tracks))
kpi2.metric("Frames processed", st.session_state.frames_processed)
kpi3.metric(
    "Avg. confidence",
    f"{np.mean([t.box.confidence for t in current_tracks.values()])*100:.0f}%"
    if current_tracks else "—"
)
kpi4.metric("Detector engine", "YOLO" if isinstance(get_detector(), YOLODetector) else "OpenCV")

st.divider()

# =========================================================================
# 12. MAIN CONTENT — MODE HANDLING
# =========================================================================
detector = get_detector()
estimator = st.session_state.estimator
tracker = st.session_state.tracker
history = st.session_state.history

left, right = st.columns([1.15, 1])

# -------------------------------------------------------------------------
# MODE: LIVE DEMO
# -------------------------------------------------------------------------
def _advance_demo(n_steps):
    """Advance the simulator/detector/tracker/history by n_steps frames."""
    frame = None
    tracks = st.session_state.last_tracks
    for _ in range(n_steps):
        frame = st.session_state.simulator.next_frame()
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for tid, t in tracks.items():
            stage = estimator.stage_for_area(t.box.area_px)
            history.add(tid, t.box.area_px, stage, t.box.confidence,
                        "Live Demo", tracker.frame_idx)
    if frame is not None:
        st.session_state.last_tracks = tracks
        st.session_state.last_frame_rgb = bgr_to_rgb(draw_tracks(frame, tracks, estimator))
        st.session_state.frames_processed = tracker.frame_idx


@st.fragment(run_every="0.8s")
def _live_demo_fragment(speed):
    """
    Auto-refreshing fragment: reruns itself on a timer (independent of the
    rest of the page) so plants visibly grow without a manual click and
    without resetting sidebar widgets or reloading the whole app each tick.
    """
    if st.session_state.demo_running:
        try:
            _advance_demo(speed)
        except Exception as e:
            st.session_state.errors.append(f"Live demo error: {e}")
            st.error(f"Something went wrong advancing the demo: {e}")

    if st.session_state.last_frame_rgb is not None:
        st.image(st.session_state.last_frame_rgb, use_container_width=True,
                  caption=f"Frame {st.session_state.frames_processed}")

    if st.session_state.demo_running:
        st.caption("🟢 Playing live — plants are growing automatically…")
    else:
        st.caption("⏸️ Paused. Toggle **Auto-play** above or click **Advance one step manually**.")


if st.session_state.mode == "🎬 Live Demo":
    with left:
        st.subheader("🎬 Live synthetic field demo")
        st.caption(
            "A generated field video streams here in real time — this is a genuine "
            "detect → track → measure loop running on synthetic plants that actually "
            "grow, so you get a working demo immediately with no uploads needed."
        )
        run = st.toggle("▶️ Auto-play (plants grow on their own)", value=st.session_state.demo_running)
        st.session_state.demo_running = run
        speed = st.slider("Playback speed (frames advanced per tick)", 1, 10, 3)

        if st.button("⏭️ Advance one step manually", disabled=run):
            try:
                _advance_demo(speed)
            except Exception as e:
                st.session_state.errors.append(f"Live demo error: {e}")
                st.error(f"Something went wrong advancing the demo: {e}")

        _live_demo_fragment(speed)

    with right:
        st.subheader("📋 Currently tracked plants")
        if current_tracks:
            rows = []
            for tid, t in sorted(current_tracks.items()):
                rows.append({
                    "Plant ID": tid,
                    "Growth stage": estimator.stage_for_area(t.box.area_px),
                    "Canopy area (px²)": t.box.area_px,
                    "Confidence": f"{t.box.confidence*100:.0f}%",
                    "Progress": estimator.stage_progress(t.box.area_px),
                })
            df_now = pd.DataFrame(rows)
            st.dataframe(
                df_now,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Progress": st.column_config.ProgressColumn(
                        "Growth progress", min_value=0, max_value=100, format="%.0f%%"
                    )
                },
            )
        else:
            st.info("No plants tracked yet — click **Advance demo** on the left.")

# -------------------------------------------------------------------------
# MODE: UPLOAD IMAGE
# -------------------------------------------------------------------------
elif st.session_state.mode == "🖼️ Upload Image":
    with left:
        st.subheader("🖼️ Upload a field image")
        uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            try:
                img = read_image_upload(uploaded)
                detections = detector.detect(img)
                tracks = tracker.update(detections)
                for tid, t in tracks.items():
                    stage = estimator.stage_for_area(t.box.area_px)
                    history.add(tid, t.box.area_px, stage, t.box.confidence,
                                uploaded.name, tracker.frame_idx)
                annotated = draw_tracks(img, tracks, estimator)
                st.session_state.last_tracks = tracks
                st.session_state.last_frame_rgb = bgr_to_rgb(annotated)
                st.session_state.frames_processed = tracker.frame_idx
                st.image(bgr_to_rgb(annotated), use_container_width=True,
                         caption=f"Detected {len(detections)} plant(s)")
                if not detections:
                    st.warning(
                        "No plants detected. Try lowering **Minimum plant size** in the "
                        "sidebar, or use an image with clearly visible green foliage."
                    )
            except Exception as e:
                st.session_state.errors.append(f"Image upload error: {e}")
                st.error(f"Couldn't process that image: {e}")
        else:
            st.info("👆 Upload a JPG/PNG field image to run detection.")

    with right:
        st.subheader("📋 Detections in this image")
        if current_tracks and st.session_state.mode == "🖼️ Upload Image":
            rows = [{
                "Plant ID": tid,
                "Growth stage": estimator.stage_for_area(t.box.area_px),
                "Canopy area (px²)": t.box.area_px,
                "Confidence": f"{t.box.confidence*100:.0f}%",
            } for tid, t in sorted(current_tracks.items())]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# MODE: UPLOAD VIDEO
# -------------------------------------------------------------------------
else:
    with left:
        st.subheader("📹 Upload a field video")
        st.caption("Short clips work best. The app samples frames for speed (beginner-friendly default).")
        uploaded = st.file_uploader("Choose a video (MP4/MOV)", type=["mp4", "mov", "avi", "m4v"])
        max_frames = st.slider("Max frames to process", 10, 300, 60, step=10)
        stride = st.slider("Frame stride (skip N-1 frames each step)", 1, 10, 3)

        if uploaded is not None:
            if st.button("▶️ Process video", type="primary"):
                progress = st.progress(0, text="Starting…")
                frame_slot = st.empty()
                last_annotated = None
                try:
                    processed = 0
                    for frame in iter_video_frames(uploaded, max_frames=max_frames, stride=stride):
                        detections = detector.detect(frame)
                        tracks = tracker.update(detections)
                        for tid, t in tracks.items():
                            stage = estimator.stage_for_area(t.box.area_px)
                            history.add(tid, t.box.area_px, stage, t.box.confidence,
                                        uploaded.name, tracker.frame_idx)
                        last_annotated = draw_tracks(frame, tracks, estimator)
                        processed += 1
                        progress.progress(min(processed / max_frames, 1.0),
                                           text=f"Processed {processed} frame(s)…")
                        if processed % 3 == 0:
                            frame_slot.image(bgr_to_rgb(last_annotated), use_container_width=True)

                    st.session_state.last_tracks = tracks
                    st.session_state.last_frame_rgb = bgr_to_rgb(last_annotated) if last_annotated is not None else None
                    st.session_state.frames_processed = tracker.frame_idx
                    progress.progress(1.0, text="Done.")
                    st.success(f"Finished. Processed {processed} frames, tracked {len(tracks)} plant(s).")
                    if last_annotated is not None:
                        frame_slot.image(bgr_to_rgb(last_annotated), use_container_width=True)
                except Exception as e:
                    st.session_state.errors.append(f"Video processing error: {e}")
                    st.error(
                        f"Couldn't process that video: {e}\n\n"
                        "Tip: try re-exporting as MP4 (H.264) or a shorter clip."
                    )
        else:
            st.info("👆 Upload an MP4/MOV field video to run detection & tracking across frames.")

    with right:
        st.subheader("📋 Detections (latest frame)")
        if current_tracks:
            rows = [{
                "Plant ID": tid,
                "Growth stage": estimator.stage_for_area(t.box.area_px),
                "Canopy area (px²)": t.box.area_px,
                "Confidence": f"{t.box.confidence*100:.0f}%",
            } for tid, t in sorted(current_tracks.items())]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Process a video to see per-plant results here.")

st.divider()

# =========================================================================
# 13. DASHBOARD — HISTORY, CHARTS, ALERTS, EXPORT
# =========================================================================
st.header("📊 Growth Dashboard")

hist_df = history.to_dataframe()

if hist_df.empty:
    st.info(
        "No measurements yet. Run the **Live Demo** (advance a few frames) or upload "
        "an image/video to start building growth history."
    )
else:
    tab_overview, tab_charts, tab_alerts, tab_data = st.tabs(
        ["📈 Overview", "🌿 Per-plant growth", "🔔 Alerts", "🗂️ Raw data & export"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        snapshot = history.latest_snapshot()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total plants ever tracked", hist_df["plant_id"].nunique())
        c2.metric("Total measurements logged", len(hist_df))
        stage_counts = snapshot["growth_stage"].value_counts()
        most_common_stage = stage_counts.idxmax() if not stage_counts.empty else "—"
        c3.metric("Most common current stage", most_common_stage)

        st.markdown("##### Growth stage distribution (current)")
        stage_order = GrowthEstimator.STAGES
        counts = snapshot["growth_stage"].value_counts().reindex(stage_order, fill_value=0)
        st.bar_chart(counts)

        st.markdown("##### Average canopy area over time (all plants)")
        avg_over_time = hist_df.groupby("frame")["canopy_area_px"].mean()
        st.line_chart(avg_over_time)

    # ---------------- Per-plant growth charts ----------------
    with tab_charts:
        plant_ids = sorted(hist_df["plant_id"].unique())
        selected = st.multiselect(
            "Select plant(s) to plot", plant_ids,
            default=plant_ids[: min(5, len(plant_ids))]
        )
        if selected:
            pivot = (
                hist_df[hist_df["plant_id"].isin(selected)]
                .pivot_table(index="frame", columns="plant_id", values="canopy_area_px", aggfunc="mean")
                .sort_index()
            )
            pivot.columns = [f"Plant {c}" for c in pivot.columns]
            st.markdown("##### Canopy area growth curve (px²)")
            st.line_chart(pivot)

            st.markdown("##### Confidence score over time")
            pivot_conf = (
                hist_df[hist_df["plant_id"].isin(selected)]
                .pivot_table(index="frame", columns="plant_id", values="confidence", aggfunc="mean")
                .sort_index()
            )
            pivot_conf.columns = [f"Plant {c}" for c in pivot_conf.columns]
            st.line_chart(pivot_conf)
        else:
            st.info("Select at least one plant ID above to see its growth curve.")

    # ---------------- Alerts ----------------
    with tab_alerts:
        st.caption("Simple rule-based alerts — good starting points to extend later.")
        snapshot = history.latest_snapshot()
        alerts = []

        low_conf = snapshot[snapshot["confidence"] < 0.35]
        for _, r in low_conf.iterrows():
            alerts.append(("⚠️", f"Plant {r['plant_id']}: low detection confidence "
                                  f"({r['confidence']*100:.0f}%) — verify manually."))

        # Stalled growth: compare last two measurements per plant
        for pid, g in hist_df.groupby("plant_id"):
            g = g.sort_values("frame")
            if len(g) >= 2:
                delta = g["canopy_area_px"].iloc[-1] - g["canopy_area_px"].iloc[-2]
                if delta <= 0:
                    alerts.append(("🟡", f"Plant {pid}: canopy area did not increase "
                                          "in the last measurement — possible stalled growth."))

        mature = snapshot[snapshot["growth_stage"] == "Mature"]
        for _, r in mature.iterrows():
            alerts.append(("✅", f"Plant {r['plant_id']}: reached **Mature** stage."))

        if alerts:
            for icon, msg in alerts[:25]:
                st.write(f"{icon} {msg}")
        else:
            st.success("No alerts — all tracked plants look healthy.")

    # ---------------- Raw data & export ----------------
    with tab_data:
        st.markdown("##### Full measurement history")
        st.dataframe(hist_df.sort_values(["plant_id", "frame"]), use_container_width=True, hide_index=True)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full history as CSV",
            data=csv_bytes,
            file_name=f"crop_growth_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        snap_csv = history.latest_snapshot().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download latest snapshot only (CSV)",
            data=snap_csv,
            file_name=f"crop_growth_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# =========================================================================
# 14. FOOTER / NOTES
# =========================================================================
st.divider()
with st.expander("ℹ️ About this app / how to extend it"):
    st.markdown(
        """
- **Detector**: defaults to a real, dependency-free OpenCV green-mask detector.
  A full YOLO code path (`YOLODetector`) is included and activates automatically
  if you install `ultralytics` and point it at a local `.pt` weights file — no
  other code changes required.
- **Tracking**: `PlantTracker` uses IoU + centroid-distance matching to keep a
  persistent integer ID per plant across frames.
- **Growth stage**: rule-based thresholds on canopy pixel area
  (Seedling → Vegetative → Flowering → Mature) — tune them in the sidebar, or
  replace `GrowthEstimator.stage_for_area` with a trained model later.
- **Storage**: `HistoryStore` is an in-memory table shaped exactly like a SQL
  table would be, so swapping in SQLite/Postgres later only touches that one
  class (see its docstring).
- **Camera**: image/video upload today; the same per-frame loop can point at
  `cv2.VideoCapture(0)` for a live camera feed later with no redesign.
- **No API keys, no paid services, fully offline.**
        """
    )