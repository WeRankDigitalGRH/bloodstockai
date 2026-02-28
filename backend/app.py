"""
BloodstockAI Backend v3.0
FastAPI + DeepLabCut SuperAnimal-Quadruped + SQLite
"""
import os, uuid, json, math, asyncio, logging, shutil
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import aiosqlite
import pandas as pd

try:
    import deeplabcut
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
    logging.warning("DeepLabCut not installed. pip install 'deeplabcut[pytorch]'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bloodstockai")

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
UPLOAD_DIR = Path(os.environ.get("BAI_UPLOAD_DIR", "./uploads"))
RESULTS_DIR = Path(os.environ.get("BAI_RESULTS_DIR", "./results"))
FRONTEND_DIR = Path(os.environ.get("BAI_FRONTEND_DIR", "../frontend"))
DB_PATH = os.environ.get("BAI_DB_PATH", "./bloodstockai.db")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VALID_GAITS = {"walk", "trot", "canter", "gallop"}

DLC_MODEL = os.environ.get("BAI_DLC_MODEL", "superanimal_quadruped")
DLC_BACKBONE = os.environ.get("BAI_DLC_BACKBONE", "hrnet_w32")
DLC_DETECTOR = os.environ.get("BAI_DLC_DETECTOR", "fasterrcnn_resnet50_fpn_v2")
DLC_PCUTOFF = float(os.environ.get("BAI_DLC_PCUTOFF", "0.15"))
DLC_VIDEO_ADAPT = os.environ.get("BAI_DLC_VIDEO_ADAPT", "false").lower() == "true"

# In-memory progress tracking (live jobs only — completed jobs in DB)
active_jobs: dict = {}

# ═══════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    filename TEXT,
    file_size INTEGER,
    gait TEXT,
    video_width INTEGER,
    video_height INTEGER,
    video_fps REAL,
    video_duration REAL,
    video_total_frames INTEGER,
    sampled_frames INTEGER,
    dlc_model TEXT,
    bai_score REAL,
    bai_band TEXT,
    pis INTEGER,
    pas INTEGER,
    cbs INTEGER,
    mvs INTEGER,
    conf_overall REAL,
    movement_overall REAL,
    risk_flags TEXT,
    results_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
"""

async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db

async def init_db():
    db = await get_db()
    await db.executescript(SCHEMA)
    await db.commit()
    await db.close()
    logger.info(f"Database initialized at {DB_PATH}")

async def db_insert_job(jid, filename, file_size, gait):
    now = datetime.now(timezone.utc).isoformat()
    db = await get_db()
    await db.execute(
        "INSERT INTO analyses (id, created_at, updated_at, status, progress, filename, file_size, gait, dlc_model) VALUES (?,?,?,?,?,?,?,?,?)",
        (jid, now, now, "queued", 0, filename, file_size, gait, f"{DLC_MODEL}/{DLC_BACKBONE}"))
    await db.commit()
    await db.close()

async def db_update_status(jid, status, progress, error=None):
    now = datetime.now(timezone.utc).isoformat()
    db = await get_db()
    await db.execute(
        "UPDATE analyses SET status=?, progress=?, error=?, updated_at=? WHERE id=?",
        (status, progress, error, now, jid))
    await db.commit()
    await db.close()

async def db_update_video_info(jid, w, h, fps, dur, total_frames, sampled):
    db = await get_db()
    await db.execute(
        "UPDATE analyses SET video_width=?, video_height=?, video_fps=?, video_duration=?, video_total_frames=?, sampled_frames=?, updated_at=? WHERE id=?",
        (w, h, fps, dur, total_frames, sampled, datetime.now(timezone.utc).isoformat(), jid))
    await db.commit()
    await db.close()

async def db_save_results(jid, bai, conf_overall, movement_overall, risks, results_json):
    now = datetime.now(timezone.utc).isoformat()
    db = await get_db()
    await db.execute(
        """UPDATE analyses SET status='complete', progress=100,
           bai_score=?, bai_band=?, pis=?, pas=?, cbs=?, mvs=?,
           conf_overall=?, movement_overall=?, risk_flags=?,
           results_json=?, updated_at=? WHERE id=?""",
        (bai["bai"], bai["band"], bai["PIS"], bai["PAS"], bai["CBS"], bai["MVS"],
         conf_overall, movement_overall, json.dumps(risks),
         results_json, now, jid))
    await db.commit()
    await db.close()

# ═══════════════════════════════════════════════════════════════
# DLC Keypoint Mapping (39 → 22)
# ═══════════════════════════════════════════════════════════════
DLC_TO_BAI = {
    "nose": "nose", "upper_jaw": "nose",
    "right_eye": "eye", "left_eye": "eye",
    "right_eartip": "nearear", "left_eartip": "farear",
    "right_earbase": "nearear", "left_earbase": "farear",
    "throat": "throat", "chin": "throat",
    "withers": "withers", "spine_mid": "back", "tailbase": "tailbase",
    "right_front_elbow": "elbow", "right_front_knee": "foreknee",
    "right_front_fetlock": "forefetlock", "right_front_hoof": "forehoof",
    "right_back_knee": "stifle",
    "right_back_fetlock": "hindfetlock", "right_back_hoof": "hindhoof",
}
DERIVED = {
    "poll": ("right_earbase", "left_earbase"),
    "shoulder": ("withers", "right_front_elbow"),
    "loin": ("spine_mid", "tailbase"),
    "croup": ("spine_mid", "tailbase", 0.67),
    "hip": ("tailbase", "right_back_elbow"),
    "hock": ("right_back_knee", "right_back_fetlock"),
    "girth": ("right_front_elbow", "right_back_elbow"),
}
BAI_KPS = [
    "nose", "eye", "nearear", "farear", "poll", "throat",
    "withers", "back", "loin", "croup", "tailbase",
    "shoulder", "elbow", "foreknee", "forefetlock", "forehoof",
    "hip", "stifle", "hock", "hindfetlock", "hindhoof", "girth"
]
BAI_GRP = {
    "nose": "head", "eye": "head", "nearear": "head", "farear": "head",
    "poll": "head", "throat": "head", "withers": "top", "back": "top",
    "loin": "top", "croup": "top", "tailbase": "top", "shoulder": "fore",
    "elbow": "fore", "foreknee": "fore", "forefetlock": "fore", "forehoof": "fore",
    "hip": "hind", "stifle": "hind", "hock": "hind", "hindfetlock": "hind",
    "hindhoof": "hind", "girth": "barrel"
}
BAI_LABELS = {k: k.replace("_", " ").title() for k in BAI_KPS}
BAI_LABELS.update({"nearear": "Near Ear", "farear": "Far Ear", "forefetlock": "Fore Fetlock",
                    "forehoof": "Fore Hoof", "foreknee": "Fore Knee", "hindfetlock": "Hind Fetlock",
                    "hindhoof": "Hind Hoof", "tailbase": "Tail Base"})


def map_dlc_to_bai(dlc_kps: dict) -> list:
    bai = {}
    for dn, bn in DLC_TO_BAI.items():
        if dn in dlc_kps and bn not in bai:
            x, y, c = dlc_kps[dn]
            if c > DLC_PCUTOFF:
                bai[bn] = (x, y, c)
    for bn, src in DERIVED.items():
        if bn in bai:
            continue
        if len(src) == 2:
            a, b = src
            if a in dlc_kps and b in dlc_kps:
                ax, ay, ac = dlc_kps[a]
                bx, by, bc = dlc_kps[b]
                if ac > DLC_PCUTOFF and bc > DLC_PCUTOFF:
                    bai[bn] = ((ax + bx) / 2, (ay + by) / 2, min(ac, bc))
        elif len(src) == 3:
            a, b, r = src
            if a in dlc_kps and b in dlc_kps:
                ax, ay, ac = dlc_kps[a]
                bx, by, bc = dlc_kps[b]
                if ac > DLC_PCUTOFF and bc > DLC_PCUTOFF:
                    bai[bn] = (ax + (bx - ax) * r, ay + (by - ay) * r, min(ac, bc))
    result = []
    for kp in BAI_KPS:
        if kp in bai:
            x, y, c = bai[kp]
            result.append({"id": kp, "label": BAI_LABELS.get(kp, kp), "x": float(x), "y": float(y), "conf": float(c), "g": BAI_GRP[kp]})
        else:
            result.append({"id": kp, "label": BAI_LABELS.get(kp, kp), "x": 0, "y": 0, "conf": 0, "g": BAI_GRP[kp]})
    return result


# ═══════════════════════════════════════════════════════════════
# Biometric Calculations
# ═══════════════════════════════════════════════════════════════
def _ang(a, b, c):
    ax, ay = a[0] - b[0], a[1] - b[1]
    bx, by = c[0] - b[0], c[1] - b[1]
    dot = ax * bx + ay * by
    m1, m2 = math.hypot(ax, ay), math.hypot(bx, by)
    if m1 == 0 or m2 == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (m1 * m2)))))

def _dst(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _avg(a):
    return sum(a) / len(a) if a else 0.0

def _std(a):
    if not a:
        return 0.0
    m = _avg(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / len(a))

def _rom(a):
    return (max(a) - min(a)) if a else 0.0


def compute_conformation(kps: list) -> dict:
    lk = {k["id"]: (k["x"], k["y"]) for k in kps if k["conf"] > 0.1}
    m = {}
    def has(*ids):
        return all(i in lk for i in ids)
    if has("nose", "poll", "withers"):
        m["headNeckAngle"] = _ang(lk["nose"], lk["poll"], lk["withers"])
    if has("withers", "shoulder", "elbow"):
        m["shoulderAngle"] = _ang(lk["withers"], lk["shoulder"], lk["elbow"])
    if has("croup", "hip", "stifle"):
        m["hipAngle"] = _ang(lk["croup"], lk["hip"], lk["stifle"])
    if has("stifle", "hock", "hindfetlock"):
        m["hockAngle"] = _ang(lk["stifle"], lk["hock"], lk["hindfetlock"])
    if has("elbow", "foreknee", "forefetlock"):
        m["foreKneeAngle"] = _ang(lk["elbow"], lk["foreknee"], lk["forefetlock"])
    if has("withers", "back", "loin", "croup"):
        m["toplineDeviation"] = ((lk["back"][1] + lk["loin"][1]) / 2 - (lk["withers"][1] + lk["croup"][1]) / 2) * 100
    if has("shoulder", "hip", "withers", "forehoof"):
        denom = _dst(lk["withers"], lk["forehoof"])
        m["bodyLengthRatio"] = _dst(lk["shoulder"], lk["hip"]) / denom if denom > 0 else 0
    if has("poll", "withers", "forehoof"):
        denom = _dst(lk["withers"], lk["forehoof"])
        m["neckLengthRatio"] = _dst(lk["poll"], lk["withers"]) / denom if denom > 0 else 0
    if has("croup", "tailbase"):
        m["croupAngle"] = abs(math.degrees(math.atan2(
            lk["tailbase"][1] - lk["croup"][1], lk["tailbase"][0] - lk["croup"][0])))
    return m


SCORE_DEFS = [
    ("shoulderAngle", 95, 30, 1.5, "Shoulder Angle", "\u224895\u00B0"),
    ("headNeckAngle", 110, 40, 1.0, "Head-Neck", "\u2248110\u00B0"),
    ("hipAngle", 95, 30, 1.5, "Hip Angle", "\u224895\u00B0"),
    ("hockAngle", 155, 30, 1.2, "Hock Angle", "\u2248155\u00B0"),
    ("foreKneeAngle", 170, 25, 1.2, "Fore Knee", "\u2248170\u00B0"),
    ("toplineDeviation", 0, 8, 1.3, "Topline Dev.", "\u22480"),
    ("bodyLengthRatio", 1.0, 0.35, 1.0, "Body Ratio", "\u22481.0"),
    ("neckLengthRatio", 0.38, 0.2, 0.8, "Neck Ratio", "\u22480.38"),
    ("croupAngle", 25, 20, 1.0, "Croup Angle", "\u224825\u00B0"),
]


def score_conformation(metrics: dict) -> tuple:
    scores = {}
    total_score = total_max = 0
    for key, ideal, tol, weight, label, ideal_str in SCORE_DEFS:
        if key in metrics:
            raw = max(0, 1 - abs(metrics[key] - ideal) / tol)
            sc = raw * 10 * weight
            mx = 10 * weight
            scores[key] = {
                "score": round(sc, 2), "maxScore": round(mx, 2),
                "raw": round(raw, 3), "value": round(metrics[key], 2),
                "ideal": ideal_str, "label": label,
                "pct": round((sc / mx) * 100, 1) if mx > 0 else 0,
            }
            total_score += sc
            total_max += mx
    overall = (total_score / total_max * 100) if total_max > 0 else 0
    return scores, round(overall, 1)


# Gait-specific scale factors for hock/shoulder ROM expectations
GAIT_SF = {"walk": 1.0, "trot": 1.4, "canter": 1.8, "gallop": 2.4}

def compute_movement(frame_kps: list, gait: str = "trot") -> dict:
    """Full 6-metric movement scoring matching frontend expectations."""
    if len(frame_kps) < 2:
        return {"overall": 0, "metrics": {}, "foreDeltas": [], "hindDeltas": [], "strides": 0}

    sf = GAIT_SF.get(gait, 1.4)
    fore_d, hind_d = [], []
    for i in range(1, len(frame_kps)):
        prev = {k["id"]: k for k in frame_kps[i - 1]}
        curr = {k["id"]: k for k in frame_kps[i]}
        for hoof_id, target in [("forehoof", fore_d), ("hindhoof", hind_d)]:
            if hoof_id in prev and hoof_id in curr:
                p, c = prev[hoof_id], curr[hoof_id]
                if p["conf"] > 0.1 and c["conf"] > 0.1:
                    target.append(math.hypot(c["x"] - p["x"], c["y"] - p["y"]))

    af, ah = _avg(fore_d), _avg(hind_d)
    symmetry = (min(af, ah) / max(af, ah) * 100) if af > 0 and ah > 0 else 0
    all_d = fore_d + hind_d
    rhythm = max(0, (1 - _std(all_d) / (_avg(all_d) or 1))) * 100

    per_frame_metrics = [compute_conformation(kps) for kps in frame_kps]

    topline_vals = [m.get("toplineDeviation", 0) for m in per_frame_metrics]
    topline_stability = max(0, (1 - _std(topline_vals) / 3)) * 100

    head_vals = [m.get("headNeckAngle", 0) for m in per_frame_metrics if m.get("headNeckAngle")]
    head_carriage = max(0, (1 - _std(head_vals) / 15)) * 100 if head_vals else 50

    hock_vals = [m.get("hockAngle", 0) for m in per_frame_metrics if m.get("hockAngle")]
    hock_engagement = min(100, (_rom(hock_vals) / (20 * sf)) * 100) if hock_vals else 50

    shoulder_vals = [m.get("shoulderAngle", 0) for m in per_frame_metrics if m.get("shoulderAngle")]
    shoulder_freedom = min(100, (_rom(shoulder_vals) / (15 * sf)) * 100) if shoulder_vals else 50

    overall = (symmetry * 20 + rhythm * 20 + topline_stability * 15 +
               head_carriage * 15 + hock_engagement * 15 + shoulder_freedom * 15) / 100

    return {
        "overall": round(min(100, overall), 1),
        "metrics": {
            "symmetry": round(symmetry, 1),
            "rhythm": round(rhythm, 1),
            "toplineStability": round(topline_stability, 1),
            "headCarriage": round(head_carriage, 1),
            "hockEngagement": round(hock_engagement, 1),
            "shoulderFreedom": round(shoulder_freedom, 1),
        },
        "foreDeltas": [round(d, 4) for d in fore_d[:300]],
        "hindDeltas": [round(d, 4) for d in hind_d[:300]],
        "strides": max(1, len(fore_d) // 8),
    }


def compute_bai(conf_score: float, move_score: float) -> dict:
    CBS = max(0, min(100, conf_score))
    PAS = max(0, min(100, move_score))
    nicking = 8 if (CBS > 70 and PAS > 60) else (4 if CBS > 60 else 0)
    PIS = max(0, min(100, CBS * 0.4 + PAS * 0.3 + 30 + nicking))
    MVS = max(0, min(100, PIS * 0.3 + CBS * 0.3 + PAS * 0.2 + 20))
    bai = PIS * 0.35 + PAS * 0.30 + CBS * 0.20 + MVS * 0.15

    if bai >= 90: band = "EXCEPTIONAL"
    elif bai >= 80: band = "OUTSTANDING"
    elif bai >= 70: band = "VERY STRONG"
    elif bai >= 60: band = "ABOVE AVG"
    elif bai >= 50: band = "AVERAGE"
    else: band = "BELOW AVG"

    return {"bai": round(bai, 1), "band": band,
            "PIS": round(PIS), "PAS": round(PAS), "CBS": round(CBS), "MVS": round(MVS)}


def detect_risks(m: dict) -> list:
    flags = []
    sa = m.get("shoulderAngle")
    if sa is not None and (sa < 40 or sa > 65):
        flags.append({"area": "Shoulder", "level": "high", "note": f"Angle {sa:.1f}\u00B0 outside 50-55\u00B0 optimal"})
    ha = m.get("hockAngle")
    if ha is not None and (ha < 140 or ha > 170):
        flags.append({"area": "Hock", "level": "high", "note": f"Geometry ({ha:.1f}\u00B0) distal limb stress"})
    fk = m.get("foreKneeAngle")
    if fk is not None and fk < 155:
        flags.append({"area": "Fore Knee", "level": "mod", "note": f"Perpendicularity ({fk:.1f}\u00B0)"})
    td = m.get("toplineDeviation")
    if td is not None and abs(td) > 5:
        flags.append({"area": "Topline", "level": "mod", "note": f"Deviation {td:.1f}"})
    return flags


# ═══════════════════════════════════════════════════════════════
# DLC Processing Pipeline (runs in thread pool)
# ═══════════════════════════════════════════════════════════════

def _run_dlc_blocking(job_id: str, video_path: Path, gait: str, max_frames: int):
    """
    Blocking function that runs DLC inference. Called via run_in_executor
    so it doesn't freeze the async event loop.
    """
    # Phase 1: Read video metadata
    active_jobs[job_id] = {"status": "reading_video", "progress": 5}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    cap.release()

    step = max(1, total_frames // max_frames)
    sampled = min(total_frames, max_frames)

    active_jobs[job_id].update({"progress": 10, "video_info": {
        "totalFrames": total_frames, "fps": round(fps, 2),
        "width": w, "height": h, "duration": round(duration, 2),
        "sampledFrames": sampled}})

    # Phase 2: Run DLC
    if not DLC_AVAILABLE:
        raise RuntimeError("DeepLabCut not installed. Run: pip install 'deeplabcut[pytorch]'")

    active_jobs[job_id].update({"status": "running_dlc", "progress": 15})
    logger.info(f"[{job_id}] Running DLC {DLC_MODEL}/{DLC_BACKBONE} on {video_path.name} ({total_frames} frames)")

    dlc_out = RESULTS_DIR / job_id / "dlc_output"
    dlc_out.mkdir(parents=True, exist_ok=True)

    # Try both parameter names for compatibility across DLC versions
    try:
        deeplabcut.video_inference_superanimal(
            [str(video_path)], DLC_MODEL,
            model_name=DLC_BACKBONE, detector_name=DLC_DETECTOR,
            videotype=video_path.suffix, video_adapt=DLC_VIDEO_ADAPT,
            scale_list=range(200, 600, 50), destfolder=str(dlc_out))
    except TypeError:
        deeplabcut.video_inference_superanimal(
            [str(video_path)], DLC_MODEL,
            model_name=DLC_BACKBONE, detector_name=DLC_DETECTOR,
            videotype=video_path.suffix, video_adapt=DLC_VIDEO_ADAPT,
            scale_list=range(200, 600, 50), dest_folder=str(dlc_out))

    active_jobs[job_id].update({"status": "parsing_keypoints", "progress": 60})

    # Phase 3: Parse DLC output (.h5)
    h5_files = list(dlc_out.glob("*.h5"))
    if not h5_files:
        h5_files = list(video_path.parent.glob(f"*{video_path.stem}*.h5"))
    if not h5_files:
        raise FileNotFoundError("DLC produced no output .h5 file. Check model installation.")

    df = pd.read_hdf(h5_files[0])
    logger.info(f"[{job_id}] DLC output: {df.shape[0]} rows, {len(df.columns)} cols")
    scorer = df.columns.get_level_values(0)[0]

    all_frames = []
    frame_indices = list(range(0, min(total_frames, len(df)), step))

    for ii, fi in enumerate(frame_indices):
        if fi >= len(df):
            break
        row = df.iloc[fi]
        dlc_kps = {}
        for bp in df.columns.get_level_values(1).unique():
            try:
                x = float(row[(scorer, bp, "x")])
                y = float(row[(scorer, bp, "y")])
                lk = float(row[(scorer, bp, "likelihood")])
                dlc_kps[bp] = (x / w, y / h, lk)
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"[{job_id}] Skipping keypoint {bp} frame {fi}: {e}")
                continue

        all_frames.append({"fi": fi, "t": round(fi / fps, 3), "kps": map_dlc_to_bai(dlc_kps)})
        active_jobs[job_id]["progress"] = 60 + int((ii / max(1, len(frame_indices))) * 25)

    if not all_frames:
        raise ValueError("No valid keypoints extracted from DLC output")

    active_jobs[job_id].update({"status": "computing_scores", "progress": 88})

    # Phase 4: Compute scores
    mid_kps = all_frames[len(all_frames) // 2]["kps"]
    conf_metrics = compute_conformation(mid_kps)
    conf_scores, conf_overall = score_conformation(conf_metrics)
    movement = compute_movement([f["kps"] for f in all_frames], gait)
    bai = compute_bai(conf_overall, movement["overall"])
    risks = detect_risks(conf_metrics)

    result = {
        "jobId": job_id,
        "videoInfo": active_jobs[job_id]["video_info"],
        "dlcModel": f"{DLC_MODEL}/{DLC_BACKBONE}",
        "frames": all_frames,
        "frameCount": len(all_frames),
        "fps": round(fps, 2),
        "gait": gait,
        "baiScore": bai,
        "conformation": {"metrics": conf_metrics, "scores": conf_scores, "overall": conf_overall},
        "movement": movement,
        "riskFlags": risks,
    }

    # Save JSON to disk
    result_path = RESULTS_DIR / job_id / "results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, default=str)

    return result, bai, conf_overall, movement["overall"], risks, w, h, fps, duration, total_frames, sampled


async def process_video(job_id: str, video_path: Path, gait: str, max_frames: int):
    """Async wrapper — runs blocking DLC in thread pool."""
    try:
        loop = asyncio.get_event_loop()
        result, bai, conf_ov, move_ov, risks, w, h, fps, dur, tf, sampled = await loop.run_in_executor(
            None, _run_dlc_blocking, job_id, video_path, gait, max_frames)

        # Persist to DB
        await db_update_video_info(job_id, w, h, fps, dur, tf, sampled)
        results_json = json.dumps(result, default=str)
        await db_save_results(job_id, bai, conf_ov, move_ov, risks, results_json)

        active_jobs[job_id] = {"status": "complete", "progress": 100}
        logger.info(f"[{job_id}] Complete. BAI: {bai['bai']} ({bai['band']})")
        # Clean up active_jobs after a delay (allow final status poll)
        await asyncio.sleep(30)
        active_jobs.pop(job_id, None)

    except Exception as e:
        logger.error(f"[{job_id}] Failed: {e}", exc_info=True)
        active_jobs[job_id] = {"status": "error", "progress": 0, "error": str(e)}
        await db_update_status(job_id, "error", 0, str(e))
        await asyncio.sleep(30)
        active_jobs.pop(job_id, None)


# ═══════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    await init_db()
    # Recover stale jobs stuck as queued/running from previous crash
    db = await get_db()
    await db.execute(
        "UPDATE analyses SET status='error', error='Server restarted during processing', updated_at=? "
        "WHERE status IN ('queued','reading_video','running_dlc','parsing_keypoints','computing_scores')",
        (datetime.now(timezone.utc).isoformat(),))
    await db.commit()
    stale = (await (await db.execute("SELECT changes()")).fetchone())[0]
    await db.close()
    if stale:
        logger.warning(f"Recovered {stale} stale job(s) from previous run")
    logger.info(f"BloodstockAI started | DLC: {'YES' if DLC_AVAILABLE else 'NO'} | Model: {DLC_MODEL}/{DLC_BACKBONE}")
    yield
    logger.info("BloodstockAI shutting down")

app = FastAPI(title="BloodstockAI", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("BAI_CORS_ORIGINS", "*").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── API routes (must be registered BEFORE static mount) ──

@app.get("/health")
async def health():
    db = await get_db()
    count = (await (await db.execute("SELECT COUNT(*) FROM analyses")).fetchone())[0]
    await db.close()
    return {"status": "ok", "dlc_available": DLC_AVAILABLE,
            "dlc_model": f"{DLC_MODEL}/{DLC_BACKBONE}",
            "active_jobs": len(active_jobs), "total_analyses": count}

@app.get("/api")
def api_info():
    return {"service": "BloodstockAI", "version": "3.0.0",
            "dlc_available": DLC_AVAILABLE, "dlc_model": f"{DLC_MODEL}/{DLC_BACKBONE}"}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    gait: str = Form("trot"),
    max_frames: int = Query(450, ge=30, le=1800),
):
    if gait not in VALID_GAITS:
        raise HTTPException(400, f"Invalid gait '{gait}'. Must be one of: {VALID_GAITS}")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    jid = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{ext}"

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        shutil.rmtree(job_dir)
        raise HTTPException(413, f"File too large ({len(content) // 1024 // 1024}MB). Max: {MAX_FILE_SIZE // 1024 // 1024}MB")

    with open(video_path, "wb") as f:
        f.write(content)

    await db_insert_job(jid, file.filename, len(content), gait)
    active_jobs[jid] = {"status": "queued", "progress": 0}
    asyncio.create_task(process_video(jid, video_path, gait, max_frames))

    return {"jobId": jid, "status": "queued"}


@app.get("/status/{jid}")
async def get_status(jid: str):
    # Check active jobs first
    if jid in active_jobs:
        aj = active_jobs[jid]
        return {"jobId": jid, "status": aj["status"], "progress": aj["progress"],
                "error": aj.get("error")}
    # Check DB
    db = await get_db()
    row = await db.execute("SELECT status, progress, error FROM analyses WHERE id=?", (jid,))
    row = await row.fetchone()
    await db.close()
    if not row:
        raise HTTPException(404, "Job not found")
    return {"jobId": jid, "status": row["status"], "progress": row["progress"], "error": row["error"]}


@app.get("/results/{jid}")
async def get_results(jid: str):
    # Check if still processing
    if jid in active_jobs and active_jobs[jid]["status"] not in ("complete", "error"):
        raise HTTPException(202, "Analysis in progress")

    db = await get_db()
    row = await db.execute("SELECT status, error, results_json FROM analyses WHERE id=?", (jid,))
    row = await row.fetchone()
    await db.close()

    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] == "error":
        raise HTTPException(500, row["error"] or "Analysis failed")
    if row["status"] != "complete":
        raise HTTPException(202, "Analysis in progress")
    if not row["results_json"]:
        # Try loading from disk
        rp = RESULTS_DIR / jid / "results.json"
        if rp.exists():
            return json.loads(rp.read_text())
        raise HTTPException(500, "Results data missing")

    return json.loads(row["results_json"])


@app.get("/history")
async def list_analyses(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    db = await get_db()
    rows = await db.execute(
        """SELECT id, created_at, status, filename, gait, bai_score, bai_band,
                  pis, pas, cbs, mvs, conf_overall, movement_overall, risk_flags,
                  video_duration, video_fps, sampled_frames
           FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?""", (limit, offset))
    rows = await rows.fetchall()
    count_row = await db.execute("SELECT COUNT(*) as cnt FROM analyses")
    count = (await count_row.fetchone())["cnt"]
    await db.close()

    items = []
    for r in rows:
        items.append({
            "id": r["id"], "createdAt": r["created_at"], "status": r["status"],
            "filename": r["filename"], "gait": r["gait"],
            "baiScore": r["bai_score"], "baiBand": r["bai_band"],
            "pis": r["pis"], "pas": r["pas"], "cbs": r["cbs"], "mvs": r["mvs"],
            "confOverall": r["conf_overall"], "movementOverall": r["movement_overall"],
            "riskFlags": json.loads(r["risk_flags"]) if r["risk_flags"] else [],
            "videoDuration": r["video_duration"], "videoFps": r["video_fps"],
            "sampledFrames": r["sampled_frames"],
        })

    return {"items": items, "total": count, "limit": limit, "offset": offset}


@app.websocket("/ws/{jid}")
async def ws_progress(websocket: WebSocket, jid: str):
    await websocket.accept()
    last_p = -1
    while True:
        aj = active_jobs.get(jid)
        if not aj:
            # Check DB
            db = await get_db()
            row = await db.execute("SELECT status, progress FROM analyses WHERE id=?", (jid,))
            row = await row.fetchone()
            await db.close()
            if row:
                await websocket.send_json({"status": row["status"], "progress": row["progress"]})
            else:
                await websocket.send_json({"error": "Not found"})
            break

        if aj["progress"] != last_p:
            last_p = aj["progress"]
            await websocket.send_json({"status": aj["status"], "progress": aj["progress"]})

        if aj["status"] in ("complete", "error"):
            break

        await asyncio.sleep(0.5)
    await websocket.close()


@app.delete("/jobs/{jid}")
async def delete_job(jid: str):
    db = await get_db()
    await db.execute("DELETE FROM analyses WHERE id=?", (jid,))
    await db.commit()
    await db.close()
    active_jobs.pop(jid, None)
    for d in [UPLOAD_DIR / jid, RESULTS_DIR / jid]:
        if d.exists():
            shutil.rmtree(d)
    return {"deleted": jid}


# ── Serve frontend (mounted AFTER API routes so /api etc take priority) ──
@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"service": "BloodstockAI", "note": "Frontend not found. Place index.html in frontend/ directory."}

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
