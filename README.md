# BloodstockAI

Equine biomechanical analysis powered by DeepLabCut SuperAnimal-Quadruped.
BAI Score™ composite scoring: PIS 35% · PAS 30% · CBS 20% · MVS 15%.

## Quick Start

### Local (Python)

```bash
# Mac / Linux
./start.sh

# Windows
start.bat
```

Opens http://localhost:8000 — backend serves the frontend and DLC API.

**Requirements:** Python 3.9–3.11, pip, GPU recommended (NVIDIA CUDA).

### Docker (recommended for production)

```bash
docker compose up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) for GPU passthrough. Falls back to CPU without it (~10× slower).

First run downloads DLC model weights (~500MB).

## Architecture

```
bloodstockai/
├── frontend/
│   └── index.html              # Self-contained React app (CDN-loaded)
├── backend/
│   ├── app.py                  # FastAPI + DeepLabCut + SQLite
│   └── requirements.txt        # Python dependencies
├── Dockerfile                  # GPU-ready container
├── docker-compose.yml          # One-command deployment
├── .env.example                # Configuration reference
├── start.sh / start.bat        # Local dev launchers
├── .dockerignore
└── .gitignore
```

## Pipeline

```
Video Upload → POST /analyze
    ↓
DeepLabCut v3.0 — SuperAnimal-Quadruped (HRNet-W32)
    ↓  39 keypoints detected per frame
Keypoint Mapping — DLC 39-pt → BAI 22-pt equine skeleton
    ↓  Direct: hooves, joints, withers, ears
    ↓  Derived: poll, shoulder, loin, croup, hock, girth
Conformation Scoring — 9 angular/ratio metrics
Movement Analysis — 6 gait quality metrics
BAI Score™ — Weighted composite
Risk Detection — Structural flags
    ↓
Results persisted to SQLite → GET /results/{id}
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves frontend |
| GET | `/api` | API info + DLC status |
| GET | `/health` | Health check (Docker) |
| POST | `/analyze` | Upload video for analysis |
| GET | `/status/{id}` | Poll job progress |
| GET | `/results/{id}` | Get completed results |
| GET | `/history` | List past analyses (paginated) |
| WS | `/ws/{id}` | Real-time progress stream |
| DELETE | `/jobs/{id}` | Clean up job + files |

## Database

SQLite (file: `bloodstockai.db`). Schema:

- `analyses` table — one row per video upload
- Stores: job metadata, video info, BAI scores, conformation/movement results, risk flags, full results JSON
- Indexed by `created_at DESC` and `status`
- Persistent via Docker named volume (`dbdata`)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BAI_DLC_MODEL` | `superanimal_quadruped` | DLC model name |
| `BAI_DLC_BACKBONE` | `hrnet_w32` | Network backbone |
| `BAI_DLC_DETECTOR` | `fasterrcnn_resnet50_fpn_v2` | Animal detector |
| `BAI_DLC_PCUTOFF` | `0.15` | Confidence threshold |
| `BAI_DLC_VIDEO_ADAPT` | `false` | Self-supervised video adaptation |
| `BAI_FRONTEND_DIR` | `../frontend` | Path to frontend files |
| `BAI_UPLOAD_DIR` | `./uploads` | Video upload storage |
| `BAI_RESULTS_DIR` | `./results` | DLC output + results JSON |
| `BAI_DB_PATH` | `./bloodstockai.db` | SQLite database path |
| `BAI_CORS_ORIGINS` | `*` | Allowed CORS origins |

## References

- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) — Markerless pose estimation
- [SuperAnimal-Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped) — Foundation model
- Ye, S. et al. (2024). SuperAnimal pretrained pose estimation models. *Nature Communications*.
