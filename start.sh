#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "════════════════════════════════════════"
echo "  BloodstockAI v3.0"
echo "  DeepLabCut SuperAnimal-Quadruped"
echo "════════════════════════════════════════"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3.9+ required."
    exit 1
fi

echo "[1/2] Installing dependencies..."
cd "$DIR/backend"
pip install -r requirements.txt --quiet 2>/dev/null || \
pip install -r requirements.txt --break-system-packages --quiet

echo "[2/2] Starting server..."
echo ""
echo "  Frontend:  http://localhost:8000"
echo "  API:       http://localhost:8000/api"
echo "  History:   http://localhost:8000/history"
echo "  Database:  $DIR/backend/bloodstockai.db"
echo ""
echo "  Ctrl+C to stop"
echo ""

export BAI_FRONTEND_DIR="$DIR/frontend"
cd "$DIR/backend"
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
