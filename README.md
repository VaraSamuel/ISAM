# iSAM (FastAPI + React)

## Local dev

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR=./data
export CORS_ORIGIN=http://localhost:5173
uvicorn app.main:app --reload --port 8000
