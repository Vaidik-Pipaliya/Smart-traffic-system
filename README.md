# Smart Multi-Agent Traffic Management System

An intelligent, multi-agent traffic control and simulation system designed to optimize urban intersection traffic flow using agent-based modeling and real-time visualization.

## Directory Structure

- `/backend`: FastAPI Python service providing REST endpoints, simulation control APIs, and backend logic.
- `/frontend`: React + TypeScript + Vite web dashboard for real-time visualization and system metrics.
- `/simulation`: Core agent-based traffic simulation engine, modeling vehicles, intersections, and traffic light policies.
- `/docs`: Architecture documentation, system schemas, and pending requirements tracker.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Launch FastAPI server
uvicorn backend.main:app --reload --port 8000
```
Health Check: `http://localhost:8000/api/health`

### Testing Backend
```bash
pytest backend/test_main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Access dashboard at `http://localhost:5173`.
