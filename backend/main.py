from fastapi import FastAPI

app = FastAPI(title="Smart Multi-Agent Traffic Management System")


@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "message": "Smart Traffic System API is operational",
        "system": "Multi-Agent Traffic Management",
    }
