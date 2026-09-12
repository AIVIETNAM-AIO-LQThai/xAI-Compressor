from fastapi import FastAPI

app = FastAPI(
    title="AeroXAI API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "project": "AeroXAI",
        "mode": "advisory",
    }