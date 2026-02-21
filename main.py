import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from util.classifier import MLPClassifier
from util.embed import embed

MODEL_PATH = os.environ.get("MODEL_PATH", "data/safety_classifier.npz")

clf: MLPClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf
    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"warning: model not found at {p}, /classify will be unavailable")
    else:
        clf = MLPClassifier.load(str(p))
        print(f"loaded model from {p} ({clf.num_classes} classes)")
    yield


app = FastAPI(title="antislopfactory", lifespan=lifespan)


# ── schemas ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str | list[str]


class ClassifyResult(BaseModel):
    text: str
    label: str
    probabilities: dict[str, float]


class ClassifyResponse(BaseModel):
    results: list[ClassifyResult]


# ── routes ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": clf is not None}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if clf is None:
        raise HTTPException(503, "model not loaded")

    texts = [req.text] if isinstance(req.text, str) else req.text
    if not texts:
        raise HTTPException(422, "text must not be empty")

    vecs = await embed(texts)
    labels = clf.predict(vecs)
    probs = clf.predict_proba(vecs)

    return ClassifyResponse(
        results=[
            ClassifyResult(text=t, label=l, probabilities=p)
            for t, l, p in zip(texts, labels, probs)
        ]
    )
