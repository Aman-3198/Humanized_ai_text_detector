"""FastAPI backend for the Humanized AI Text Detector.
Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000   (the frontend is served from the same app)"""
import re, threading
from contextlib import asynccontextmanager
from pathlib import Path
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import detector_v5                      # must be imported so joblib can unpickle DetectorV5
from detector_v5 import CLASSES         # ['plain_ai', 'humanized_ai', 'human']

BASE = Path(__file__).parent
MAX_CHARS, MAX_SENTENCES = 20000, 150
KEY = {'plain_ai': 'ai', 'humanized_ai': 'hum', 'human': 'human'}   # frontend names
STATE = {}
LOCK = threading.Lock()                 # spaCy / sklearn calls are serialised

@asynccontextmanager
async def lifespan(app):
    STATE['model'] = joblib.load(BASE / 'detector_v5.joblib')
    STATE['model'].predict_proba(['Warm up the grammar pipeline before the first request arrives.'])
    yield

app = FastAPI(title='Humanized AI Text Detector', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class Req(BaseModel):
    text: str = Field(..., min_length=1)

def pct(p):
    """3 probabilities -> 3 ints that sum to exactly 100 (largest remainder)."""
    raw = np.asarray(p) * 100; out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[:100 - out.sum()]: out[i] += 1
    return out.tolist()

def split_sentences(text):
    return [s.strip() for s in re.findall(r'[^.!?]+[.!?]*', text) if len(s.split()) > 0]

def scores(p, label=None):
    a, h, m = pct([p[0], p[2], p[1]])                    # ai, human, hum  (p is in CLASSES order)
    d = {'ai': a, 'human': h, 'hum': m}
    d['top'] = max(d, key=d.get); return d

@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'detector_v5', 'classes': CLASSES, 'loaded': 'model' in STATE}

@app.post('/api/analyze')
def analyze(req: Req):
    text = req.text.strip()
    if len(text.split()) < 8:
        raise HTTPException(422, 'Add at least eight words.')
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f'Text is too long (max {MAX_CHARS} characters).')
    sents = split_sentences(text)[:MAX_SENTENCES]
    model = STATE['model']
    with LOCK:
        P = model.predict_proba([text] + sents)           # row 0 = whole text, rest = sentences
        labels = model.predict([text])
    overall = scores(P[0])
    label = 'uncertain' if labels[0] == 'uncertain' else KEY[labels[0]]
    out = []
    for s, p in zip(sents, P[1:]):
        sc = scores(p); out.append({'text': s, 'words': len(s.split()), **sc, 'confidence': round(float(p.max()), 3)})
    return {**overall, 'label': label, 'uncertain': label == 'uncertain',
            'confidence': round(float(P[0].max()), 3), 'model': 'detector_v5', 'sentences': out}

@app.get('/')
def index():
    return FileResponse(BASE / 'index.html')
