from fastapi import FastAPI
from pydantic import BaseModel
import os, torch
from transformers import AutoTokenizer
from .train_multitask import CivicBalance
from .config import POSITIVE_LABELS, TOXIC_LABELS, ARTIFACT_DIR

MODEL_NAME = 'bert-base-uncased'
ARTIFACT = os.path.join(ARTIFACT_DIR, 'civicbalance_bert.pt')

app = FastAPI(title="CivicBalance API")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = CivicBalance(MODEL_NAME)
model.load_state_dict(torch.load(ARTIFACT, map_location='cpu'))
model.eval()

class Req(BaseModel):
    text: str
    threshold: float = 0.5
    w_pos: float = 1.0
    w_tox: float = 1.2

@app.post('/predict')
def predict(r: Req):
    enc = tok(r.text, return_tensors='pt', truncation=True, padding=True, max_length=192)
    with torch.no_grad():
        lt, lp = model(**enc)
        pt = lt.sigmoid().numpy()[0]
        pp = lp.sigmoid().numpy()[0]
    tox = dict(zip(TOXIC_LABELS, [float(x) for x in pt]))
    pos = dict(zip(POSITIVE_LABELS, [float(x) for x in pp]))
    ngs = r.w_pos * (sum(pos.values())/len(pos)) - r.w_tox * (sum(tox.values())/len(tox))
    tox_bin = {k:int(v>=r.threshold) for k,v in tox.items()}
    pos_bin = {k:int(v>=r.threshold) for k,v in pos.items()}
    return {"toxicity": tox, "toxicity_bin": tox_bin, "positivity": pos, "positivity_bin": pos_bin, "net_goodness": float(ngs)}