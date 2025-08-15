import os
import torch
from transformers import AutoTokenizer
from .train_multitask import CivicBalance
from .config import POSITIVE_LABELS, TOXIC_LABELS, ARTIFACT_DIR

ARTIFACT = os.path.join(ARTIFACT_DIR, 'civicbalance_bert.pt')
MODEL_NAME = 'bert-base-uncased'

SAMPLES = [
    "I appreciate your help, thanks a ton!",
    "You are a complete idiot and should be banned.",
    "Proud of the team for shipping on time!",
]

def soft_scores(texts):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CivicBalance(MODEL_NAME)
    model.load_state_dict(torch.load(ARTIFACT, map_location='cpu'))
    model.eval()
    res = []
    for t in texts:
        enc = tok(t, return_tensors='pt', truncation=True, padding=True, max_length=192)
        with torch.no_grad():
            lt, lp = model(**enc)
            pt = lt.sigmoid().numpy()[0]
            pp = lp.sigmoid().numpy()[0]
        res.append({
            'text': t,
            'toxicity': dict(zip(TOXIC_LABELS, pt)),
            'positivity': dict(zip(POSITIVE_LABELS, pp)),
        })
    return res

if __name__ == '__main__':
    out = soft_scores(SAMPLES)
    for r in out:
        print("\nTEXT:", r['text'])
        print("toxicity:")
        for k,v in r['toxicity'].items():
            print(f"  {k}: {v:.2f}")
        print("positivity:")
        for k,v in r['positivity'].items():
            print(f"  {k}: {v:.2f}")