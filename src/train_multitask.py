import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from .config import POSITIVE_LABELS, TOXIC_LABELS, GOEMOTIONS_CSV, JIGSAW_TRAIN_CSV, ARTIFACT_DIR
from .utils import ensure_dir, compute_pos_weight

@dataclass
class Args:
    model_name: str = 'bert-base-uncased'
    max_len: int = 192
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    alpha: float = 1.0  # toxic loss weight
    beta: float = 1.0   # positive loss weight
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class MixedDataset(Dataset):
    def __init__(self, jigsaw_csv, goemotions_csv, tokenizer, max_len=192):
        self.tok = tokenizer
        self.max_len = max_len
        # Load jigsaw
        jg = pd.read_csv(jigsaw_csv)
        jg = jg.rename(columns={'comment_text':'text'})
        for c in TOXIC_LABELS:
            if c not in jg.columns:
                raise ValueError(f"Jigsaw missing {c}")
        jg = jg[['text'] + TOXIC_LABELS]
        jg['__src__'] = 'jigsaw'
        # Load goemotions
        ge = pd.read_csv(goemotions_csv)
        if 'text' not in ge.columns and 'comment_text' in ge.columns:
            ge = ge.rename(columns={'comment_text':'text'})
        present = [c for c in POSITIVE_LABELS if c in ge.columns]
        ge = ge[['text'] + present]
        for c in POSITIVE_LABELS:
            if c not in ge.columns:
                ge[c] = 0
        ge['__src__'] = 'goemotions'
        self.df = pd.concat([jg, ge], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(str(r['text']), truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k:v.squeeze(0) for k,v in enc.items()}
        if r['__src__'] == 'jigsaw':
            yT = torch.tensor(r[TOXIC_LABELS].values.astype('float32'))
            yP = torch.full((len(POSITIVE_LABELS),), float('nan'))
        else:
            yP = torch.tensor(r[POSITIVE_LABELS].values.astype('float32'))
            yT = torch.full((len(TOXIC_LABELS),), float('nan'))
        item['yT'] = yT
        item['yP'] = yP
        return item

class CivicBalance(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hid = self.backbone.config.hidden_size
        self.toxic_head = nn.Linear(hid, len(TOXIC_LABELS))
        self.pos_head   = nn.Linear(hid, len(POSITIVE_LABELS))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        return self.toxic_head(out), self.pos_head(out)


def train(**kwargs):
    args = Args(**kwargs)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = MixedDataset(JIGSAW_TRAIN_CSV, GOEMOTIONS_CSV, tok, max_len=args.max_len)

    # Split
    val_ratio = 0.1
    n_val = int(len(ds)*val_ratio)
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val])
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va, batch_size=args.batch_size)

    model = CivicBalance(args.model_name).to(args.device)

    # pos_weight from full CSVs
    jg = pd.read_csv(JIGSAW_TRAIN_CSV).rename(columns={'comment_text':'text'})
    yT = jg[TOXIC_LABELS].values.astype('float32')
    ge = pd.read_csv(GOEMOTIONS_CSV)
    if 'text' not in ge.columns and 'comment_text' in ge.columns:
        ge = ge.rename(columns={'comment_text':'text'})
    for c in POSITIVE_LABELS:
        if c not in ge.columns:
            ge[c] = 0
    yP = ge[POSITIVE_LABELS].values.astype('float32')

    posw_T = torch.tensor(compute_pos_weight(yT), device=args.device)
    posw_P = torch.tensor(compute_pos_weight(yP), device=args.device)

    crit_T = nn.BCEWithLogitsLoss(pos_weight=posw_T, reduction='none')
    crit_P = nn.BCEWithLogitsLoss(pos_weight=posw_P, reduction='none')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * math.ceil(n_train/args.batch_size)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio*total_steps), total_steps)

    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for batch in tr_dl:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            yT = batch['yT'].to(args.device)
            yP = batch['yP'].to(args.device)

            opt.zero_grad()
            logitT, logitP = model(input_ids, attention_mask)

            # Masked multi-task losses
            maskT = ~torch.isnan(yT)
            maskP = ~torch.isnan(yP)
            yTn = torch.nan_to_num(yT)
            yPn = torch.nan_to_num(yP)

            lossT_all = crit_T(logitT, yTn)
            lossP_all = crit_P(logitP, yPn)

            lossT = (lossT_all * maskT.float()).sum() / maskT.float().sum().clamp(min=1.0)
            lossP = (lossP_all * maskP.float()).sum() / maskP.float().sum().clamp(min=1.0)

            loss = args.alpha*lossT + args.beta*lossP
            loss.backward()
            opt.step(); sch.step()
            tr_loss += loss.item()

        print(f"Epoch {ep+1}/{args.epochs} - train_loss: {tr_loss/len(tr_dl):.4f}")

        # quick validation micro-F1
        from sklearn.metrics import f1_score
        model.eval()
        predsT, targsT, predsP, targsP = [], [], [], []
        with torch.no_grad():
            for batch in va_dl:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                yT = batch['yT']
                yP = batch['yP']
                logitT, logitP = model(input_ids, attention_mask)
                pT = (logitT.sigmoid().cpu().numpy() >= 0.5).astype(int)
                pP = (logitP.sigmoid().cpu().numpy() >= 0.5).astype(int)
                maskT = ~torch.isnan(yT)
                maskP = ~torch.isnan(yP)
                if maskT.any():
                    predsT.append(pT[maskT.any(dim=1)])
                    targsT.append(torch.nan_to_num(yT)[maskT.any(dim=1)].numpy())
                if maskP.any():
                    predsP.append(pP[maskP.any(dim=1)])
                    targsP.append(torch.nan_to_num(yP)[maskP.any(dim=1)].numpy())
        import numpy as np
        if targsT:
            PT = np.vstack(predsT); TT = np.vstack(targsT)
            print("  Toxic micro-F1:", f1_score(TT, PT, average='micro'))
        if targsP:
            PP = np.vstack(predsP); TP = np.vstack(targsP)
            print("  Positive micro-F1:", f1_score(TP, PP, average='micro'))

    ensure_dir(ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'civicbalance_bert.pt'))
    print('Saved model to', os.path.join(ARTIFACT_DIR, 'civicbalance_bert.pt'))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='bert-base-uncased')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_len', type=int, default=192)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    args = vars(ap.parse_args())
    train(**args)