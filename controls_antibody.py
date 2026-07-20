from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

SEED = 42
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

def run_antibody():
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from dataset import BindingDataset, Collator, antigen_holdout_split, build_tokenizer
    from model import AntibodyAntigenBindingModel

    PROCESSED = Path("data") / "processed"
    TRAIN_CSV = PROCESSED / "train.csv"
    df = pd.read_csv(TRAIN_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tok = build_tokenizer("facebook/esm2_t6_8M_UR50D")
    collate = Collator(tok, max_antibody_length=192, max_antigen_length=800)

    def train_eval(fit_df, val_df, tag):
        model = AntibodyAntigenBindingModel(num_cross_attn_layers=2).to(device)
        encoder_params = list(model.antibody_encoder.parameters()) + list(model.antigen_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        opt = torch.optim.AdamW([{"params": encoder_params, "lr": 1e-5},
                                 {"params": head_params, "lr": 1e-4}])
        n_pos = (fit_df["label"] == 1).sum()
        n_neg = (fit_df["label"] == 0).sum()
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dl = DataLoader(BindingDataset(fit_df), batch_size=16, shuffle=True, collate_fn=collate)
        for epoch in range(5):
            model.train()
            for b in dl:
                opt.zero_grad()
                out = model(b["antibody_input_ids"].to(device), b["antibody_attention_mask"].to(device),
                            b["antigen_input_ids"].to(device), b["antigen_attention_mask"].to(device))
                loss = lossf(out["binding_logit"], b["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            print(f"  [{tag}] epoch {epoch+1}/5 done")
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in DataLoader(BindingDataset(val_df), batch_size=32, collate_fn=collate):
                out = model(b["antibody_input_ids"].to(device), b["antibody_attention_mask"].to(device),
                            b["antigen_input_ids"].to(device), b["antigen_attention_mask"].to(device))
                ps.append(torch.sigmoid(out["binding_logit"]).cpu()); ys.append(b["labels"])
        ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        r = {"arm": tag, "auroc": float(roc_auc_score(ys, ps)),
             "auprc": float(average_precision_score(ys, ps)),
             "n_fit": int(len(fit_df)), "n_val": int(len(ys)),
             "pos_weight": float(pos_weight)}
        print(f"  {tag:22} AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  (n_fit={r['n_fit']:,}, n_val={r['n_val']:,})")
        return r

    print("Arm A - antigen holdout (dataset.antigen_holdout_split, seed=0, n=3 - matches train.py's default):")
    fit_a, val_a = antigen_holdout_split(TRAIN_CSV, n_holdout_antigens=3, seed=0)
    a = train_eval(fit_a, val_a, "antigen holdout")

    print("Arm B - random split (same val-set SIZE as Arm A, stratified by label; the shortcut being measured):")
    fit_b, val_b = train_test_split(df, test_size=len(val_a) / len(df),
                                    random_state=SEED, stratify=df["label"])
    b = train_eval(fit_b, val_b, "random split")

    res = {"arms": [a, b], "inflation_auroc": round(b["auroc"] - a["auroc"], 4),
           "inflation_auprc": round(b["auprc"] - a["auprc"], 4),
           "held_out_antigens": sorted(val_a["Ag_label"].unique().tolist())}
    (OUT / "controls_antibody.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  => random split inflates AUROC by {res['inflation_auroc']:+.4f}, AUPRC by {res['inflation_auprc']:+.4f}")
    print("  Paste into Table 7, row 'Antigen holdout'.")

if __name__ == "__main__":
    run_antibody()
