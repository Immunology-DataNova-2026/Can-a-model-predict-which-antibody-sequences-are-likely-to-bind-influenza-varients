"""
DataNova 2026 — control experiments for the voice and antibody pipelines.

Two independent parts. Run whichever repo you are in:

    python controls_voice_antibody.py voice      # in the voice repo
    python controls_voice_antibody.py antibody   # in the antibody repo

VOICE part fills two gaps the paper names:
  * bootstrap 95% CIs for Sakar and SVD (Figure 14 currently says "pending re-run")
  * acoustic feature importance (Section 8, item 5) — which features drive the model

ANTIBODY part fills one:
  * the antigen-holdout ablation (Table 7, last row) — what a random split is worth
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path("reports"); OUT.mkdir(exist_ok=True)


def _family(col: str) -> str:
    lc = col.lower()
    if "jitter" in lc: return "jitter"
    if "shimmer" in lc: return "shimmer"
    if "harmonicity" in lc or "hnr" in lc or "nhr" in lc: return "HNR"
    if "mfcc" in lc: return "MFCC"
    if "tqwt" in lc: return "TQWT"
    return "other"


def _bootstrap_ci(hy, hp, n_iter=2000, seed=SEED):
    from sklearn.metrics import accuracy_score, roc_auc_score
    rng, aucs, accs = np.random.default_rng(seed), [], []
    for _ in range(n_iter):
        idx = rng.integers(0, len(hy), len(hy))
        if len(np.unique(hy[idx])) == 2:
            aucs.append(roc_auc_score(hy[idx], hp[idx]))
            accs.append(accuracy_score(hy[idx], (hp[idx] >= .5).astype(int)))
    alo, ahi = np.percentile(aucs, [2.5, 97.5])
    clo, chi = np.percentile(accs, [2.5, 97.5])
    return (roc_auc_score(hy, hp), alo, ahi,
            accuracy_score(hy, (hp >= .5).astype(int)), clo, chi)


def _speaker_probs(X, y, groups, model):
    from sklearn.model_selection import GroupKFold, cross_val_predict
    p = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=groups,
                          method="predict_proba")[:, 1]
    agg = (pd.DataFrame({"g": groups, "y": y, "p": p})
             .groupby("g").agg(y=("y", "first"), p=("p", "mean")))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


# ============================================================ VOICE
def run_voice():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    CSV = "external_datasets/Parkinsons-Sakar/pd_speech_features.csv"
    df = pd.read_csv(CSV, header=1).drop_duplicates().reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("id", "gender", "class")
            and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df[cols].apply(lambda s: s.fillna(s.median()))
    cols = [c for c in cols if df[c].nunique() > 1]
    X, y = df[cols].to_numpy(float), df["class"].astype(int).to_numpy()
    groups = df["id"].astype(str).to_numpy()

    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(600, class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1))
    hy, hp = _speaker_probs(X, y, groups, model)
    auc, alo, ahi, acc, clo, chi = _bootstrap_ci(hy, hp)

    lines = ["# Voice controls — Parkinson's (Sakar)", "",
             f"- speakers: {len(hy)}  ({(hy==0).sum()} healthy, {(hy==1).sum()} PD)",
             f"- per-speaker AUC: {auc:.3f}  95% CI [{alo:.3f}, {ahi:.3f}]",
             f"- per-speaker accuracy: {acc:.3f}  95% CI [{clo:.3f}, {chi:.3f}]", ""]

    # ---- which acoustic features actually matter
    mi = pd.Series(mutual_info_classif(np.nan_to_num(X), y, random_state=SEED),
                   index=cols).sort_values(ascending=False)
    model.fit(X, y)
    imp = pd.Series(model.steps[-1][1].feature_importances_, index=cols).sort_values(ascending=False)
    top25_names = mi.head(25).index
    # NOTE: pd.DataFrame({"mi": mi, "imp": imp}).head(25) silently reindexes when the
    # two Series share the same labels in a different order, so it does NOT give the
    # top-25-by-MI rows. Build it explicitly in mi's order instead.
    feat = pd.DataFrame({"mutual_information": mi.loc[top25_names],
                        "rf_importance": imp.loc[top25_names]})
    feat["family"] = [_family(c) for c in feat.index]
    feat.to_csv(OUT / "feature_importance_sakar.csv")

    fam_counts = feat["family"].value_counts()
    fam_order = ["jitter", "shimmer", "HNR", "MFCC", "TQWT", "other"]
    lines += ["## Top 15 features by mutual information", ""]
    lines += [f"{i:>2}. {n:<28} MI={v:.4f}  RF={imp[n]:.4f}"
              for i, (n, v) in enumerate(mi.head(15).items(), 1)]
    lines += ["", "Saved: reports/feature_importance_sakar.csv", "",
              "## Top-25 features grouped by acoustic family", "",
              "| family | count in top 25 | share |", "|---|---|---|"]
    for f in fam_order:
        c = int(fam_counts.get(f, 0))
        lines.append(f"| {f} | {c} | {c/25:.0%} |")

    # ---- SVD bootstrap (added: not in the original draft, requested "if straightforward")
    svd_csv = Path("external_datasets/SVD/svd_boost_features.csv")
    if svd_csv.exists():
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        sdf = pd.read_csv(svd_csv).drop_duplicates().reset_index(drop=True)
        scols = [c for c in sdf.columns if c not in ("speaker", "label")
                 and pd.api.types.is_numeric_dtype(sdf[c])]
        sdf[scols] = sdf[scols].apply(lambda s: s.fillna(s.median()))
        scols = [c for c in scols if sdf[c].nunique() > 1]
        sX = sdf[scols].to_numpy(float)
        sy = sdf["label"].astype(int).to_numpy()
        sgroups = sdf["speaker"].astype(str).to_numpy()
        smodel = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               SVC(kernel="rbf", C=10, probability=True,
                                   class_weight="balanced", random_state=SEED))
        shy, shp = _speaker_probs(sX, sy, sgroups, smodel)
        sauc, salo, sahi, sacc, sclo, schi = _bootstrap_ci(shy, shp)
        lines += ["", "# Voice controls — voice disorder (SVD)", "",
                  f"- speakers: {len(shy)}  ({(shy==0).sum()} healthy, {(shy==1).sum()} pathological)",
                  f"- per-speaker AUC: {sauc:.3f}  95% CI [{salo:.3f}, {sahi:.3f}]",
                  f"- per-speaker accuracy: {sacc:.3f}  95% CI [{sclo:.3f}, {schi:.3f}]"]
        svd_result = dict(auc=sauc, auc_ci=[salo, sahi], acc=sacc, acc_ci=[sclo, schi], n=len(shy))
    else:
        lines += ["", "# Voice controls — voice disorder (SVD)", "",
                  f"- SKIPPED: {svd_csv} not present."]
        svd_result = None

    (OUT / "controls_voice.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return dict(sakar=dict(auc=auc, auc_ci=[alo, ahi], acc=acc, acc_ci=[clo, chi], n=len(hy)),
                svd=svd_result, family_counts=fam_counts.to_dict())


# ============================================================ ANTIBODY
def run_antibody():
    """Ablate the antigen-holdout split: how much does a random split inflate the result?

    Both arms use the SAME training recipe as train.py (the run that produced the
    paper's committed numbers) - class-weighted BCE loss, gradient clipping,
    lr_encoder=1e-5/lr_head=1e-4, max_antibody_length=192, max_antigen_length=800,
    batch_size=16, 5 epochs - so the only thing that differs between arms is the
    split methodology, which is the thing being measured.

    Fixes vs. the original draft (verified against dataset.py/model.py/train.py):
      - BindingDataset takes ONE arg (a Path or DataFrame); tokenization happens in
        Collator, not the Dataset. The draft called BindingDataset(df, tok), which
        does not match the real constructor.
      - Arm A now uses the repo's own antigen_holdout_split() (seed=0, matching
        train.py's default, so it is comparable to whatever protocol produced the
        committed results) instead of `unique()[:3]`, which is pandas' arbitrary
        first-seen order, not a random or reproducible split.
      - Added the class-imbalance pos_weight and grad-norm clipping train.py uses,
        so an arm doesn't look worse merely from being trained less carefully.

    This retrains, so budget the same time as one training run per arm.
    """
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
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "voice":
        run_voice()
    elif which == "antibody":
        run_antibody()
    else:
        print(__doc__)
