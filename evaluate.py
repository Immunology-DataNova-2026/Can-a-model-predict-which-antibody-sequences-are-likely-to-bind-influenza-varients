"""Evaluate a trained checkpoint on the held-out AVIDa-SARS-CoV-2 test set.

Run: python evaluate.py [--checkpoint outputs/checkpoints/best_model.pt]

Reports overall AUROC/AUPRC plus a per-antigen breakdown, so generalization
failures on specific variants are visible rather than hidden in an aggregate score.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BindingDataset, Collator, build_tokenizer
from model import AntibodyAntigenBindingModel

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    print(f"loaded checkpoint from epoch {ckpt['epoch']} "
          f"(cold-validation AUPRC at save time: {ckpt['cold_val_auprc']:.4f})")

    tokenizer = build_tokenizer(train_args["backbone"])
    collate = Collator(tokenizer, train_args["max_antibody_length"], train_args["max_antigen_length"])

    test_ds = BindingDataset(PROCESSED_DIR / "test.csv")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers
    )

    model = AntibodyAntigenBindingModel(
        backbone=train_args["backbone"],
        num_cross_attn_layers=train_args["num_cross_attn_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_probs, all_ag_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluating"):
            logits = model(
                batch["antibody_input_ids"].to(device),
                batch["antibody_attention_mask"].to(device),
                batch["antigen_input_ids"].to(device),
                batch["antigen_attention_mask"].to(device),
            )["binding_logit"]
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"])
            all_ag_labels.extend(batch["ag_label"])

    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    overall_auroc = roc_auc_score(labels, probs)
    overall_auprc = average_precision_score(labels, probs)
    print(f"\noverall test AUROC: {overall_auroc:.4f}")
    print(f"overall test AUPRC: {overall_auprc:.4f}")

    results_df = pd.DataFrame({"Ag_label": all_ag_labels, "label": labels, "prob": probs})
    rows = []
    for ag, group in results_df.groupby("Ag_label"):
        if group["label"].nunique() < 2:
            auroc, auprc = float("nan"), float("nan")
        else:
            auroc = roc_auc_score(group["label"], group["prob"])
            auprc = average_precision_score(group["label"], group["prob"])
        rows.append({"Ag_label": ag, "n": len(group), "auroc": auroc, "auprc": auprc})
    per_antigen = pd.DataFrame(rows).sort_values("auprc")
    print("\nper-antigen breakdown:")
    print(per_antigen.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test_eval_results.csv"
    per_antigen.to_csv(out_path, index=False)
    print(f"\nper-antigen results written to {out_path}")


if __name__ == "__main__":
    main()
