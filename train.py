"""Train the antibody-antigen binding model on AVIDa-SARS-CoV-2.

Run: python train.py [--epochs 5] [--batch-size 16] [--backbone facebook/esm2_t6_8M_UR50D]

Notes on the split used here:
  - data/processed/test.csv is reserved as the untouched final test set (see evaluate.py).
  - data/processed/train.csv is further split into a fit set and a cold-validation
    set by holding a handful of antigens out entirely (see dataset.antigen_holdout_split),
    so validation AUROC/AUPRC during training reflects generalization to antigens the
    model never trained on, not memorization.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BindingDataset, Collator, antigen_holdout_split, build_tokenizer
from model import AntibodyAntigenBindingModel

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "outputs" / "checkpoints"


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["antibody_input_ids"].to(device),
                batch["antibody_attention_mask"].to(device),
                batch["antigen_input_ids"].to(device),
                batch["antigen_attention_mask"].to(device),
            )["binding_logit"]
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch["labels"])
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--max-antibody-length", type=int, default=192)
    parser.add_argument("--max-antigen-length", type=int, default=800)
    parser.add_argument("--num-cross-attn-layers", type=int, default=2)
    parser.add_argument("--n-holdout-antigens", type=int, default=3)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.backbone)
    collate = Collator(tokenizer, args.max_antibody_length, args.max_antigen_length)

    fit_df, val_df = antigen_holdout_split(
        PROCESSED_DIR / "train.csv", n_holdout_antigens=args.n_holdout_antigens, seed=args.seed
    )
    print(f"fit set: {len(fit_df)} rows, {fit_df['Ag_label'].nunique()} antigens")
    print(f"cold-validation set: {len(val_df)} rows, {val_df['Ag_label'].nunique()} antigens "
          f"({sorted(val_df['Ag_label'].unique())})")

    fit_ds = BindingDataset(fit_df)
    val_ds = BindingDataset(val_df)

    fit_loader = DataLoader(
        fit_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers
    )

    model = AntibodyAntigenBindingModel(
        backbone=args.backbone,
        num_cross_attn_layers=args.num_cross_attn_layers,
        freeze_encoders=args.freeze_encoders,
    ).to(device)

    encoder_params = list(model.antibody_encoder.parameters()) + list(model.antigen_encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in encoder_params if p.requires_grad], "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ]
    )

    # class imbalance: weight the positive class by the fit-set negative/positive ratio
    n_pos = (fit_df["label"] == 1).sum()
    n_neg = (fit_df["label"] == 0).sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight (neg/pos ratio): {pos_weight.item():.3f}")

    best_auprc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(fit_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(
                batch["antibody_input_ids"].to(device),
                batch["antibody_attention_mask"].to(device),
                batch["antigen_input_ids"].to(device),
                batch["antigen_attention_mask"].to(device),
            )["binding_logit"]
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch["labels"])
            pbar.set_postfix(loss=loss.item())

        train_loss = total_loss / len(fit_ds)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"cold_val_auroc={val_metrics['auroc']:.4f} cold_val_auprc={val_metrics['auprc']:.4f}"
        )

        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            ckpt_path = CKPT_DIR / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "cold_val_auprc": best_auprc,
                },
                ckpt_path,
            )
            print(f"  saved new best checkpoint -> {ckpt_path} (auprc={best_auprc:.4f})")

    print(f"training done. best cold-validation AUPRC: {best_auprc:.4f}")


if __name__ == "__main__":
    main()
