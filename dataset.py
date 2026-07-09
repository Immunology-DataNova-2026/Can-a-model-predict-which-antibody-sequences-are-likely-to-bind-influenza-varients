"""PyTorch Dataset + collation for antibody-antigen binding pairs."""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"


class BindingDataset(Dataset):
    """Wraps a processed AVIDa-SARS-CoV-2 split (antibody_sequence, antigen_sequence, label, Ag_label)."""

    def __init__(self, data: "Path | pd.DataFrame"):
        self.df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "antibody_sequence": row["antibody_sequence"],
            "antigen_sequence": row["antigen_sequence"],
            "label": float(row["label"]),
            "ag_label": row["Ag_label"],
        }


class Collator:
    """Tokenizes a batch of raw sequences into padded tensors for both encoders."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_antibody_length: int = 192,
        max_antigen_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_antibody_length = max_antibody_length
        self.max_antigen_length = max_antigen_length

    def __call__(self, batch: list[dict]) -> dict:
        antibody_seqs = [b["antibody_sequence"] for b in batch]
        antigen_seqs = [b["antigen_sequence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        antibody_enc = self.tokenizer(
            antibody_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_antibody_length,
            return_tensors="pt",
        )
        antigen_enc = self.tokenizer(
            antigen_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_antigen_length,
            return_tensors="pt",
        )

        return {
            "antibody_input_ids": antibody_enc["input_ids"],
            "antibody_attention_mask": antibody_enc["attention_mask"],
            "antigen_input_ids": antigen_enc["input_ids"],
            "antigen_attention_mask": antigen_enc["attention_mask"],
            "labels": labels,
            "ag_label": [b["ag_label"] for b in batch],
        }


def build_tokenizer(backbone: str = "facebook/esm2_t6_8M_UR50D") -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(backbone)


def antigen_holdout_split(
    train_csv: Path,
    n_holdout_antigens: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the training set into fit/cold-validation by holding entire antigens out.

    This approximates the antigen-based "cold split" evaluation described in the
    design doc: the model never sees the held-out antigens' binding pairs during
    training, so validation performance reflects generalization to novel antigens
    rather than memorization of antigen-specific patterns.
    """
    df = pd.read_csv(train_csv)
    antigens = sorted(df["Ag_label"].unique())
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(antigens), generator=rng).tolist()
    holdout = {antigens[i] for i in perm[:n_holdout_antigens]}

    val_df = df[df["Ag_label"].isin(holdout)].reset_index(drop=True)
    fit_df = df[~df["Ag_label"].isin(holdout)].reset_index(drop=True)
    return fit_df, val_df
