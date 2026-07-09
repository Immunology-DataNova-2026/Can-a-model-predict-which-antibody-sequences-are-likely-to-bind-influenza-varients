"""Download the AVIDa-SARS-CoV-2 antibody-antigen binding dataset.

Source: COGNANO/AVIDa-SARS-CoV-2 on the Hugging Face Hub
(https://huggingface.co/datasets/COGNANO/AVIDa-SARS-CoV-2), CC BY-NC 4.0.

Fetches the three CSV files that make up the dataset:
  - train.csv, test.csv : VHH_sequence, Ag_label, label, subject_species, subject_name, subject_sex
  - antigen_sequences.csv : Ag_label -> Ag_sequence lookup

and joins the antigen amino-acid sequence onto each split so downstream
code has (antibody_sequence, antigen_sequence, label) triples directly.
"""

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "COGNANO/AVIDa-SARS-CoV-2"
REPO_TYPE = "dataset"
FILES = ["train.csv", "test.csv", "antigen_sequences.csv"]

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def download_raw_files(cache_dir: Path = None) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname in FILES:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=fname,
            local_dir=str(RAW_DIR),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        paths[fname] = Path(local_path)
        print(f"downloaded {fname} -> {local_path}")
    return paths


def build_processed_splits(paths: dict[str, Path]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    antigens = pd.read_csv(paths["antigen_sequences.csv"])
    assert {"Ag_label", "Ag_sequence"}.issubset(antigens.columns)

    for split in ["train", "test"]:
        df = pd.read_csv(paths[f"{split}.csv"])
        merged = df.merge(antigens, on="Ag_label", how="left")
        n_missing = merged["Ag_sequence"].isna().sum()
        if n_missing:
            print(f"warning: {n_missing} rows in {split}.csv have no matching antigen sequence")

        merged = merged.rename(
            columns={
                "VHH_sequence": "antibody_sequence",
                "Ag_sequence": "antigen_sequence",
            }
        )
        out_cols = [
            "antibody_sequence",
            "antigen_sequence",
            "Ag_label",
            "label",
            "subject_species",
            "subject_name",
            "subject_sex",
        ]
        merged = merged[out_cols]
        out_path = PROCESSED_DIR / f"{split}.csv"
        merged.to_csv(out_path, index=False)
        print(f"wrote {len(merged)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional custom Hugging Face cache directory",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    paths = download_raw_files(cache_dir=cache_dir)
    build_processed_splits(paths)
    print("done.")


if __name__ == "__main__":
    main()
