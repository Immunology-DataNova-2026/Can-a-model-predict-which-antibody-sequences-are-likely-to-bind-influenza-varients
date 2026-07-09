"""Exploratory data analysis for the AVIDa-SARS-CoV-2 binding dataset.

Reads data/processed/{train,test}.csv (produced by download_data.py) and
writes summary stats + figures to outputs/figures/.

Run: python data_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid")


def load_splits() -> pd.DataFrame:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def print_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"total rows: {len(df)}")
    print(df.groupby("split").size())
    print("\nlabel balance (0=non-binding, 1=binding):")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))
    print(f"\nunique antibody sequences: {df['antibody_sequence'].nunique()}")
    print(f"unique antigens (Ag_label): {df['Ag_label'].nunique()}")
    print(f"unique subjects: {df['subject_name'].nunique()}")

    dup_pairs = df.duplicated(subset=["antibody_sequence", "Ag_label"]).sum()
    print(f"\nduplicate (antibody, antigen) pairs: {dup_pairs}")

    # check for antibody sequences that appear with conflicting labels
    conflict = (
        df.groupby(["antibody_sequence", "Ag_label"])["label"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"antibody/antigen pairs with conflicting labels: {conflict}")
    print("=" * 60)


def plot_label_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="split", hue="label", ax=ax)
    ax.set_title("Binding label balance by split")
    ax.set_xlabel("")
    ax.legend(title="label", labels=["non-binding (0)", "binding (1)"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_balance.png", dpi=150)
    plt.close(fig)


def plot_sequence_length_distributions(df: pd.DataFrame) -> None:
    df = df.copy()
    df["antibody_len"] = df["antibody_sequence"].str.len()
    df["antigen_len"] = df["antigen_sequence"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["antibody_len"], bins=40, ax=axes[0])
    axes[0].set_title("VHH (antibody) sequence length")
    axes[0].set_xlabel("length (aa)")

    sns.histplot(df["antigen_len"], bins=40, ax=axes[1])
    axes[1].set_title("Antigen sequence length")
    axes[1].set_xlabel("length (aa)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequence_length_distributions.png", dpi=150)
    plt.close(fig)


def plot_antigen_distribution(df: pd.DataFrame) -> None:
    counts = df.groupby(["Ag_label", "split"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Sample count per antigen variant")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "antigen_distribution.png", dpi=150)
    plt.close(fig)


def plot_binding_rate_per_antigen(df: pd.DataFrame) -> None:
    rate = df.groupby("Ag_label")["label"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    rate.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Binding rate per antigen variant")
    ax.set_xlabel("fraction of binding pairs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "binding_rate_per_antigen.png", dpi=150)
    plt.close(fig)


def plot_antigen_overlap_train_test(df: pd.DataFrame) -> None:
    train_ag = set(df[df["split"] == "train"]["Ag_label"].unique())
    test_ag = set(df[df["split"] == "test"]["Ag_label"].unique())
    overlap = train_ag & test_ag
    print(f"\nantigens in train: {len(train_ag)}, in test: {len(test_ag)}, "
          f"overlap: {len(overlap)} -> "
          f"{'RANDOM split (antigens shared)' if overlap else 'COLD split (disjoint antigens)'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_splits()

    print_overview(df)
    plot_antigen_overlap_train_test(df)
    plot_label_balance(df)
    plot_sequence_length_distributions(df)
    plot_antigen_distribution(df)
    plot_binding_rate_per_antigen(df)

    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
