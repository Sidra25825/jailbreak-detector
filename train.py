"""CLI: train the detector, print cross-validation metrics, save nothing.

Usage:
    python train.py
"""
from src.dataset import build_dataset
from src.evaluate import cross_validate


def main() -> None:
    data = build_dataset(n_per_class=200)
    prompts = [p for p, _ in data]
    labels = [l for _, l in data]
    print(f"Dataset: {len(prompts)} prompts, {sum(labels)} positives")
    print("Running 5-fold cross-validation...\n")
    summary = cross_validate(prompts, labels, n_splits=5)
    for key, value in summary.items():
        print(f"  {key:16s} {value}")


if __name__ == "__main__":
    main()
