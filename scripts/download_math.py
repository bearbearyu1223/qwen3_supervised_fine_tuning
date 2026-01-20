"""
Download the MATH dataset (Hendrycks et al., 2021) from HuggingFace.

Running:

```
python scripts/download_math.py
```

This will save the dataset to data/math/ directory.

Note: The original hendrycks/competition_math dataset was disabled due to DMCA.
This script uses a mirror: nlile/hendrycks-MATH-benchmark
"""
import argparse
import json
import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)

# The original dataset (hendrycks/competition_math) was taken down due to DMCA.
# Using a mirror instead.
DATASET_NAME = "nlile/hendrycks-MATH-benchmark"


def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading MATH dataset from HuggingFace ({DATASET_NAME})...")
    dataset = load_dataset(DATASET_NAME)

    for split in dataset.keys():
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        logger.info(f"Writing {len(dataset[split])} examples to {output_path}")

        with open(output_path, "w") as f:
            for example in dataset[split]:
                f.write(json.dumps(example) + "\n")

    logger.info(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Download the MATH dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/math",
        help="Directory to save the dataset (default: data/math)",
    )
    args = parser.parse_args()
    main(args.output_dir)
