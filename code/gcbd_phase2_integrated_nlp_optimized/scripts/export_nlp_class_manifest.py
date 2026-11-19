#!/usr/bin/env python3
"""Export label order from a locally staged Hugging Face DatasetDict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-prefix", default="The text category is")
    args = parser.parse_args()
    loaded = load_from_disk(args.dataset_path)
    split = loaded["train"] if "train" in loaded else loaded
    feature = split.features.get("label")
    names = getattr(feature, "names", None)
    if not names:
        raise ValueError(
            "The saved dataset label feature does not expose ClassLabel.names. "
            "Provide a manually audited ordered class manifest instead of guessing label order."
        )
    class_names = [str(x) for x in names]
    payload = {
        "dataset": args.dataset,
        "class_names": class_names,
        "class_texts": [f"{args.prompt_prefix} {name}." for name in class_names],
        "label_order_source": "saved Hugging Face DatasetDict train.features['label'].names",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(class_names)} ordered class prompts to {output}")


if __name__ == "__main__":
    main()
