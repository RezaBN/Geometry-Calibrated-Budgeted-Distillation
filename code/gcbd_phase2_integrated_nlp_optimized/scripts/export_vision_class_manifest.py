#!/usr/bin/env python3
"""Export exact Vision classifier order and readable class prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision.datasets import CIFAR100, ImageFolder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("CIFAR100", "ImageNet"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imagenet-label-map", help="JSON mapping WNID to readable label; required for ImageNet")
    args = parser.parse_args()
    root = Path(args.root)

    if args.dataset == "CIFAR100":
        dataset = CIFAR100(root=str(root), train=True, download=False)
        class_names = list(dataset.classes)
        readable = [name.replace("_", " ") for name in class_names]
    else:
        dataset = ImageFolder(str(root / "train"), transform=None)
        class_names = list(dataset.classes)  # WNID order used by ImageFolder classifier labels.
        if not args.imagenet_label_map:
            raise ValueError("ImageNet requires --imagenet-label-map with every WNID mapped to a readable class name")
        label_map = json.loads(Path(args.imagenet_label_map).read_text(encoding="utf-8"))
        missing = [name for name in class_names if name not in label_map]
        if missing:
            raise ValueError(f"ImageNet label map is missing {len(missing)} WNIDs; first missing: {missing[:5]}")
        readable = [str(label_map[name]) for name in class_names]

    payload = {
        "dataset": args.dataset,
        "class_names": class_names,
        "class_texts": [f"A photograph of {name}." for name in readable],
        "class_text_source": "CIFAR100 torchvision classes" if args.dataset == "CIFAR100" else str(Path(args.imagenet_label_map).resolve()),
        "label_order_source": "torchvision dataset class order",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(class_names)} ordered class prompts to {output}")


if __name__ == "__main__":
    main()
