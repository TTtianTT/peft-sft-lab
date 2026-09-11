#!/usr/bin/env python3
"""Select the core scale/shape controls from a forgetting variant manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORE_METHODS = {
    "original_lora",
    "original_hns",
    "common_lora",
    "common_hns",
    "common_per_module",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text())
    selected = [row for row in source["variants"] if row["method"] in CORE_METHODS]
    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 core variants, found {len(selected)}")
    result = dict(source)
    result["variants"] = selected
    result["selection"] = sorted(CORE_METHODS)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[Manifest] {destination}: {len(selected)} core variants")


if __name__ == "__main__":
    main()
