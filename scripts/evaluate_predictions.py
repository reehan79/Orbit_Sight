from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from orbitsight.evaluation import evaluate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OrbitSight prediction files")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = evaluate_dataset(args.gt_dir, args.pred_dir, args.iou)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
