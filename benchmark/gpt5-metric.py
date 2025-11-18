#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import numpy as np
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any

NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

def extract_last_number(text: str) ->  Optional[float]:
    """Extract the last float/integer in a string; return None if none found."""
    if text is None:
        return None
    m = NUM_PATTERN.findall(str(text))
    return float(m[-1]) if m else None

def load_predictions(pred_path: str) -> dict[str, float]:
    """
    Load predictions JSON of the form:
      {
        "5507_img.jpg": {"response": "The reading is 4.1 mA"},
        ...
      }
    Returns: {image_name: pred_value_float}
    """
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = {}
    if isinstance(data, dict):
        for k, v in data.items():
            resp = v.get("response") if isinstance(v, dict) else v
            num = extract_last_number(resp)
            if num is not None:
                preds[k] = num
    else:
        raise ValueError("Predictions JSON should be a dict keyed by image name.")
    return preds

def load_ground_truths(gt_path: str) -> dict[str, dict]:
    """
    Load ground truth JSON of the form (array of objects):
      [
        {"image": "10325_img.jpg", "reading": "7.8", "range": 10, ...},
        ...
      ]
    Returns: {image_name: {"reading": float, "range": float}}
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gts = {}
    if isinstance(data, list):
        for item in data:
            img = item.get("image")
            if not img:
                continue
            reading = extract_last_number(item.get("reading"))
            rng = extract_last_number(item.get("range"))
            if reading is None or rng is None:
                # Skip entries without numeric reading or range
                continue
            gts[img] = {"reading": float(reading), "range": float(rng)}
    elif isinstance(data, dict):
        # Fallback: support dict mapping as well
        for img, item in data.items():
            reading = extract_last_number(item.get("reading"))
            rng = extract_last_number(item.get("range"))
            if reading is None or rng is None:
                continue
            gts[img] = {"reading": float(reading), "range": float(rng)}
    else:
        raise ValueError("Ground-truth JSON must be a list of objects or a dict.")
    return gts

def compute_metrics(matches: list[dict]) -> dict:
    """
    matches: list of dicts with keys: image, pred, gt, range, abs_err, theta, gamma,
             correct_theta (bool), correct_eps (bool), nonzero_gt (bool)
    """
    # Theta: over nonzero gt
    theta_vals = [m["theta"] for m in matches if m["nonzero_gt"] and m["theta"] is not None]
    theta_hat = float(np.mean(theta_vals)) if theta_vals else None

    # Gamma: over all with valid range
    gamma_vals = [m["gamma"] for m in matches if m["gamma"] is not None]
    gamma_hat = float(np.mean(gamma_vals)) if gamma_vals else None

    # Accuracy (Theta < 0.05): over nonzero gt
    nonzero_cnt = sum(1 for m in matches if m["nonzero_gt"])
    correct_theta_cnt = sum(1 for m in matches if m["nonzero_gt"] and m["correct_theta"])
    acc_theta = (correct_theta_cnt / nonzero_cnt) if nonzero_cnt > 0 else None

    # Accuracy@ε (ε=0.01*range): over all processed matches
    processed_cnt = len(matches)
    correct_eps_cnt = sum(1 for m in matches if m["correct_eps"])
    acc_eps = (correct_eps_cnt / processed_cnt) if processed_cnt > 0 else None

    return {
        "theta": theta_hat,
        "gamma": gamma_hat,
        "accuracy_theta": acc_theta,
        "accuracy_epsilon": acc_eps,
        "counts": {
            "processed": processed_cnt,
            "nonzero_gt": nonzero_cnt,
            "correct_theta": correct_theta_cnt,
            "correct_epsilon": correct_eps_cnt,
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Compute SLT analogue-meter metrics from two JSONs.")
    ap.add_argument("--pred", required=True, help="Path to predictions JSON (image->response).")
    ap.add_argument("--gt", required=True, help="Path to ground-truth JSON (list of {image, reading, range}).")
    ap.add_argument("--csv_out", default=None, help="Optional path to save per-sample CSV.")
    args = ap.parse_args()

    preds = load_predictions(args.pred)
    gts = load_ground_truths(args.gt)

    # Match by image name
    common = sorted(set(preds.keys()) & set(gts.keys()))
    missing_pred = sorted(set(gts.keys()) - set(preds.keys()))
    missing_gt = sorted(set(preds.keys()) - set(gts.keys()))

    matches = []
    for img in common:
        pred = float(preds[img])
        gt = float(gts[img]["reading"])
        rng = float(gts[img]["range"])

        abs_err = abs(pred - gt)

        # Theta per-sample: |pred-gt| / gt (nonzero gt), capped at 100
        if gt != 0:
            theta = min(abs_err / gt, 100.0)
            nonzero_gt = True
            correct_theta = (theta < 0.05)
        else:
            theta = None
            nonzero_gt = False
            correct_theta = False  # not counted anyway

        # Gamma per-sample: |pred-gt| / range, capped at 10
        gamma = None
        if rng is not None and rng != 0:
            gamma = min(abs_err / rng, 10.0)

        # Accuracy@ε per-sample: ε = 0.01 * range
        eps = 0.01 * rng if rng is not None else None
        correct_eps = (abs_err <= eps) if eps is not None else False

        matches.append({
            "image": img,
            "pred": pred,
            "gt": gt,
            "range": rng,
            "abs_err": abs_err,
            "theta": theta,
            "gamma": gamma,
            "correct_theta": correct_theta,
            "correct_eps": correct_eps,
            "nonzero_gt": nonzero_gt,
        })

    # Metrics
    metrics = compute_metrics(matches)

    # Print summary
    print("\n--- Coverage ---")
    print(f" Ground-truth images: {len(gts)}")
    print(f" Prediction images  : {len(preds)}")
    print(f" Matched            : {len(common)}")
    print(f" Missing in preds   : {len(missing_pred)}")
    print(f" Missing in GT      : {len(missing_gt)}")

    print("\n--- Final Evaluation Results ---")
    if metrics["theta"] is not None:
        print(f" Theta (Θ̂)                : {metrics['theta']:.4f}")
    else:
        print(" Theta (Θ̂)                : N/A (no nonzero GT)")
    if metrics["gamma"] is not None:
        print(f" Gamma (Γ̂)                : {metrics['gamma']:.4f}")
    else:
        print(" Gamma (Γ̂)                : N/A (no valid ranges)")
    if metrics["accuracy_theta"] is not None:
        print(f" Accuracy (Theta < 0.05)   : {metrics['accuracy_theta']:.4f}")
    else:
        print(" Accuracy (Theta < 0.05)   : N/A")
    if metrics["accuracy_epsilon"] is not None:
        print(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}")
    else:
        print(" Accuracy@ε                : N/A")

    c = metrics["counts"]
    print("\n--- Counts ---")
    print(f" Processed samples  : {c['processed']}")
    print(f" Nonzero GT samples : {c['nonzero_gt']}")
    print(f" Correct (Theta<.05): {c['correct_theta']}")
    print(f" Correct @ε         : {c['correct_epsilon']}")

    # Optional CSV
    if args.csv_out:
        outp = Path(args.csv_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image","pred","gt","range","abs_err","theta","gamma",
                        "correct_theta","correct_epsilon"])
            for m in matches:
                w.writerow([
                    m["image"], m["pred"], m["gt"], m["range"], m["abs_err"],
                    "" if m["theta"] is None else m["theta"],
                    "" if m["gamma"] is None else m["gamma"],
                    int(m["correct_theta"]), int(m["correct_eps"])
                ])
        print(f"\nPer-sample details saved to: {outp}")

    # Report missing keys for debugging
    if missing_pred:
        print("\nImages missing predictions (in GT but not preds):")
        for k in missing_pred[:20]:
            print(" ", k)
        if len(missing_pred) > 20:
            print(" ... (truncated)")

    if missing_gt:
        print("\nImages missing ground truths (in preds but not GT):")
        for k in missing_gt[:20]:
            print(" ", k)
        if len(missing_gt) > 20:
            print(" ... (truncated)")

if __name__ == "__main__":
    main()
