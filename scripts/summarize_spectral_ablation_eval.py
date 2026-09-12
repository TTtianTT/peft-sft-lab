#!/usr/bin/env python3
"""Collect HumanEval and MBPP metrics from a spectral-ablation run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _write_temperature_curve(root: Path, rows: list[dict]) -> None:
    points = []
    for row in rows:
        prefix = "temperature-tau-"
        if not row["label"].startswith(prefix):
            continue
        tau = float(row["label"][len(prefix) :].replace("p", "."))
        human = row["humaneval_pass@1"]
        mbpp = row["mbpp_pass@1"]
        if human is not None and mbpp is not None:
            points.append((tau, float(human), float(mbpp)))
    points.sort()
    if not points:
        return

    csv_lines = ["tau,humaneval_pass_at_1,mbpp_pass_at_1"]
    csv_lines.extend(f"{tau:.2f},{human:.12g},{mbpp:.12g}" for tau, human, mbpp in points)
    (root / "temperature_curve.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    width, height = 760, 470
    left, right, top, bottom = 82, 28, 45, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    all_scores = [score for _, human, mbpp in points for score in (human, mbpp)]
    y_lo = math.floor((min(all_scores) - 0.01) * 20) / 20
    y_hi = math.ceil((max(all_scores) + 0.01) * 20) / 20
    if y_hi <= y_lo:
        y_hi = y_lo + 0.05

    def xy(tau: float, score: float) -> tuple[float, float]:
        return left + tau * plot_w, top + (y_hi - score) / (y_hi - y_lo) * plot_h

    human_points = " ".join(f"{x:.1f},{y:.1f}" for tau, human, _ in points for x, y in [xy(tau, human)])
    mbpp_points = " ".join(f"{x:.1f},{y:.1f}" for tau, _, mbpp in points for x, y in [xy(tau, mbpp)])
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}</style>',
        f'<text x="{width/2:.0f}" y="25" text-anchor="middle" class="title">Qwen3-8B Magicoder spectral-temperature curve</text>',
    ]
    for i in range(6):
        score = y_lo + i * (y_hi - y_lo) / 5
        _, y = xy(0, score)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{score*100:.1f}</text>')
    for tau, _, _ in points:
        x, _ = xy(tau, y_lo)
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="tick">{tau:.2f}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<polyline points="{human_points}" fill="none" stroke="#2563eb" stroke-width="3"/>',
        f'<polyline points="{mbpp_points}" fill="none" stroke="#dc2626" stroke-width="3"/>',
    ])
    for tau, human, mbpp in points:
        for score, color in ((human, "#2563eb"), (mbpp, "#dc2626")):
            x, y = xy(tau, score)
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
    svg.extend([
        f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" class="label">temperature tau (0 = flat spectrum, 1 = original spectrum)</text>',
        f'<text x="20" y="{top+plot_h/2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 20 {top+plot_h/2:.1f})">pass@1 (%)</text>',
        f'<line x1="{left+18}" y1="{top+18}" x2="{left+48}" y2="{top+18}" stroke="#2563eb" stroke-width="3"/><text x="{left+56}" y="{top+23}" class="label">HumanEval</text>',
        f'<line x1="{left+160}" y1="{top+18}" x2="{left+190}" y2="{top+18}" stroke="#dc2626" stroke-width="3"/><text x="{left+198}" y="{top+23}" class="label">MBPP</text>',
        '</svg>',
    ])
    (root / "temperature_curve.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = label_font = title_font = ImageFont.load_default()
    draw.text((width / 2, 14), "Qwen3-8B Magicoder spectral-temperature curve", fill="#222222", font=title_font, anchor="ma")
    for i in range(6):
        score = y_lo + i * (y_hi - y_lo) / 5
        _, y = xy(0, score)
        draw.line((left, y, left + plot_w, y), fill="#dddddd", width=1)
        draw.text((left - 10, y), f"{score * 100:.1f}", fill="#222222", font=font, anchor="rm")
    draw.line((left, top, left, top + plot_h), fill="#222222", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#222222", width=2)
    for tau, _, _ in points:
        x, _ = xy(tau, y_lo)
        draw.text((x, top + plot_h + 12), f"{tau:.2f}", fill="#222222", font=font, anchor="ma")
    human_xy = [xy(tau, human) for tau, human, _ in points]
    mbpp_xy = [xy(tau, mbpp) for tau, _, mbpp in points]
    draw.line(human_xy, fill="#2563eb", width=4, joint="curve")
    draw.line(mbpp_xy, fill="#dc2626", width=4, joint="curve")
    for coords, color in ((human_xy, "#2563eb"), (mbpp_xy, "#dc2626")):
        for x, y in coords:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.text((left + plot_w / 2, height - 28), "temperature tau (0 = flat spectrum, 1 = original spectrum)", fill="#222222", font=label_font, anchor="ma")
    y_label = "pass@1 (%)"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    label_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((4, 4), y_label, fill="#222222", font=label_font, anchor="la")
    label_image = label_image.rotate(90, expand=True)
    image.paste(label_image, (7, round(top + plot_h / 2 - label_image.height / 2)), label_image)
    draw.line((left + 18, top + 18, left + 48, top + 18), fill="#2563eb", width=4)
    draw.text((left + 56, top + 18), "HumanEval", fill="#222222", font=label_font, anchor="lm")
    draw.line((left + 168, top + 18, left + 198, top + 18), fill="#dc2626", width=4)
    draw.text((left + 206, top + 18), "MBPP", fill="#222222", font=label_font, anchor="lm")
    image.save(root / "temperature_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root)
    labels = json.loads((root / "experiment_paths.json").read_text(encoding="utf-8"))
    rows = []
    for item in labels:
        row = {"label": item["label"], "adapter_path": item["adapter_path"]}
        for benchmark in ("humaneval", "mbpp"):
            path = root / "eval" / item["label"] / benchmark / "metrics.json"
            if not path.is_file():
                row[f"{benchmark}_pass@1"] = None
                row[f"{benchmark}_correct"] = None
                row[f"{benchmark}_total"] = None
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            row[f"{benchmark}_pass@1"] = metrics.get("pass@1")
            row[f"{benchmark}_correct"] = metrics.get("correct")
            row[f"{benchmark}_total"] = metrics.get("total")
        rows.append(row)

    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = ["label\thumaneval_pass@1\thumaneval_correct\tmbpp_pass@1\tmbpp_correct"]
    for row in rows:
        lines.append(
            "\t".join(
                str(value)
                for value in (
                    row["label"],
                    row["humaneval_pass@1"],
                    row["humaneval_correct"],
                    row["mbpp_pass@1"],
                    row["mbpp_correct"],
                )
            )
        )
    (root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_temperature_curve(root, rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
