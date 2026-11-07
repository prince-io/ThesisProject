"""Generate curated baseline records from a training/evaluation run."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _front_matter(title: str, sources: list[str]) -> str:
    lines = [
        "---",
        f"title: {title}",
        "status: validated",
        f"date: {date.today().isoformat()}",
        "sources:",
    ]
    lines.extend(f"  - {source}" for source in sources)
    lines.append("---")
    return "\n".join(lines)


def _overall_table(metrics: dict) -> list[str]:
    lines = [
        "| Split | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, values in metrics["overall"].items():
        lines.append(
            f"| {split} | {_fmt(values['precision'])} | {_fmt(values['recall'])} | "
            f"{_fmt(values['map50'])} | {_fmt(values['map50_95'])} |"
        )
    return lines


def _per_class_table(metrics: dict) -> list[str]:
    splits = list(metrics["overall"])
    classes = sorted(metrics["overall"][splits[0]]["per_class"])
    header = [
        "| Class | "
        + " | ".join(f"{split} AP50 | {split} AP50-95" for split in splits)
        + " |"
    ]
    separator = ["|---|" + "---:|" * (2 * len(splits))]
    rows = []
    for name in classes:
        values = []
        for split in splits:
            entry = metrics["overall"][split]["per_class"][name]
            values.append(f"{_fmt(entry['ap50'])} | {_fmt(entry['ap50_95'])}")
        rows.append(f"| {name} | " + " | ".join(values) + " |")
    return header + separator + rows


def _group_table(metrics: dict, group: str) -> list[str]:
    splits = list(metrics["overall"])
    names = sorted(metrics[group])
    header = [
        "| Name | "
        + " | ".join(f"{split} mAP50 | {split} mAP50-95" for split in splits)
        + " |"
    ]
    separator = ["|---|" + "---:|" * (2 * len(splits))]
    rows = []
    for name in names:
        values = []
        for split in splits:
            entry = metrics[group][name].get(split)
            values.append(
                f"{_fmt(entry['map50'] if entry else None)} | "
                f"{_fmt(entry['map50_95'] if entry else None)}"
            )
        rows.append(f"| {name} | " + " | ".join(values) + " |")
    return header + separator + rows


def _efficiency_table(metrics: dict) -> list[str]:
    efficiency = metrics["efficiency"]
    return [
        "| Metric | Value |",
        "|---|---:|",
        f"| Parameters (M) | {_fmt(efficiency['params_m'], 3)} |",
        f"| GFLOPs (640) | {_fmt(efficiency['gflops'], 3)} |",
        f"| Latency (ms/image) | {_fmt(efficiency['latency_ms'], 3)} |",
        f"| FPS | {_fmt(efficiency['fps'], 2)} |",
        f"| Peak VRAM (MB) | {_fmt(efficiency['peak_vram_mb'], 1)} |",
    ]


def _figures(results_path: Path, figures_dir: Path | None) -> list[str]:
    if figures_dir is None or not figures_dir.exists():
        return ["- Figures not copied; see the evaluation run directory."]
    try:
        prefix = figures_dir.resolve().relative_to(results_path.parent.resolve())
    except ValueError:
        prefix = figures_dir
    lines = []
    for path in sorted(figures_dir.glob("*.png")):
        lines.append(f"- [{path.name}]({prefix / path.name})")
    return lines or ["- No figures found."]


def _write_results(
    train_run: Path,
    eval_run: Path,
    results_path: Path,
    figures_dir: Path | None,
    metrics: dict,
    run_meta: dict,
) -> None:
    sources = [str(train_run), str(eval_run)]
    lines = [
        _front_matter("Baseline Results — YOLOv8n", sources),
        "",
        "# Baseline Results — YOLOv8n",
        "",
        f"- Train run: `{train_run}`",
        f"- Evaluation run: `{eval_run}`",
        f"- Weights: `{metrics['weights']}`",
        f"- Seed: {run_meta['config'].get('seed')}",
        f"- Class weighting power (`cls_pw`): {run_meta['config'].get('cls_pw')}",
        "",
        "## Overall",
        "",
        *_overall_table(metrics),
        "",
        "## Per class",
        "",
        *_per_class_table(metrics),
        "",
        "## Per condition",
        "",
        *_group_table(metrics, "per_condition"),
        "",
        "## Per source",
        "",
        *_group_table(metrics, "per_source"),
        "",
        "## Efficiency",
        "",
        *_efficiency_table(metrics),
        "",
        "## Figures",
        "",
        *_figures(results_path, figures_dir),
        "",
        "## Artifacts",
        "",
        f"- Training directory: `{train_run}`",
        f"- Evaluation directory: `{eval_run}`",
        f"- Config: `{run_meta['config'].get('data')}`",
        f"- Git commit: `{run_meta.get('git_commit', '')}`",
        "",
        "_Facts only; interpretation belongs in `thesis/findings/`._",
        "",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines), encoding="utf-8")


def _write_experiment(
    train_run: Path,
    eval_run: Path,
    experiment_path: Path,
    metrics: dict,
    run_meta: dict,
) -> None:
    config = run_meta["config"]
    overall = metrics["overall"]
    lines = [
        _front_matter("EXP-0001 — YOLOv8n Baseline", [str(train_run), str(eval_run)]),
        "",
        "# EXP-0001 — YOLOv8n Baseline",
        "",
        "## Objective",
        "",
        "Establish the reference YOLOv8n detector on `adverse16k` under the thesis "
        "training protocol.",
        "",
        "## Setup",
        "",
        f"- Model: `{config.get('model')}` (pretrained, `cls_remap=True`)",
        f"- Data: `{config.get('data')}` (16,000 images; 4,000 per condition)",
        f"- Input: {config.get('imgsz')}x{config.get('imgsz')}; "
        f"batch {config.get('batch')}; AMP {config.get('amp')}",
        f"- Effective batch: `nbs={config.get('nbs')}`",
        f"- Optimizer: {config.get('optimizer')}, lr0={config.get('lr0')}, "
        f"cosine={config.get('cos_lr')}",
        f"- Epochs: {config.get('epochs')}; patience {config.get('patience')}",
        f"- Class weighting: `cls_pw={config.get('cls_pw')}`",
        f"- Augmentation: mosaic={config.get('mosaic')}, mixup={config.get('mixup')}, "
        f"close_mosaic={config.get('close_mosaic')}",
        f"- Seed: {config.get('seed')}",
        f"- Hardware: {run_meta.get('gpu')} ({run_meta.get('gpu_total_mb')} MB)",
        f"- Peak VRAM: {_fmt(run_meta.get('peak_vram_mb'), 1)} MB",
        f"- Git commit: `{run_meta.get('git_commit', '')}`",
        "",
        "## Command",
        "",
        "```bash",
        run_meta.get("command", ""),
        "```",
        "",
        "## Results",
        "",
        "See `thesis/results/baseline.md` for the full tables.",
        "",
        "- Val mAP50 / mAP50-95: "
        f"{_fmt(overall['val']['map50'])} / {_fmt(overall['val']['map50_95'])}",
        f"- Test mAP50 / mAP50-95: "
        f"{_fmt(overall['test']['map50']) if 'test' in overall else '—'} / "
        f"{_fmt(overall['test']['map50_95']) if 'test' in overall else '—'}",
        "",
        "## Artifacts",
        "",
        f"- Training: `{train_run}`",
        f"- Evaluation: `{eval_run}`",
        "- Results: `thesis/results/baseline.md`",
        "- Figures: `thesis/results/figures/baseline/`",
        "",
    ]
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Document a training/evaluation run")
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("thesis/results/baseline.md"))
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("thesis/experiments/EXP-0001-baseline.md"),
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("thesis/results/figures/baseline")
    )
    args = parser.parse_args()

    run_meta = json.loads((args.train_run / "run_meta.json").read_text(encoding="utf-8"))
    resolved = args.train_run / "resolved_config.yaml"
    if resolved.exists():
        run_meta["config"] = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    metrics = json.loads((args.eval_run / "metrics.json").read_text(encoding="utf-8"))
    _write_results(
        args.train_run, args.eval_run, args.results, args.figures_dir, metrics, run_meta
    )
    _write_experiment(
        args.train_run, args.eval_run, args.experiment, metrics, run_meta
    )
    print(args.results)
    print(args.experiment)


if __name__ == "__main__":
    main()
