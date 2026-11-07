"""Generate curated run records from training and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
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


def _config_table(config: dict) -> list[str]:
    lines = ["| Key | Value |", "|---|---|"]
    for key in sorted(config):
        lines.append(f"| `{key}` | `{config[key]}` |")
    return lines


def _environment_table(meta: dict) -> list[str]:
    return [
        "| Item | Value |",
        "|---|---|",
        f"| Python | {meta.get('python')} |",
        f"| PyTorch | {meta.get('torch')} |",
        f"| Ultralytics | {meta.get('ultralytics')} |",
        f"| CUDA | {meta.get('cuda')} |",
        f"| GPU | {meta.get('gpu')} |",
        f"| GPU memory | {meta.get('gpu_total_mb')} MB |",
        f"| Git commit | `{meta.get('git_commit')}` |",
    ]


def _vram_table(meta: dict, metrics: dict, vram: dict | None) -> list[str]:
    efficiency = metrics["efficiency"]
    allocated = meta.get("training_peak_allocated_mb") or meta.get("peak_vram_mb")
    reserved = meta.get("training_peak_reserved_mb")
    if vram:
        allocated = allocated if allocated is not None else vram.get("pytorch_peak_allocated_mb")
        reserved = reserved if reserved is not None else vram.get("pytorch_peak_reserved_mb")
    lines = [
        "| Measurement | Value |",
        "|---|---:|",
        f"| Training PyTorch peak allocated | {_fmt(allocated, 1)} MB |",
        f"| Training PyTorch peak reserved | {_fmt(reserved, 1)} MB |",
    ]
    if vram:
        lines.extend(
            [
                "| Training GPU used peak (nvidia-smi) | "
                f"{_fmt(vram.get('gpu_used_peak_mb'), 1)} MB |",
                "| Training GPU used baseline | "
                f"{_fmt(vram.get('gpu_used_baseline_mb'), 1)} MB |",
                "| Training GPU used peak delta | "
                f"{_fmt(vram.get('gpu_used_peak_delta_mb'), 1)} MB |",
            ]
        )
    lines.extend(
        [
            f"| Inference PyTorch peak allocated | {_fmt(efficiency.get('peak_vram_mb'), 1)} MB |",
        ]
    )
    if vram:
        lines.extend(
            [
                "",
                "_GPU-used and reserved values come from an identical 1-epoch calibration run._",
            ]
        )
    return lines


def _commands(
    train_run: Path,
    eval_run: Path,
    config_path: str,
    results_path: Path,
    experiment_path: Path,
    figures_dir: Path,
    vram_report: Path | None,
    title: str,
    experiment_title: str,
) -> list[str]:
    vram_flag = f" --vram-report {vram_report}" if vram_report else ""
    return [
        "```bash",
        f".venv/bin/python scripts/train.py --config {config_path} --name {train_run.name}",
        ".venv/bin/python scripts/evaluate.py "
        f"--weights {train_run}/weights/best.pt "
        "--data configs/datasets/adverse16k.yaml "
        f"--splits val test --out {eval_run} "
        "--subsets data/processed/adverse16k/subsets "
        f"--figures-dir {figures_dir}",
        ".venv/bin/python scripts/error_analysis.py "
        f"--weights {train_run}/weights/best.pt "
        f"--out {eval_run}/error_analysis --splits val test",
        ".venv/bin/python scripts/document_results.py "
        f"--train-run {train_run} --eval-run {eval_run} "
        f"--results {results_path} --experiment {experiment_path} "
        f"--figures-dir {figures_dir}{vram_flag} "
        f'--title "{title}" --experiment-title "{experiment_title}"',
        "```",
    ]


def _training_summary(train_run: Path) -> list[str]:
    results_path = train_run / "results.csv"
    if not results_path.exists():
        return []
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    total_hours = float(rows[-1]["time"]) / 3600
    return [
        "## Training",
        "",
        f"- Epochs completed: {len(rows)}",
        f"- Best epoch (mAP50-95): {best['epoch']}",
        f"- Best val mAP50 / mAP50-95: "
        f"{_fmt(float(best['metrics/mAP50(B)']))} / {_fmt(float(best['metrics/mAP50-95(B)']))}",
        f"- Total training time: {total_hours:.2f} h",
        "",
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
    vram: dict | None,
    title: str,
) -> None:
    sources = [str(train_run), str(eval_run)]
    lines = [
        _front_matter(title, sources),
        "",
        f"# {title}",
        "",
        f"- Train run: `{train_run}`",
        f"- Evaluation run: `{eval_run}`",
        f"- Weights: `{metrics['weights']}`",
        f"- Seed: {run_meta['config'].get('seed')}",
        f"- Class weighting power (`cls_pw`): {run_meta['config'].get('cls_pw')}",
        f"- Dataset checksum: `{run_meta.get('dataset_checksums_sha1')}`",
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
        *_training_summary(train_run),
        "## Efficiency",
        "",
        f"- Parameters: {_fmt(metrics['efficiency']['params_m'], 3)} M",
        f"- GFLOPs at 640: {_fmt(metrics['efficiency']['gflops'], 3)}",
        f"- Inference latency: {_fmt(metrics['efficiency']['latency_ms'], 3)} ms/image",
        f"- FPS: {_fmt(metrics['efficiency']['fps'], 2)}",
        "",
        "### VRAM",
        "",
        *_vram_table(run_meta, metrics, vram),
        "",
        "## Figures",
        "",
        *_figures(results_path, figures_dir),
        "",
        "## Artifacts",
        "",
        f"- Training directory: `{train_run}`",
        f"- Evaluation directory: `{eval_run}`",
        f"- Run config: `{train_run}/resolved_config.yaml`",
        f"- Error analysis: `{eval_run}/error_analysis/`",
        f"- Error-analysis facts: `thesis/results/{results_path.stem}-error-analysis.md`",
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
    results_path: Path,
    figures_dir: Path,
    vram_report: Path | None,
    metrics: dict,
    run_meta: dict,
    vram: dict | None,
    title: str,
    experiment_title: str,
) -> None:
    config = run_meta["config"]
    overall = metrics["overall"]
    command = run_meta.get("command", "")
    config_path = (
        command.split("--config")[-1].strip().split()[0]
        if "--config" in command
        else "configs/train/baseline.yaml"
    )
    lines = [
        _front_matter(experiment_title, [str(train_run), str(eval_run)]),
        "",
        f"# {experiment_title}",
        "",
        "## Objective",
        "",
        "Record the detector run and its measured outcomes on `adverse16k`.",
        "",
        "## Environment",
        "",
        *_environment_table(run_meta),
        "",
        "## Configuration",
        "",
        *_config_table(config),
        "",
        "## Commands",
        "",
        *_commands(
            train_run,
            eval_run,
            config_path,
            results_path,
            experiment_path,
            figures_dir,
            vram_report,
            title,
            experiment_title,
        ),
        "",
        "## Dataset",
        "",
        f"- Dataset config: `{config.get('data')}`",
        f"- Checksums: `{run_meta.get('dataset_checksums')}`",
        f"- Checksums SHA-1: `{run_meta.get('dataset_checksums_sha1')}`",
        "",
        "## VRAM",
        "",
        *_vram_table(run_meta, metrics, vram),
        "",
        "## Results",
        "",
        "See the sibling results document for the full tables.",
        "",
        "- Val mAP50 / mAP50-95: "
        f"{_fmt(overall['val']['map50'])} / {_fmt(overall['val']['map50_95'])}",
        f"- Test mAP50 / mAP50-95: "
        f"{_fmt(overall['test']['map50']) if 'test' in overall else '—'} / "
        f"{_fmt(overall['test']['map50_95']) if 'test' in overall else '—'}",
        "",
        "## Error Analysis",
        "",
        f"- Raw artifacts: `{eval_run}/error_analysis/`",
        f"- Curated facts: `thesis/results/{results_path.stem}-error-analysis.md`",
        "",
        "## Artifacts",
        "",
        f"- Training: `{train_run}`",
        f"- Evaluation: `{eval_run}`",
        f"- Figures: `thesis/results/figures/{train_run.name}/`",
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
    parser.add_argument("--vram-report", type=Path, default=None)
    parser.add_argument("--title", default="Baseline Results — YOLOv8n")
    parser.add_argument("--experiment-title", default="EXP-0001 — YOLOv8n Baseline")
    args = parser.parse_args()

    run_meta = json.loads((args.train_run / "run_meta.json").read_text(encoding="utf-8"))
    resolved = args.train_run / "resolved_config.yaml"
    if resolved.exists():
        run_meta["config"] = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    metrics = json.loads((args.eval_run / "metrics.json").read_text(encoding="utf-8"))
    vram = (
        json.loads(args.vram_report.read_text(encoding="utf-8"))
        if args.vram_report and args.vram_report.exists()
        else None
    )
    _write_results(
        args.train_run,
        args.eval_run,
        args.results,
        args.figures_dir,
        metrics,
        run_meta,
        vram,
        args.title,
    )
    _write_experiment(
        args.train_run,
        args.eval_run,
        args.experiment,
        args.results,
        args.figures_dir,
        args.vram_report,
        metrics,
        run_meta,
        vram,
        args.title,
        args.experiment_title,
    )
    print(args.results)
    print(args.experiment)


if __name__ == "__main__":
    main()
