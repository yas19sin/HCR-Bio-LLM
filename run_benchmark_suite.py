from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    task: str
    note: str = ""


CORE_CAUSAL_MODELS = [
    ModelSpec("transformer_baseline", "configs/transformer_baseline.yaml", "causal", "baseline"),
    ModelSpec("hcr_kan_mean", "configs/hcr_kan_mean.yaml", "causal", "kan_bridge"),
    ModelSpec("hcr_moment", "configs/hcr_moment.yaml", "causal", "moment_state"),
    ModelSpec("hcr_density", "configs/hcr_density.yaml", "causal", "density_state"),
    ModelSpec("hcr_joint_pairwise", "configs/hcr_joint_pairwise.yaml", "causal", "pairwise_state"),
]

BLOCKWISE_MODEL = ModelSpec(
    "hcr_blockwise_joint",
    "configs/hcr_blockwise_joint.yaml",
    "causal",
    "paper_direct_small",
)

REFINEMENT_MODEL = ModelSpec(
    "hcr_bidirectional_refinement",
    "configs/hcr_bidirectional_refinement.yaml",
    "denoising",
    "different_task",
)

FAIR_CAUSAL_MODELS = [
    ModelSpec("transformer_baseline", "configs/fair_4m/transformer_baseline.yaml", "causal", "fair_4m_baseline"),
    ModelSpec("hcr_kan_mean", "configs/fair_4m/hcr_kan_mean.yaml", "causal", "fair_4m_kan_bridge"),
    ModelSpec("hcr_moment", "configs/fair_4m/hcr_moment.yaml", "causal", "fair_4m_moment_state"),
    ModelSpec("hcr_density", "configs/fair_4m/hcr_density.yaml", "causal", "fair_4m_density_state"),
    ModelSpec("hcr_joint_pairwise", "configs/fair_4m/hcr_joint_pairwise.yaml", "causal", "fair_4m_pairwise_state"),
    ModelSpec("hcr_blockwise_joint", "configs/fair_4m/hcr_blockwise_joint.yaml", "causal", "fair_4m_paper_direct"),
]

FAIR_REFINEMENT_MODEL = ModelSpec(
    "hcr_bidirectional_refinement",
    "configs/fair_4m/hcr_bidirectional_refinement.yaml",
    "denoising",
    "fair_4m_different_task",
)

FAITHFUL_HCR_MODEL = ModelSpec(
    "hcr_blockwise_joint",
    "configs/faithful_hcr/ntp_stable.yaml",
    "causal",
    "faithful_hcr_ntp_stable",
)

SUITES = {
    "causal-core": CORE_CAUSAL_MODELS,
    "causal": [*CORE_CAUSAL_MODELS, BLOCKWISE_MODEL],
    "denoising": [REFINEMENT_MODEL],
    "all": [*CORE_CAUSAL_MODELS, BLOCKWISE_MODEL],
    "research-all": [*CORE_CAUSAL_MODELS, BLOCKWISE_MODEL, REFINEMENT_MODEL],
    "fair-causal": FAIR_CAUSAL_MODELS,
    "fair-denoising": [FAIR_REFINEMENT_MODEL],
    "fair-all": FAIR_CAUSAL_MODELS,
    "fair-research-all": [*FAIR_CAUSAL_MODELS, FAIR_REFINEMENT_MODEL],
    "faithful-hcr": [FAITHFUL_HCR_MODEL],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched benchmark suites without filling notebooks with raw logs. "
            "Raw train/eval/sample outputs are saved under the suite directory."
        )
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="causal")
    parser.add_argument("--models", nargs="*", help="Optional model-name subset from the selected suite.")
    parser.add_argument("--steps", type=int, default=10000, help="Matched max_steps for every selected model.")
    parser.add_argument("--seed", type=int, default=1337, help="Matched random seed for every selected model.")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--analysis-batches", type=int, default=10)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--sample-prompt", default="First ")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="runs/benchmark_suites")
    parser.add_argument("--run-name", help="Suite directory name. Defaults to suite_steps_timestamp.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocesses.")
    parser.add_argument("--skip-train", action="store_true", help="Use existing run directories/checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip final eval.py pass.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analyze_uncertainty.py pass.")
    parser.add_argument("--skip-sample", action="store_true", help="Skip causal sample generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a model command fails.")
    parser.add_argument(
        "--progress",
        default="train",
        choices=["off", "train", "all"],
        help="Mirror subprocess output to the notebook while still saving logs.",
    )
    parser.add_argument(
        "--train-stdout",
        default="compact",
        choices=["compact", "full", "quiet"],
        help="Forwarded to train.py via --set stdout=...",
    )
    return parser.parse_args()


def value_for_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_command(
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
    echo: bool = False,
    echo_prefix: str = "",
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        if dry_run:
            return 0

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if echo:
                text = line.rstrip()
                if text:
                    print(f"{echo_prefix}{text}", flush=True)
        return process.wait()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def parse_metrics(run_dir: Path) -> dict[str, Any]:
    records = read_jsonl(run_dir / "metrics.jsonl")
    start = next((record for record in records if record.get("event") == "start"), {})
    train_records = [record for record in records if record.get("event") == "train"]
    eval_records = [record for record in records if record.get("event") == "eval"]
    final_records = [record for record in records if record.get("event") == "final"]
    best_eval = min(eval_records, key=lambda item: item.get("val_loss", float("inf"))) if eval_records else {}
    return {
        "start": start,
        "last_train": train_records[-1] if train_records else {},
        "last_eval": eval_records[-1] if eval_records else {},
        "best_eval": best_eval,
        "final": final_records[-1] if final_records else {},
        "records": len(records),
    }


def command_with_overrides(python: str, config: str, overrides: dict[str, Any]) -> list[str]:
    cmd = [python, "-u", "train.py", "--config", config]
    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={value_for_override(value)}"])
    return cmd


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def metric(result: dict[str, Any], key: str) -> Any:
    eval_metrics = result.get("eval", {})
    if key in eval_metrics:
        return eval_metrics[key]
    best_eval = result.get("train_metrics", {}).get("best_eval", {})
    val_key = f"val_{key}"
    if val_key in best_eval:
        return best_eval[val_key]
    return None


def print_result_row(result: dict[str, Any], baseline_loss: float | None = None) -> None:
    loss = metric(result, "loss")
    delta = ""
    if baseline_loss is not None and result["name"] != "transformer_baseline" and loss is not None:
        delta = f" d_loss={float(loss) - baseline_loss:+.4f}"
    print(
        f"{result['name']}: status={result['status']} task={result['task']} "
        f"params={format_int(result.get('params'))} loss={format_float(loss)} "
        f"acc={format_float(metric(result, 'accuracy'))} ppl={format_float(metric(result, 'perplexity'))}"
        f"{delta}"
    )


def build_markdown(summary: dict[str, Any]) -> str:
    if str(summary["suite"]).startswith("fair-"):
        fairness_note = (
            "Fairness note: causal rows share dataset, split, context length, batch size, optimizer settings, "
            "step count, seed, eval interval, eval batch count, and an approximate 4M trainable-parameter target. "
            "They are not compute-matched, so throughput is shown. Denoising rows are a separate task."
        )
    else:
        fairness_note = (
            "Fairness note: causal rows share dataset, split, context length, batch size, optimizer settings, "
            "step count, seed, eval interval, and eval batch count through matched overrides. "
            "They are not parameter-matched, so params are shown. Denoising rows are a separate task."
        )
    lines = [
        "# Benchmark Suite Summary",
        "",
        f"- suite: `{summary['suite']}`",
        f"- steps: `{summary['steps']}`",
        f"- seed: `{summary['seed']}`",
        f"- eval_batches: `{summary['eval_batches']}`",
        f"- output_dir: `{summary['output_dir']}`",
        "",
        fairness_note,
        "",
        "## Causal LM",
        "",
        "| model | note | params | loss | d_loss | acc | ppl | ece | brier | corrupt_x | tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    causal = [item for item in summary["results"] if item["task"] == "causal"]
    baseline = next((item for item in causal if item["name"] == "transformer_baseline"), None)
    baseline_loss = metric(baseline, "loss") if baseline else None
    baseline_loss = float(baseline_loss) if baseline_loss is not None else None

    for item in causal:
        loss = metric(item, "loss")
        delta = ""
        if baseline_loss is not None and loss is not None and item["name"] != "transformer_baseline":
            delta = format_float(float(loss) - baseline_loss)
            if not delta.startswith("-"):
                delta = f"+{delta}"
        lines.append(
            "| {name} | {note} | {params} | {loss} | {delta} | {acc} | {ppl} | {ece} | {brier} | {corrupt} | {tps} |".format(
                name=item["name"],
                note=item.get("note", ""),
                params=format_int(item.get("params")),
                loss=format_float(loss),
                delta=delta,
                acc=format_float(metric(item, "accuracy")),
                ppl=format_float(metric(item, "perplexity")),
                ece=format_float(metric(item, "ece")),
                brier=format_float(metric(item, "brier")),
                corrupt=format_float(metric(item, "corruption_degradation")),
                tps=format_float(item.get("tokens_per_sec"), 1),
            )
        )

    denoising = [item for item in summary["results"] if item["task"] != "causal"]
    if denoising:
        lines.extend(
            [
                "",
                "## Separate Tasks",
                "",
                "| model | task | note | params | loss | acc | ppl | ece | brier | tok/s |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in denoising:
            lines.append(
                "| {name} | {task} | {note} | {params} | {loss} | {acc} | {ppl} | {ece} | {brier} | {tps} |".format(
                    name=item["name"],
                    task=item["task"],
                    note=item.get("note", ""),
                    params=format_int(item.get("params")),
                    loss=format_float(metric(item, "loss")),
                    acc=format_float(metric(item, "accuracy")),
                    ppl=format_float(metric(item, "perplexity")),
                    ece=format_float(metric(item, "ece")),
                    brier=format_float(metric(item, "brier")),
                    tps=format_float(item.get("tokens_per_sec"), 1),
                )
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| model | train log | metrics | eval | uncertainty | sample |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in summary["results"]:
        paths = item.get("paths", {})
        lines.append(
            "| {name} | {train_log} | {metrics} | {eval} | {uncertainty} | {sample} |".format(
                name=item["name"],
                train_log=paths.get("train_log", ""),
                metrics=paths.get("metrics", ""),
                eval=paths.get("eval", ""),
                uncertainty=paths.get("uncertainty", ""),
                sample=paths.get("sample", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Treat lower causal `loss` and `ppl` as the first-pass quality signal.",
            "- Treat lower `ece` and `brier` as calibration signals, not proof of better generation.",
            "- Treat `corrupt_x` above `1.0` as robustness cost under corrupted context.",
            "- Do not compare denoising/refinement loss directly against causal LM loss.",
            "- A variant only earns a real claim after beating the baseline under the same suite settings.",
            "",
        ]
    )
    return "\n".join(lines)


def selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    models = SUITES[args.suite]
    if not args.models:
        return models
    requested = set(args.models)
    available = {model.name for model in models}
    missing = sorted(requested - available)
    if missing:
        raise SystemExit(f"model(s) not in suite {args.suite}: {', '.join(missing)}")
    return [model for model in models if model.name in requested]


def main() -> None:
    args = parse_args()
    models = selected_models(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.suite}_{args.steps}_{timestamp}"
    suite_dir = Path(args.output_root) / run_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"suite={args.suite} models={','.join(model.name for model in models)} output_dir={suite_dir}")
    if args.dry_run:
        print("dry_run=true")

    results: list[dict[str, Any]] = []
    baseline_loss: float | None = None
    suite_start = time.perf_counter()

    for index, model in enumerate(models, start=1):
        run_dir = suite_dir / model.name
        logs_dir = run_dir / "suite_logs"
        train_log = logs_dir / "train.log"
        eval_path = logs_dir / "eval.json"
        uncertainty_path = logs_dir / "uncertainty.json"
        sample_path = logs_dir / "sample.txt"
        print(f"[{index}/{len(models)}] {model.name}: training")

        status = "ok"
        errors: list[str] = []
        elapsed_start = time.perf_counter()

        if not args.skip_train:
            overrides = {
                "output_dir": run_dir,
                "max_steps": args.steps,
                "seed": args.seed,
                "eval_interval": args.eval_interval,
                "eval_batches": args.eval_batches,
                "log_interval": args.log_interval,
                "sample_at_end": False,
                "stdout": args.train_stdout,
                "stdout_events": "start,eval,final",
                "device": args.device,
            }
            cmd = command_with_overrides(args.python, model.config, overrides)
            code = run_command(
                cmd,
                train_log,
                args.dry_run,
                echo=args.progress in {"train", "all"},
                echo_prefix=f"{model.name} | ",
            )
            if code != 0:
                status = "failed"
                errors.append(f"train.py exited {code}")
                print(f"{model.name}: train failed, see {train_log}")
                if not args.keep_going:
                    raise SystemExit(code)

        checkpoint = run_dir / "best.pt"
        if not args.dry_run and status == "ok" and not checkpoint.exists():
            status = "failed"
            errors.append("missing best.pt")
            if not args.keep_going:
                raise SystemExit(f"{model.name}: missing checkpoint {checkpoint}")

        if status == "ok" and not args.skip_eval:
            print(f"[{index}/{len(models)}] {model.name}: final eval")
            cmd = [
                args.python,
                "-u",
                "eval.py",
                "--checkpoint",
                str(checkpoint),
                "--eval-batches",
                str(args.eval_batches),
                "--device",
                args.device,
            ]
            code = run_command(
                cmd,
                eval_path,
                args.dry_run,
                echo=args.progress == "all",
                echo_prefix=f"{model.name} eval | ",
            )
            if code != 0:
                status = "failed"
                errors.append(f"eval.py exited {code}")
                if not args.keep_going:
                    raise SystemExit(code)

        if status == "ok" and not args.skip_analysis:
            print(f"[{index}/{len(models)}] {model.name}: uncertainty")
            cmd = [
                args.python,
                "-u",
                "analyze_uncertainty.py",
                "--checkpoint",
                str(checkpoint),
                "--batches",
                str(args.analysis_batches),
                "--device",
                args.device,
            ]
            code = run_command(
                cmd,
                uncertainty_path,
                args.dry_run,
                echo=args.progress == "all",
                echo_prefix=f"{model.name} uncertainty | ",
            )
            if code != 0:
                status = "failed"
                errors.append(f"analyze_uncertainty.py exited {code}")
                if not args.keep_going:
                    raise SystemExit(code)

        if status == "ok" and model.task == "causal" and not args.skip_sample:
            print(f"[{index}/{len(models)}] {model.name}: sample")
            cmd = [
                args.python,
                "-u",
                "sample.py",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                args.sample_prompt,
                "--max-new-tokens",
                str(args.sample_tokens),
                "--device",
                args.device,
            ]
            code = run_command(
                cmd,
                sample_path,
                args.dry_run,
                echo=args.progress == "all",
                echo_prefix=f"{model.name} sample | ",
            )
            if code != 0:
                status = "failed"
                errors.append(f"sample.py exited {code}")
                if not args.keep_going:
                    raise SystemExit(code)

        train_metrics = parse_metrics(run_dir)
        eval_metrics = read_json_file(eval_path) if not args.dry_run else {}
        uncertainty = read_json_file(uncertainty_path) if not args.dry_run else {}
        start_metrics = train_metrics.get("start", {})
        last_train = train_metrics.get("last_train", {})
        final_metrics = train_metrics.get("final", {})

        result = {
            "name": model.name,
            "config": model.config,
            "task": model.task,
            "note": model.note,
            "status": status,
            "errors": errors,
            "params": start_metrics.get("params") or final_metrics.get("params"),
            "tokens_per_sec": last_train.get("tokens_per_sec"),
            "elapsed_seconds": time.perf_counter() - elapsed_start,
            "train_metrics": {
                key: value for key, value in train_metrics.items() if key != "records"
            },
            "train_records": train_metrics.get("records", 0),
            "eval": eval_metrics,
            "uncertainty": uncertainty,
            "paths": {
                "run_dir": str(run_dir),
                "train_log": str(train_log),
                "metrics": str(run_dir / "metrics.jsonl"),
                "eval": str(eval_path),
                "uncertainty": str(uncertainty_path),
                "sample": str(sample_path) if model.task == "causal" else "",
            },
        }
        results.append(result)

        if model.name == "transformer_baseline":
            loss = metric(result, "loss")
            baseline_loss = float(loss) if loss is not None else None
        print_result_row(result, baseline_loss)

    summary = {
        "suite": args.suite,
        "steps": args.steps,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
        "eval_batches": args.eval_batches,
        "analysis_batches": args.analysis_batches,
        "output_dir": str(suite_dir),
        "elapsed_seconds": time.perf_counter() - suite_start,
        "results": results,
    }

    summary_json = suite_dir / "summary.json"
    summary_md = suite_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
