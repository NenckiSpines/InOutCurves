"""Simulation design, sample-size power/FPR, and method-agreement poster workflow.

Run the thin entry points in analysis/, not an old root-level script.
Shared inference: scripts.permutation_tests; calibration: scripts.simulation.
Quantitative caches/checkpoints go to dumps/; only plots and captions to figures/.
Both supplied population models are retained, explicitly distinguished in captions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import csv
from typing import Any
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .paths import ROOT,INPUT_DATA,DUMPS,FIGURES
from .cache import utc_now,sha256_file,jsonable,write_json,write_csv,project_code_hash
from .simulation import SourceData,Model,load_source,calibrate,simulate_experiment
from .distances import l2_squared,integration_cholesky
from .reference_tests import anova_group_effect
from .metrics import binomial_interval
from .permutation_tests import test_with_cholesky
from .plotting import configure_plots


def permutation_l2(a,b,chol,permutations,rng,exact_limit=50000):
    result=test_with_cholesky((a,b),chol,permutations=permutations,rng=rng,exact_limit=exact_limit)
    return result.statistic,result.pvalue,result.method,result.evaluated_permutations,result.total_allocations

# ============================================================================
# USER SETTINGS: change these, OR use the command-line flags shown in --help.
# ============================================================================
DEFAULT_MODE = "test"                   # Change to "final" when ready.
DEFAULT_MODEL = "within-group"          # Or "archive" (old generator, fixed test).
DEFAULT_SEED = 20260919
PRESETS = {
    "test": {
        "permutations": 199,             # B: small; test that the pipeline runs.
        "experiments": 30,               # R: NOT enough for scientific conclusions.
        "sample_sizes": [5, 10, 20],
        "dpi": 160,
    },
    "final": {
        "permutations": 9999,            # B per test (unless exact is cheaper).
        "experiments": 5000,             # R PER n PER null/alternative scenario.
        "sample_sizes": [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        "dpi": 300,
    },
}
DEFAULT_ALPHA = 0.05                     # Main poster comparison; not alpha=0.001.
DEFAULT_FORMATS = ["png", "pdf", "svg"]
DEFAULT_EXACT_LIMIT = 50000              # Safety cap on exact enumeration.
PERMUTATION_BATCH_SIZE = 512             # Memory control, not number of permutations.
EXPERIMENTS_PER_CHECKPOINT = 10
ENGINE_VERSION = "2.0.0-clean"
# ============================================================================

SCENARIOS = ("null", "alternative")
METHODS = (("p_l2", "Permutation, squared L2"), ("p_anova", "Mixed ANOVA, Group effect"))
RAW_FIELDS = [
    "scenario", "n", "experiment", "statistic_l2_squared", "p_l2", "f_anova",
    "p_anova", "permutation_mode", "permutations_evaluated", "total_allocations",
]


def make_rng(seed: int, n: int, scenario: str, replicate: int, stream: int) -> np.random.Generator:
    """Separate deterministic streams; identical data for every B / worker count."""
    return np.random.default_rng(np.random.SeedSequence([seed, n, SCENARIOS.index(scenario), replicate, stream]))


_WORKER_MODEL: Model | None = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_CHOL: np.ndarray | None = None


def initialize_worker(model: Model, config: dict[str, Any]) -> None:
    global _WORKER_MODEL, _WORKER_CONFIG, _WORKER_CHOL
    _WORKER_MODEL, _WORKER_CONFIG = model, config
    _WORKER_CHOL = integration_cholesky(model.x)


def analyze_chunk(task: tuple[int, str, list[int]]) -> list[dict[str, Any]]:
    n, scenario, replicates = task
    model, c, chol = _WORKER_MODEL, _WORKER_CONFIG, _WORKER_CHOL
    assert model is not None and c is not None and chol is not None
    output = []
    for r in replicates:
        a, b = simulate_experiment(model, n, scenario, make_rng(c["seed"], n, scenario, r, 0))
        observed, p, mode, draws, total = permutation_l2(
            a, b, chol, c["permutations"], make_rng(c["seed"], n, scenario, r, 1), c["exact_limit"])
        f_value, p_anova = anova_group_effect(a, b)
        if not np.isfinite([observed, p, p_anova]).all() or not 0 <= p_anova <= 1:
            raise ArithmeticError(f"Invalid test output for {scenario}, n={n}, experiment={r}.")
        output.append(dict(zip(RAW_FIELDS, [scenario, n, r, observed, p, f_value, p_anova, mode, draws, total])))
    return output


def load_checkpoint(path: Path, config: dict[str, Any]) -> dict[tuple[int, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    # An interruption can leave a partial final row. Remove ONLY that unfinished
    # suffix in this script's own output; completed rows are never discarded.
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        if cut < 0:
            raise ValueError("Checkpoint has no complete header. Choose a new --cache-dir directory.")
        path.write_bytes(data[:cut + 1])
        print("Discarded an interrupted, incomplete final checkpoint line; it will be recomputed.")
    saved = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RAW_FIELDS:
            raise ValueError("Unexpected raw_results.csv header; choose a new cache directory.")
        for row in reader:
            for name in ("n", "experiment", "permutations_evaluated", "total_allocations"):
                row[name] = int(row[name])
            for name in ("statistic_l2_squared", "p_l2", "f_anova", "p_anova"):
                row[name] = float(row[name])
            key = (row["n"], row["scenario"], row["experiment"])
            valid_key = (key[0] in config["sample_sizes"] and key[1] in SCENARIOS
                         and 0 <= key[2] < config["experiments"])
            if not valid_key or key in saved or not 0 < row["p_l2"] <= 1 or not 0 <= row["p_anova"] <= 1:
                raise ValueError(f"Invalid/duplicate checkpoint row {key}.")
            saved[key] = row
    return saved


def run_benchmark(model: Model, config: dict[str, Any], output: Path, jobs: int,
                  replot: bool) -> list[dict[str, Any]]:
    raw_path = output / "raw_results.csv"
    saved = load_checkpoint(raw_path, config)
    expected = 2 * len(config["sample_sizes"]) * config["experiments"]
    if replot and len(saved) != expected:
        raise ValueError(f"Replot requires {expected} completed experiments; found {len(saved)}. Run without --replot to resume.")
    if len(saved) == expected:
        print(f"Using all {expected:,} matching saved experiments (no new simulations).")
        return [saved[k] for k in sorted(saved)]
    print(f"Completed checkpoint: {len(saved):,}/{expected:,} experiments.")
    initialize_worker(model, config)
    executor = (ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"),
                                   initializer=initialize_worker, initargs=(model, config)) if jobs > 1 else None)
    started = time.perf_counter()
    newly_finished = 0
    try:
        with raw_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, lineterminator="\n")
            if raw_path.stat().st_size == 0:
                writer.writeheader()
                handle.flush()
            for n in config["sample_sizes"]:
                for scenario in SCENARIOS:
                    missing = [r for r in range(config["experiments"]) if (n, scenario, r) not in saved]
                    if not missing:
                        continue
                    tasks = [(n, scenario, missing[i:i + EXPERIMENTS_PER_CHECKPOINT])
                             for i in range(0, len(missing), EXPERIMENTS_PER_CHECKPOINT)]
                    iterator = executor.map(analyze_chunk, tasks, chunksize=1) if executor else map(analyze_chunk, tasks)
                    for step, rows in enumerate(iterator, 1):
                        writer.writerows(rows)
                        handle.flush()
                        for row in rows:
                            saved[(row["n"], row["scenario"], row["experiment"])] = row
                        newly_finished += len(rows)
                        if step == len(tasks) or step % max(1, len(tasks) // 10) == 0:
                            elapsed = time.perf_counter() - started
                            print(f"  n={n:>2} {scenario:>11}: {len(saved):,}/{expected:,} total; "
                                  f"{newly_finished / max(elapsed, 1e-6):.1f} experiments/s", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return [saved[k] for k in sorted(saved)]


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rates, agreement = [], []
    for n in config["sample_sizes"]:
        for scenario in SCENARIOS:
            block = [r for r in rows if r["n"] == n and r["scenario"] == scenario]
            if len(block) != config["experiments"]:
                raise ValueError(f"Incomplete experiment block: {scenario}, n={n}.")
            decisions = {}
            for key, label in METHODS:
                decisions[key] = np.asarray([r[key] <= config["alpha"] for r in block])
                hits, count = int(decisions[key].sum()), len(block)
                lo, hi = binomial_interval(hits, count)
                rate = hits / count
                rates.append({"n": n, "scenario": scenario, "method": label,
                              "metric": "power" if scenario == "alternative" else "false_positive_rate",
                              "alpha": config["alpha"], "experiments": count, "rejections": hits,
                              "estimate": rate, "ci_low": lo, "ci_high": hi,
                              "mcse": math.sqrt(rate * (1 - rate) / count)})
            p, a = decisions["p_l2"], decisions["p_anova"]
            agreement.append({"n": n, "scenario": scenario, "experiments": len(block),
                              "alpha": config["alpha"], "both": int((p & a).sum()),
                              "l2_only": int((p & ~a).sum()), "anova_only": int((~p & a).sum()),
                              "neither": int((~p & ~a).sum())})
    return rates, agreement


def low_precision(config: dict[str, Any]) -> bool:
    return (config["mode"] == "test" or config["experiments"] < 2000
            or config["permutations"] < 4999 or config["experiments"] * config["alpha"] < 100)


def add_header(fig: Any, title: str, config: dict[str, Any], info: str = "") -> None:
    fig.suptitle(title, x=0.5, y=0.985, fontsize=19, fontweight="bold")
    fig.text(0.5, 0.926, info, ha="center", va="top", fontsize=10.5)
    tag = ("TEST / LOW-PRECISION RUN - layout and execution check only" if low_precision(config)
           else f"{config['model']} model | independent simulated subjects | fixed empirical templates")
    fig.text(0.5, 0.873, tag, ha="center", va="top", fontsize=10, fontweight="bold")


def save_figure(fig: Any, output: Path, stem: str, args: argparse.Namespace) -> None:
    for extension in args.formats:
        fig.savefig(output / f"{stem}.{extension}", dpi=args.dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def plot_figure1(source: SourceData, model: Model, config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    """Figure 1 caption is produced by write_explanations(); no significance selection."""
    n = config["representative_n"]
    examples = {s: simulate_experiment(model, n, s, make_rng(config["seed"], n, s, 0, 90)) for s in SCENARIOS}
    display_source = source.raw if config["model"] == "archive" else source.subjects
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.745, wspace=0.22)
    add_header(fig, "Experimental templates and simulated groups", config,
               f"Source: {args.data.name} | simulated examples: n = {n} per group | {config['model']} calibration")
    titles = ["A  Experimental recordings", "B  Alternative: control vs LTP", "C  Null: same population"]
    labels = [f"Control ({source.labels[0]})", f"LTP ({source.labels[1]})"]
    panels = [display_source, examples["alternative"], examples["null"]]
    for j, (ax, arrays) in enumerate(zip(axes, panels)):
        ax.set_title(titles[j], loc="left", pad=12, fontweight="bold")
        for g, curves in enumerate(arrays):
            ax.plot(model.x, curves.T, color=colors[g], alpha=0.17, linewidth=0.75)
            name = labels[g] if j < 2 else f"Null group {g + 1}"
            ax.plot(model.x, curves.mean(axis=0), color=colors[g], linewidth=2.6,
                    label=f"{name}: mean (n={len(curves)})")
        if j == 1:
            for g in range(2):
                ax.plot(model.x, model.templates[g], color=colors[g], linewidth=1.7, linestyle="--")
        elif j == 2:
            ax.plot(model.x, model.null_template, color=colors[2], linewidth=1.8, linestyle="--", label="Shared true template")
        ax.set_xlabel(args.stimulation_label)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.13)
        ax.margins(x=0.025)
    axes[0].set_ylabel(args.response_label)
    fig.text(0.5, 0.025, "Thin lines: individual curves. Solid lines: sample means. Dashed lines: generating templates.",
             ha="center", fontsize=10)
    save_figure(fig, output, "poster_simulation_design", args)
    exported = []
    for scenario, groups in examples.items():
        for g, curves in enumerate(groups):
            for i, curve in enumerate(curves):
                for x, value in zip(model.x, curve):
                    exported.append({"scenario": scenario, "group": g + 1, "subject": i + 1,
                                     "stimulation": float(x), "response": float(value)})
    write_csv(args.cache_dir / "simulation_example_curves.csv", exported)


def plot_figure2(rates: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.745, wspace=0.29)
    add_header(fig, "Sample-size dependence", config,
               f"R = {config['experiments']:,} experiments per scenario and n | B = {config['permutations']:,} random allocations* | alpha = {config['alpha']:g}")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, scenario, title in zip(axes, ["alternative", "null"], ["A  Detection power", "B  False-positive rate"]):
        for j, (_, label) in enumerate(METHODS):
            records = [r for r in rates if r["scenario"] == scenario and r["method"] == label]
            n = np.asarray([r["n"] for r in records])
            mean = np.asarray([r["estimate"] for r in records])
            lo = np.asarray([r["ci_low"] for r in records])
            hi = np.asarray([r["ci_high"] for r in records])
            ax.fill_between(n, lo, hi, color=colors[j], alpha=0.12, linewidth=0)
            ax.errorbar(n, mean, yerr=[mean - lo, hi - mean], color=colors[j], linewidth=1.8,
                        marker=("o", "s")[j], markersize=5, linestyle=("-", "--")[j],
                        elinewidth=0.85, capsize=2.2, label=label)
        reference = 0.8 if scenario == "alternative" else config["alpha"]
        ax.axhline(reference, linestyle=":", linewidth=1.2, color=ax.spines["left"].get_edgecolor(),
                   label="80% power reference" if scenario == "alternative" else f"Nominal alpha = {config['alpha']:g}")
        ax.set_title(title, loc="left", pad=12, fontweight="bold")
        ax.set_xlabel("Independent subjects per group, n")
        ax.set_ylabel("Power" if scenario == "alternative" else "False-positive rate")
        ax.set_xticks(config["sample_sizes"])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        max_ci = max(r["ci_high"] for r in rates if r["scenario"] == scenario)
        ax.set_ylim(0, 1.025 if scenario == "alternative" else min(1.025, max(0.12, max_ci * 1.12, reference * 1.3)))
        ax.grid(axis="y", alpha=0.16)
        ax.legend(loc="best")
        ax.margins(x=0.045)
    fig.text(0.5, 0.025, "Pointwise exact 95% binomial intervals. *All allocations are enumerated when cheaper (see captions).",
             ha="center", fontsize=9.5)
    save_figure(fig, output, "poster_sample_size", args)


def plot_figure3(rows: list[dict[str, Any]], agreements: list[dict[str, Any]], config: dict[str, Any],
                 args: argparse.Namespace, output: Path) -> dict[str, Any]:
    n = config["representative_n"]
    selected = [r for r in rows if r["n"] == n]
    all_values = np.asarray([r[key] for r in selected for key in ("p_l2", "p_anova")])
    positive = all_values[all_values > 0]
    lower = max(1e-12, 10.0 ** math.floor(math.log10(min(float(positive.min()), config["alpha"] / 2))))
    clipped = int(np.count_nonzero(all_values < lower))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.745, wspace=0.28)
    add_header(fig, "Agreement between statistical methods", config,
               f"Prespecified n = {n} per group | R = {config['experiments']:,} experiments in each panel | alpha = {config['alpha']:g}")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        block = [r for r in selected if r["scenario"] == scenario]
        p_l2 = np.asarray([r["p_l2"] for r in block])
        p_anova = np.asarray([r["p_anova"] for r in block])
        ax.scatter(np.maximum(p_l2, lower), np.maximum(p_anova, lower), s=17 if len(block) < 300 else 8,
                   alpha=0.6 if len(block) < 300 else 0.22, color=colors[j], linewidths=0, rasterized=True)
        reference_color = ax.spines["left"].get_edgecolor()
        ax.plot([lower, 1], [lower, 1], linestyle=":", linewidth=1.1, color=reference_color, label="Equal p-values")
        ax.axvline(config["alpha"], linestyle="--", linewidth=1, color=reference_color)
        ax.axhline(config["alpha"], linestyle="--", linewidth=1, color=reference_color)
        ax.set(xscale="log", yscale="log", xlim=(lower / 1.2, 1.2), ylim=(lower / 1.2, 1.2))
        ax.set_xlabel("Permutation squared-L2 p-value")
        ax.set_title("A  Null experiments" if scenario == "null" else "B  Control-LTP experiments",
                     loc="left", fontweight="bold", pad=12)
        ax.grid(which="major", alpha=0.12)
        agreement = next(r for r in agreements if r["n"] == n and r["scenario"] == scenario)
        count = agreement["experiments"]
        text = (f"Both reject: {agreement['both']}/{count} | L2 only: {agreement['l2_only']}/{count}\n"
                f"ANOVA only: {agreement['anova_only']}/{count} | Neither: {agreement['neither']}/{count}")
        ax.text(0.5, -0.235, text, transform=ax.transAxes, ha="center", va="top", fontsize=10, linespacing=1.5)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Mixed ANOVA, Group-effect p-value")
    note = "Each point is one experiment analyzed by both methods. Dashed lines: significance threshold. No p-value jitter."
    if clipped:
        note += f"\n{clipped} plotted values below {lower:g} are shown at that floor; raw values are retained."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9.5)
    save_figure(fig, output, "poster_method_agreement", args)
    details = {"representative_n": n, "log_plot_lower_floor": lower,
               "clipped_values_for_display_only": clipped, "jitter": False, "plotted_experiments": len(selected)}
    write_json(args.cache_dir / "method_agreement_plot_details.json", details)
    return details


def export_calibration(source: SourceData, model: Model, output: Path) -> None:
    info = {"description": model.description, "source_group_labels": source.labels,
            "source_subject_counts": [len(y) for y in source.subjects], "source_recording_counts": source.row_counts,
            "ignored_source_groups": source.ignored_groups,
            "shared_source_identifiers": len(set(source.ids[0]) & set(source.ids[1])),
            "design": "independent synthetic subjects; not a paired-data analysis",
            "gain_location": model.gain_location, "gain_scale": model.gain_scale,
            "gain_mean": model.gain_location + model.gain_scale, "noise_sd": model.noise_sd,
            "recordings_per_simulated_subject": model.recordings_per_subject,
            "source_parameters": model.source_parameters}
    write_json(output / "calibration.json", info)
    np.savez_compressed(output / "calibration_arrays.npz", x=model.x, templates=model.templates,
                        null_template=model.null_template, empirical_gains=model.calibration_gains,
                        empirical_centered_residuals=model.calibration_residuals,
                        noise_sd=model.noise_sd, source_control=source.subjects[0], source_ltp=source.subjects[1])
    write_csv(output / "templates.csv", [{"stimulation": float(x), "control_template": float(c),
                                         "ltp_template": float(l), "null_template": float(z)}
                                        for x, c, l, z in zip(model.x, model.templates[0], model.templates[1], model.null_template)])


def write_explanations(source: SourceData, model: Model, config: dict[str, Any],
                       plot_details: dict[str, Any], output: Path) -> None:
    n = config["representative_n"]
    exact_n = [v for v in config["sample_sizes"] if math.comb(2 * v, v) <= min(config["permutations"], config["exact_limit"])]
    exact_text = ("Exact enumeration was used at " + ", ".join(f"n={v} ({math.comb(2*v,v):,} allocations)" for v in exact_n)
                  + "; Monte Carlo sampling was used at the remaining sample sizes.") if exact_n else "Monte Carlo sampling was used at all evaluated sample sizes."
    model_text = ("Variability was calibrated from each source subject's deviations from its own condition-specific mean, "
                  "then pooled across conditions. Each simulated subject contributed one generated curve.") if config["model"] == "within-group" else (
                  "The archive's fixed gain/noise constants and recording-weighted templates were retained. Three independently "
                  "distorted recordings were averaged per simulated subject, reproducing the old generator's variance reduction.")
    warning = ("TEST / LOW-PRECISION OUTPUT: intended to check execution and figure layout, not to support scientific claims.\n\n"
               if low_precision(config) else "")
    caption1 = ("Figure 1. Empirically informed simulation of input-output curves. "
                f"(A) Source recordings for control ({source.labels[0]}) and LTP ({source.labels[1]}), with "
                f"{len(source.subjects[0])} and {len(source.subjects[1])} source identifiers, respectively. "
                + ("Repeated recordings were averaged within subject and condition before calibration. " if config["model"] == "within-group" else
                   "Individual source recordings and recording-weighted means are displayed. ") +
                f"(B) One independently generated alternative experiment with n={n} subjects per group, using separate control "
                "and LTP templates. (C) One independently generated null experiment, in which both groups share a pooled "
                "template and the same variability distribution. Thin lines represent individual curves, solid lines their "
                "sample means, and dashed lines the generating templates. Null groups are independent samples, not duplicated "
                "curves. Examples were selected by a fixed seed, without screening for significance. " + model_text +
                " Original response signs and scaling are retained; confirm the biological response units before poster submission.")
    caption2 = ("Figure 2. Sample-size dependence of statistical power and false-positive rate. "
                "(A) Power: the proportion of control-LTP alternative experiments yielding a significant result. "
                "(B) False-positive rate: the corresponding proportion of common-population null experiments. "
                f"For every n, R={config['experiments']:,} independent experiments were generated under each scenario, and both "
                f"methods were applied to the same datasets at alpha={config['alpha']:g}. Lines/markers show rejection "
                "proportions; error bars and shading show pointwise exact 95% binomial confidence intervals. "
                "Reference lines indicate 80% power and the nominal significance threshold. "
                f"The permutation test used B={config['permutations']:,} random allocations per experiment unless the full "
                "allocation space was smaller. " + exact_text + " The permutation statistic is the squared L2 distance "
                "between piecewise-linear group-mean curves. The ANOVA comparator is the mixed-design Group main effect, "
                "not the Group x Stimulation interaction. Templates and variability parameters were held fixed while n varied. "
                "Intervals quantify simulation uncertainty, not uncertainty in the empirically estimated templates.")
    caption3 = ("Figure 3. Agreement between permutation testing and the mixed-ANOVA Group effect. "
                f"Each point represents one of R={config['experiments']:,} experiments with n={n} independent subjects per group, "
                "analyzed by both methods. (A) Null experiments. (B) Control-LTP alternative experiments. "
                "Both axes are logarithmic; the dotted diagonal indicates equal p-values, and dashed lines mark "
                f"alpha={config['alpha']:g}. Counts below each panel report joint and method-specific rejections. "
                "All simulated experiments at the prespecified sample size are included; overlapping p-values are not jittered. "
                "Differences between p-values alone do not establish method superiority; interpret them with the power and "
                "false-positive results in Figure 2 and the hypotheses targeted by the tests.")
    if plot_details["clipped_values_for_display_only"]:
        caption3 += (f" For display only, {plot_details['clipped_values_for_display_only']} values below "
                     f"{plot_details['log_plot_lower_floor']:g} were placed at that floor; raw values are unchanged.")
    for i,c in enumerate([caption1,caption2,caption3],1):
        if i in config["figures"]:
            stem={1:"poster_simulation_design",2:"poster_sample_size",3:"poster_method_agreement"}[i]
            (output/f"{stem}_caption.txt").write_text(warning+c+"\n",encoding="utf-8")
    (output / "captions.txt").write_text(warning + "\n\n".join(c for i,c in enumerate([caption1,caption2,caption3],1) if i in config["figures"]) + "\n", encoding="utf-8")
    short_model = ("One curve was generated per independent subject after within-condition calibration."
                   if config["model"] == "within-group" else
                   "The archive model averages three independently generated recordings per subject.")
    short_captions = [
        f"Figure 1. Simulation design. (A) Experimental control ({source.labels[0]}) and LTP "
        f"({source.labels[1]}) curves. (B) An alternative experiment generated around separate "
        f"control/LTP templates. (C) A null experiment generated from one common template and "
        f"variability model, with independent draws. Examples use n={n} per group. Thin lines: "
        "individual curves; solid lines: sample means; dashed lines: generating templates. " + short_model,
        "Figure 2. Power and false-positive control versus sample size. (A) Rejection rates under "
        "the control-LTP alternative. (B) Rejection rates under the common-population null. "
        f"Both methods analyze the same R={config['experiments']:,} experiments per scenario and n; "
        f"alpha={config['alpha']:g}. Error bars/shading: pointwise exact 95% binomial intervals. "
        f"The permutation test uses B={config['permutations']:,} allocations or exact enumeration "
        "when cheaper. ANOVA denotes the Group main effect. Reference lines mark 80% power and nominal alpha.",
        "Figure 3. Method agreement. Each point is one experiment analyzed by both tests at "
        f"prespecified n={n}; R={config['experiments']:,} per scenario. (A) Null. (B) Control-LTP "
        "alternative. Axes are logarithmic. The dotted line indicates equal p-values; dashed "
        f"lines mark alpha={config['alpha']:g}. Counts summarize joint and method-specific rejections. "
        "All experiments are included. Interpret disagreement alongside power and false-positive control."
    ]
    (output / "poster_captions.txt").write_text(warning + "\n\n".join(c for i,c in enumerate(short_captions,1) if i in config["figures"]) + "\n", encoding="utf-8")
    explanation = warning + f"""METHODS SUBSECTION
Empirically Calibrated Simulations for Sample-Size-Dependent Method Comparison

{model.description}
{model_text}

At stimulation x_k, an individual generated recording is
    Y_gi(x_k) = A_gi * mu_g(x_k) + epsilon_gik.
A = location + Exponential(scale), with location={model.gain_location:.10g},
scale={model.gain_scale:.10g}, and mean={model.gain_location + model.gain_scale:.10g}.
The same gain distribution and stimulus-specific noise SDs are used in both groups.
Noise is Gaussian and independent across stimulation points, conditional on gain.
The gain induces a shared multiplicative component within each curve. The model
does not reproduce every empirical residual correlation or gain-residual dependence.
No outliers are removed, no sign inversion or clipping is applied, and no cubic
residual detrending is used in this new script.

Under H1, mu_g is the group's experimental template. Under H0, both groups use
one pooled template and identical variability distributions, with independent draws.
The original data may be paired; they supply templates here. The power experiment
itself concerns INDEPENDENT groups. n means independent simulated experimental
units, not stimulation points, repeated recordings, or necessarily literal subjects.

SETTINGS
n per group: {config['sample_sizes']}
R independent experiments per scenario per n: {config['experiments']:,}
B random allocations within one test: {config['permutations']:,}
Alpha: {config['alpha']:g}
Random seed: {config['seed']}
Generated recordings averaged per subject: {model.recordings_per_subject}
{exact_text}

STATISTICS
T = integral [mean_control(x) - mean_LTP(x)]^2 dx.
For each interval of width h, with endpoint differences d0 and d1,
the exact contribution for linear interpolants is h*(d0^2+d0*d1+d1^2)/3.
This is the SQUARED L2 distance; there is no square root. Complete subject-level
curves are exchanged between groups, preserving group sizes and within-curve data.
Monte Carlo p = (1 + count(T_perm >= T_observed))/(B+1). Exact tests use the
inclusive tail count / total allocations, including the observed allocation.
A relative tolerance of 100 machine epsilons handles floating-point ties.
The nondirectional distance uses its upper tail; its p-value is not doubled.

The ANOVA comparator matches the Group main effect from scripts/reference_tests.py.
With k complete repeated stimulation levels, both SS_Group and SS_Subject(Group)
contain a factor k, which cancels in F. Thus it is computed as the equal-variance
one-way F test of subject-wise arithmetic means across stimulation points.
Stimulation points are not treated as independent biological observations.
This is NOT a Group x Stimulation test or a combined curve-equality omnibus test.

Power = number of H1 p-values <= alpha / R.
False-positive rate = number of H0 p-values <= alpha / R.
False-negative rate = 1 - power.
Pointwise 95% exact Clopper-Pearson binomial intervals accompany the rates.
Monte Carlo standard errors are also saved in summary_rates.csv. For zero/complete
rejection the plug-in MCSE is zero; the nonzero-width exact interval remains the
appropriate uncertainty display. Both tests share each experiment's generated data.

RESULTS SUBSECTION
Sample-Size Dependence of Statistical Power and Type I Error

INTERPRETATION
Compare detection rates together with false-positive control. A larger rejection
rate is not automatically better. The comparison is conditional on this particular
control-LTP difference, variability model and independent-group design. The original
mean curves/variability are estimated, but template-estimation uncertainty is not
included in these intervals. Sample-size curves do not give universal n requirements.
Nonparametric does not mean assumption-free: null exchangeability is required.
Do not infer general outlier robustness, paired-design validity, patch-clamp
validation, or universal superiority from these figures.

REPRODUCIBILITY
The benchmark does not read historical p-value caches or call
its old get_p_value function. All results are newly generated or resumed from this
script's own fingerprint-matched raw_results.csv. Seeds, checksums, versions and
parameters are in run_manifest.json. Figure 1 uses separate fixed random streams.
Figure 3 includes every experiment at prespecified n={n}; no significance-based
selection. Explicit units must be supplied after checking the experimental metadata.

REFERENCES
Phipson B, Smyth GK (2010). DOI: 10.2202/1544-6115.1585.
Morris TP, White IR, Crowther MJ (2019). DOI: 10.1002/sim.8086.
SciPy permutation_test documentation; Pingouin mixed_anova documentation.
See the script docstring for the official documentation URLs.
"""
    (output / "methods_and_interpretation.txt").write_text(explanation, encoding="utf-8")


def self_test() -> None:
    """Fast invariants independent of the source archive and of Pingouin."""
    rng = np.random.default_rng(9182)
    x = np.array([0.0, 0.7, 2.0, 4.2])
    chol = integration_cholesky(x)
    for _ in range(30):
        d = rng.normal(size=4)
        np.testing.assert_allclose(np.sum((d @ chol) ** 2), l2_squared(d, np.zeros(4), x), rtol=2e-14)
    np.testing.assert_allclose(l2_squared(np.array([3., 3.]), np.zeros(2), np.array([0., 1.])), 9.)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(3, 4)) + 0.5
    observed, p, mode, draws, total = permutation_l2(a, b, chol, 99, rng)
    pooled = np.concatenate([a, b])
    manual = []
    for indices in itertools.combinations(range(6), 3):
        take = np.array(indices)
        other = np.array([i for i in range(6) if i not in indices])
        manual.append(l2_squared(pooled[take].mean(0), pooled[other].mean(0), x))
    wanted = np.count_nonzero(np.asarray(manual) >= observed - abs(observed) * 1e-12) / len(manual)
    np.testing.assert_allclose(p, wanted)
    assert mode == "exact" and draws == total == 20
    zeros = np.zeros((3, 4))
    assert permutation_l2(zeros, zeros, chol, 99, rng)[1] == 1.0
    assert permutation_l2(zeros, zeros, chol, 9, rng)[1] == 1.0
    assert 0 < permutation_l2(a, b, chol, 9, rng)[1] <= 1
    f, pa = anova_group_effect(a, b)
    reference = stats.f_oneway(a.mean(1), b.mean(1))
    np.testing.assert_allclose([f, pa], [reference.statistic, reference.pvalue], rtol=2e-12)
    t = stats.ttest_ind(a.mean(1), b.mean(1), equal_var=True)
    np.testing.assert_allclose(f, t.statistic ** 2, rtol=2e-12)
    np.testing.assert_allclose(pa, t.pvalue, rtol=2e-12)
    assert anova_group_effect(zeros, zeros) == (0.0, 1.0)
    assert binomial_interval(0, 25)[0] == 0 and binomial_interval(25, 25)[1] == 1
    print("Self-tests passed: squared-L2 integral, exact tail, ties, Monte Carlo range, ANOVA F/t equivalence, binomial intervals.")


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("SCIENTIFIC DESIGN")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=PRESETS, default=DEFAULT_MODE)
    parser.add_argument("--model", choices=["within-group", "archive"], default=DEFAULT_MODEL)
    parser.add_argument("--project-dir", type=Path, default=ROOT,
                        help="Clean project root; detected from the package location.")
    parser.add_argument("--data", type=Path, default=None, help="Source CSV, absolute or relative to the project root.")
    parser.add_argument("--cache-dir", type=Path, help="Quantitative checkpoints under dumps/validation by default.")
    parser.add_argument("--regenerate", action="store_true", help="Restart this managed benchmark; source inputs are never changed.")
    parser.add_argument("--figures", nargs="+", type=int, choices=[1,2,3], default=[1,2,3])
    parser.add_argument("--out", type=Path, default=None, help="Figure/caption folder; numerical caches use --cache-dir.")
    parser.add_argument("--permutations", type=int, default=None, help="B random allocations per permutation test.")
    parser.add_argument("--experiments", type=int, default=None, help="R independent experiments per scenario and n.")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None, help="Subjects per group, e.g. 5 10 20.")
    parser.add_argument("--representative-n", type=int, default=None, help="Prespecified n for Figure 3 (and Figure 1 examples).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes; default 1. Optional: --jobs 4.")
    parser.add_argument("--exact-limit", type=int, default=DEFAULT_EXACT_LIMIT)
    parser.add_argument("--control-group", default="PRE")
    parser.add_argument("--ltp-group", default="POST")
    parser.add_argument("--response-label", default="fEPSP slope (source units)", help="Replace only after confirming biological units.")
    parser.add_argument("--stimulation-label", default="Stimulation current (\u00b5A)")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--formats", choices=["png", "pdf", "svg"], nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument("--replot", action="store_true", help="Only redraw a complete matching run; never simulate.")
    parser.add_argument("--self-test", action="store_true", help="Run numerical checks and exit (no source data needed).")
    args = parser.parse_args(argv)
    if args.replot and args.regenerate:
        parser.error("--replot and --regenerate cannot be combined.")
    args.figures=sorted(set(args.figures))
    preset = PRESETS[args.mode]
    for name in ("permutations", "experiments", "sample_sizes", "dpi"):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])
    args.sample_sizes = sorted(set(args.sample_sizes))
    if not args.sample_sizes or min(args.sample_sizes) < 2:
        parser.error("Every sample size must be >=2 subjects per group.")
    if args.representative_n is None:
        args.representative_n = min(args.sample_sizes, key=lambda v: abs(v - 10))
    if args.representative_n not in args.sample_sizes:
        parser.error("--representative-n must be in --sample-sizes.")
    if args.permutations < 1 or args.experiments < 2 or args.jobs < 1 or args.exact_limit < 0 or args.seed < 0:
        parser.error("Require B>=1, R>=2, jobs>=1, exact-limit>=0, seed>=0.")
    if not 0 < args.alpha < 1 or args.dpi < 72:
        parser.error("Require 0 < alpha < 1 and dpi >=72.")
    if args.control_group == args.ltp_group:
        parser.error("Control and LTP group labels must differ.")
    return args


def main(argv=None) -> None:
    args = parse_arguments(argv)
    if args.self_test:
        self_test()
        return
    project = args.project_dir.expanduser().resolve()
    if args.data is None:
        candidates = [project / "input_data" / "ltp_paired.csv"]
        args.data = next((p for p in candidates if p.exists()), candidates[0])
    else:
        args.data = args.data.expanduser()
        if not args.data.is_absolute():
            args.data = project / args.data
    args.data = args.data.resolve()
    if not args.data.is_file():
        raise FileNotFoundError(f"Source CSV not found: {args.data}\nUse the supplied input_data files, "
                                "or supply --project-dir /path/to/InOutCurves-main.")
    output = args.out.expanduser() if args.out else project / "figures" / "validation" / f"{args.mode}_{args.model}"
    if not output.is_absolute():
        output = project / output
    output = output.resolve()
    if output == project or output == args.data.parent:
        raise ValueError("Choose a dedicated output SUBFOLDER, not the project/data folder itself.")
    config = {k: getattr(args, k) for k in ["mode", "model", "permutations", "experiments", "sample_sizes",
                                          "representative_n", "alpha", "seed", "exact_limit", "control_group", "ltp_group"]}
    source = load_source(args.data, args.control_group, args.ltp_group)
    model = calibrate(source, args.model, project)
    scientific_identity = {"engine_version": ENGINE_VERSION, "script_sha256": sha256_file(Path(__file__)), "shared_code_sha256": project_code_hash(),
                           "source_sha256": sha256_file(args.data), "config": config,
                           "model_parameters": model.source_parameters,
                           "numpy_version": np.__version__, "scipy_version": scipy.__version__}
    fingerprint = hashlib.sha256(json.dumps(scientific_identity, sort_keys=True, default=jsonable).encode()).hexdigest()
    args.cache_dir = (args.cache_dir or project / "dumps" / "validation" / f"{args.mode}_{args.model}").expanduser()
    if not args.cache_dir.is_absolute():
        args.cache_dir = project / args.cache_dir
    args.cache_dir = args.cache_dir.resolve()
    if args.cache_dir in (project, project/"input_data", args.data.parent):
        raise ValueError("Choose a dedicated cache subfolder, not a source-data folder.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "run_manifest.json"
    if args.regenerate and any(i in args.figures for i in (2,3)):
        if manifest_path.exists():
            previous=json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("workflow")!="poster_validation":
                raise ValueError("Refusing to overwrite an unrelated run manifest; choose a new cache folder.")
        for name in ("run_manifest.json", "raw_results.csv"):
            (args.cache_dir/name).unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("This output folder contains a run with different settings, code, data, or numerical libraries. "
                             "Use --regenerate or a NEW --cache-dir; existing results were not changed.")
    else:
        if args.replot:
            raise ValueError("No run manifest found. Run simulations before using --replot.")
        if args.mode == "final" and any(i in args.figures for i in (2,3)) and not args.regenerate:
            raise ValueError("No matching final benchmark cache. Start explicitly with --regenerate --mode final.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"workflow": "poster_validation", "fingerprint": fingerprint, "identity": scientific_identity, "created_utc": utc_now(),
                    "source_file": str(args.data), "project_directory": str(project),
                    "python_version": sys.version, "platform": platform.platform(),
                    "matplotlib_version": matplotlib.__version__, "complete": False}
        write_json(manifest_path, manifest)
    print(f"\nSource: {args.data}\nOutput: {output}\nModel: {model.description}")
    print(f"n = {config['sample_sizes']}; R = {args.experiments:,} per scenario/n; B = {args.permutations:,}; jobs = {args.jobs}")
    if low_precision(config):
        print("TEST / LOW-PRECISION: figures will be labeled as execution checks, not final evidence.")
    if args.experiments * args.alpha < 100:
        print("CAUTION: few expected null rejections at this R and alpha; inspect the binomial intervals.")
    if source.ignored_groups:
        print(f"Only requested groups used; ignored: {source.ignored_groups}")
    export_calibration(source, model, args.cache_dir)
    rows=[]
    configure_plots()
    if any(i in args.figures for i in (2,3)):
        if args.mode == "final" and not args.regenerate and not (args.cache_dir/"raw_results.csv").exists():
            raise ValueError("Final simulation cache is absent. Start explicitly with --regenerate --mode final.")
        rows = run_benchmark(model, config, args.cache_dir, args.jobs, args.replot)
        rates, agreements = summarize(rows, config)
        write_csv(args.cache_dir / "summary_rates.csv", rates)
        write_csv(args.cache_dir / "method_agreement.csv", agreements)
    if 1 in args.figures:
        plot_figure1(source, model, config, args, output)
    if 2 in args.figures:
        plot_figure2(rates, config, args, output)
    details = {"clipped_values_for_display_only": 0}
    if 3 in args.figures:
        details = plot_figure3(rows, agreements, config, args, output)
    write_explanations(source, model, {**config,"figures":args.figures}, details, output)
    manifest.update({"complete": bool(rows) or manifest.get("complete",False), "last_completed_utc": utc_now(), "completed_experiments": len(rows) or manifest.get("completed_experiments",0),
                     "last_plot_settings": {"dpi": args.dpi, "formats": args.formats,
                                            "response_label": args.response_label, "stimulation_label": args.stimulation_label},
                     "last_jobs": args.jobs})
    write_json(manifest_path, manifest)
    print(f"\nDone. Figures and captions: {output}\nNumerical caches: {args.cache_dir}")
    if low_precision(config):
        print("These are TEST/LOW-PRECISION figures. Use the final preset without reduced R/B for the main analysis.")


if __name__ == "__main__":
    mp.freeze_support()  # Needed for multiprocessing on Windows/frozen executables.
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed checkpoints are retained; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, PermissionError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
