"""Experimental and notebook-derived poster figures, reorganized without changing layouts.

Use analysis/poster_curve_randomization.py, poster_permutation_precision.py,
poster_pvalue_distributions.py, or poster_notebook_suite.py. Managed numeric
pickles live under dumps/notebook/. The default reuses matching saved values;
--regenerate explicitly starts fresh. Historical zero-valued p-values may only
be shown with --legacy-pickles and are never repaired or labelled as a new run.
The archival simulation model (three recordings per subject) is retained here;
the validation workflow has a separate within-group calibrated default model.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
from typing import Any
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .paths import ROOT,INPUT_DATA,DUMPS,FIGURES
from .cache import (utc_now as now, sha256_file as sha256, jsonable as primitive,
                    write_json,write_csv as csv_out, NumericUnpickler,numeric_pickle,
                    fingerprint,save_cache,read_cache,project_code_hash)
from .simulation import SourceData as Curves,load_csv,archive_parameters,simulated_recordings
from .distances import l2_squared,integration_cholesky
from .permutation_tests import (independent_distribution,test_with_cholesky,tail_threshold)
from .metrics import binomial_interval as binomial_ci
from .plotting import configure_plots,COLORS


def allocation_statistics(a,b,chol,draws,rng,exact=False):
    return independent_distribution((a,b),chol,draws,rng,exact)


def permutation_test(a,b,chol,draws,rng):
    result=test_with_cholesky((a,b),chol,permutations=draws,rng=rng)
    return result.statistic,result.pvalue,result.method,result.evaluated_permutations

# ============================================================================
# USER SETTINGS -- CLI options override these; default reuses compatible caches.
# ============================================================================
USE_PICKLE_FILES = True                  # False = regenerate, including simulations.
DEFAULT_MODE = "test"                 # "test", "notebook", or "final".
DEFAULT_SEED = 20260919
DEFAULT_SUBJECTS_PER_GROUP = 6             # n, independent artificial subjects.
DEFAULT_RECORDINGS_PER_SUBJECT = 3         # m; kept to match the notebook generator.
DEFAULT_FIGURES = [1, 2, 3, 4]
DEFAULT_FORMATS = ["png", "pdf", "svg"]
DEFAULT_EXPERIMENTAL_GROUPS = ["ANH", "RES"]
DEFAULT_SIMULATION_GROUPS = ["PRE", "POST"]
EXPERIMENTAL_RESPONSE_LABEL = "Response (source units)"
SIMULATION_RESPONSE_LABEL = "fEPSP slope (source units)"
STIMULATION_LABEL = "Stimulation current (\u00b5A)"
PRESETS = {
    "test": {
        "permutations": 199,              # B per newly simulated experiment.
        "experiments": 30,                # R PER null/alternative scenario: smoke test.
        "histogram_permutations": 199,    # Figure 2A, fixed real dataset.
        "stability_counts": [49, 199, 999],
        "stability_repeats": 20,          # Re-estimates for each B in Figure 2B.
        "dpi": 160,
    },
    "notebook": {                         # The settings currently in the notebook.
        "permutations": 2000,
        "experiments": 10000,
        "histogram_permutations": 10000,
        "stability_counts": [250, 1000, 5000],
        "stability_repeats": 100,
        "dpi": 300,
    },
    "final": {
        "permutations": 9999,
        "experiments": 5000,
        "histogram_permutations": 9999,
        "stability_counts": [250, 1000, 5000],
        "stability_repeats": 100,
        "dpi": 300,
    },
}
EXACT_LIMIT = 50000
BATCH_SIZE = 512
CHECKPOINT_EVERY = 100
ENGINE_VERSION = "2.0.0-clean"
LEGACY_FILENAMES = ("p_values_null.pkl", "p_values_alternative.pkl")
STEMS = {1: "poster_curve_randomization", 2: "poster_permutation_precision",
         3: "poster_simulated_recordings", 4: "poster_pvalue_distributions"}
# ============================================================================


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *stream]))


def source_spec(source: Curves, kind: str) -> dict[str, Any]:
    return {"engine": ENGINE_VERSION, "shared_code_sha256": project_code_hash(), "numpy": np.__version__, "scipy": scipy.__version__, "kind": kind, "data_sha256": sha256(source.path), "groups": list(source.groups)}


def experimental_example(source: Curves, args: argparse.Namespace) -> dict[str, Any]:
    spec = {**source_spec(source, "experimental_example"), "seed": args.seed}
    path = args.cache_dir / "experimental_example.pkl"
    cached = read_cache(path, spec, args.regenerate)
    if cached:
        print(f"[cache] {path.name}")
        return cached["data"]
    a, b = source.subjects
    idx = rng_for(args.seed, 11).permutation(len(a)+len(b))
    data = {"assignment": idx.tolist(), "subject_counts": [len(a), len(b)], "created_utc": now()}
    save_cache(path, spec, data)
    print(f"[new] Single seeded reassignment -> {path.name}")
    return data


def inference_data(source: Curves, args: argparse.Namespace) -> dict[str, Any]:
    spec = {**source_spec(source, "inference"), "seed": args.seed,
            "histogram_permutations": args.histogram_permutations,
            "stability_counts": args.stability_counts, "stability_repeats": args.stability_repeats}
    path = args.cache_dir / "permutation_inference.pkl"
    cached = read_cache(path, spec, args.regenerate)
    if cached:
        print(f"[cache] {path.name}")
        return cached["data"]
    a, b = source.subjects
    chol = integration_cholesky(source.x)
    total = math.comb(len(a)+len(b), len(a))
    complete = None
    if total <= EXACT_LIMIT:
        observed, complete = allocation_statistics(a, b, chol, 1, rng_for(args.seed, 20), True)
        exact_p = float(np.mean(complete >= tail_threshold(observed)))
    else:
        observed = l2_squared(a.mean(axis=0), b.mean(axis=0), source.x)
        exact_p = None
    histogram_rng = rng_for(args.seed, 21)
    histogram = (complete[histogram_rng.integers(len(complete), size=args.histogram_permutations)]
                 if complete is not None else allocation_statistics(a, b, chol, args.histogram_permutations, histogram_rng)[1])
    observed_for_test = observed
    threshold = tail_threshold(observed_for_test)
    p_mc = (1+int(np.count_nonzero(histogram >= threshold)))/(len(histogram)+1)
    series = []
    for count in args.stability_counts:
        values = []
        for repeat in range(args.stability_repeats):
            rng = rng_for(args.seed, 22, count, repeat)
            draws = (complete[rng.integers(len(complete), size=count)] if complete is not None else
                     allocation_statistics(a, b, chol, count, rng)[1])
            values.append(float((1+np.count_nonzero(draws >= threshold))/(count+1)))
        series.append(values)
    data = {"observed_statistic": observed, "histogram_statistics": histogram.tolist(), "histogram_p": float(p_mc),
            "exact_p": exact_p, "total_allocations": total, "exact_statistics": None if complete is None else complete.tolist(),
            "stability_counts": args.stability_counts, "stability_pvalues": series,
            "stability_repeats": args.stability_repeats, "created_utc": now(),
            "pvalue_formula": "(1 + inclusive exceedances)/(B + 1); exact: inclusive count/all allocations"}
    save_cache(path, spec, data)
    print(f"[new] Inference -> {path.name}; L2={observed:.6g}, p_MC={p_mc:.6g}, p_exact={exact_p}")
    return data


def example_simulations(source: Curves, parameters: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spec = {**source_spec(source, "simulation_examples"), "parameters": parameters, "seed": args.seed,
            "subjects": args.subjects, "recordings": args.recordings}
    path = args.cache_dir / "simulation_examples.pkl"
    cached = read_cache(path, spec, args.regenerate)
    if cached:
        print(f"[cache] {path.name}")
        return cached["data"]
    data = {name: simulated_recordings(source, parameters, args.subjects, args.recordings,
                                      name == "alternative", rng_for(args.seed, 30, j)).tolist()
            for j, name in enumerate(("null", "alternative"))}
    data.update({"created_utc": now(), "subjects_per_group": args.subjects, "recordings_per_subject": args.recordings})
    save_cache(path, spec, data)
    print(f"[new] Simulation examples -> {path.name}")
    return data


def validate_pvalues(values: Any, label: str) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError(f"{label} must be a nonempty, finite 1D p-value sequence in [0,1].")
    return a


def legacy_pvalues(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.project / "dumps" / "legacy" / name for name in LEGACY_FILENAMES]
    if not all(p.exists() for p in paths):
        missing = ", ".join(str(p) for p in paths if not p.exists())
        raise FileNotFoundError(f"Missing notebook p-value pickle(s): {missing}. "
                                "No expensive run was started. Use --regenerate --mode test, then --mode final when ready.")
    data = {"origin": "legacy_notebook_pickles", "permutations": None, "subjects": None, "recordings": None,
            "generation_settings_verified": False, "note": "Stored p-values unchanged; no generating-run manifest."}
    for scenario, path in zip(("null", "alternative"), paths):
        values = validate_pvalues(numeric_pickle(path), str(path))
        data[scenario] = values.tolist()
        data[scenario + "_source"] = {"path": str(path), "sha256": sha256(path), "count": len(values),
                                       "zero_count": int(np.count_nonzero(values == 0))}
    print("[legacy] Loaded original p-value pickles UNCHANGED; B/n/seed unknown, not today's settings.")
    print(f"         R_null={len(data['null'])}, R_alternative={len(data['alternative'])}; "
          f"zeros={data['null_source']['zero_count']}/{data['alternative_source']['zero_count']}")
    return data


def pvalue_data(source: Curves, parameters: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spec = {**source_spec(source, "simulated_pvalues"), "parameters": parameters, "seed": args.seed,
            "subjects": args.subjects, "recordings": args.recordings, "experiments": args.experiments,
            "permutations": args.permutations}
    path = args.cache_dir / "simulation_pvalues.pkl"
    if args.legacy_pickles:
        return legacy_pvalues(args)
    cached = read_cache(path, spec, args.regenerate)
    if cached is None and not args.regenerate:
        raise FileNotFoundError("No compatible managed p-value cache. Use --regenerate --mode test (or final). Historical values require --legacy-pickles explicitly.")
    if cached:
        data = cached["data"]
        for scenario in ("null", "alternative"):
            if data[scenario]:
                validate_pvalues(data[scenario], str(path))
            if len(data[scenario]) != len(data[scenario + "_statistics"]):
                raise ValueError("Inconsistent partial p-value cache; use --regenerate.")
        if cached["complete"]:
            if any(len(data[s]) != args.experiments for s in ("null", "alternative")):
                raise ValueError("Complete cache does not match requested experiment count.")
            print(f"[cache] {path.name}: corrected p-values, R={args.experiments} per scenario")
            return data
        print(f"[resume] Partial simulation cache: null={len(data['null'])}, alternative={len(data['alternative'])}")
    else:
        allocations = math.comb(2*args.subjects, args.subjects)
        exact = allocations <= min(args.permutations, EXACT_LIMIT)
        data = {"origin": "regenerated_archive_model", "null": [], "alternative": [], "null_statistics": [],
                "alternative_statistics": [], "permutations": args.permutations, "subjects": args.subjects,
                "recordings": args.recordings, "seed": args.seed, "experiments_per_scenario": args.experiments,
                "permutation_mode": "exact" if exact else "monte_carlo",
                "allocations_evaluated_per_test": allocations if exact else args.permutations,
                "generation_settings_verified": True, "created_utc": now(), "model_parameters": parameters}
        save_cache(path, spec, data, complete=False)
    chol = integration_cholesky(source.x)
    print(f"[simulate] n={args.subjects}, m={args.recordings}, R={args.experiments}/scenario, "
          f"{data['permutation_mode']}: {data['allocations_evaluated_per_test']} allocations/test")
    for si, scenario in enumerate(("null", "alternative")):
        for r in range(len(data[scenario]), args.experiments):
            raw = simulated_recordings(source, parameters, args.subjects, args.recordings,
                                       scenario == "alternative", rng_for(args.seed, 40, si, r))
            a, b = raw.mean(axis=2)
            t, p, _, _ = permutation_test(a, b, chol, args.permutations, rng_for(args.seed, 41, si, r))
            data[scenario].append(p)
            data[scenario + "_statistics"].append(t)
            if (r+1) % CHECKPOINT_EVERY == 0 or r+1 == args.experiments:
                save_cache(path, spec, data, complete=False)
                print(f"  {scenario}: {r+1}/{args.experiments}", flush=True)
    save_cache(path, spec, data, complete=True)
    return data


# ---------------------------------------------------------------------------
# PLOTTING -- same typography, palette, headings and export style as poster_figures.
# ---------------------------------------------------------------------------


def header(fig: Any, title: str, info: str, status: str, args: argparse.Namespace) -> None:
    fig.suptitle(title, x=0.5, y=0.985, fontsize=19*args.font_scale, fontweight="bold")
    fig.text(0.5, 0.926, info, ha="center", va="top", fontsize=10.5*args.font_scale)
    fig.text(0.5, 0.873, status, ha="center", va="top", fontsize=10*args.font_scale, fontweight="bold")


def panel(ax: Any, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.13)
    ax.set_axisbelow(True)


def footer(fig: Any, text: str, args: argparse.Namespace) -> None:
    fig.text(0.5, 0.022, text, ha="center", va="bottom", fontsize=9.5*args.font_scale, linespacing=1.5)


def save_figure(fig: Any, number: int, args: argparse.Namespace) -> None:
    for fmt in args.formats:
        fig.savefig(args.out / f"{STEMS[number]}.{fmt}", dpi=args.dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"[figure {number}] {STEMS[number]} ({', '.join(args.formats)})")


def curves_long(x: np.ndarray, groups: list[np.ndarray] | tuple[np.ndarray, ...], labels: list[str] | tuple[str, ...],
                scenario: str) -> list[dict[str, Any]]:
    return [{"scenario": scenario, "group": label, "curve": i+1, "stimulation": float(v), "response": float(y)}
            for label, curves in zip(labels, groups) for i, curve in enumerate(curves) for v, y in zip(x, curve)]


def figure1(source: Curves, data: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Notebook cell 3: 2x2 original/shuffled subject curves above mean\u00b1SEM.

    The shuffled panel is descriptive. No claim of a tested p-value for that
    surrogate assignment is made; figure 2 tests the original grouping only.
    """
    a, b = source.subjects
    pooled = np.concatenate([a, b])[np.asarray(data["assignment"], dtype=int)]
    shuffled = (pooled[:len(a)], pooled[len(a):])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.15, top=0.77, wspace=0.25, hspace=0.40)
    header(fig, "Curve-wise randomization of experimental data",
           f"Source: {source.path.name} | {source.groups[0]}: n = {len(a)} | {source.groups[1]}: n = {len(b)}",
           "Complete subject-level curves are reassigned; group sizes are preserved", args)
    names = (source.groups, ("Surrogate group 1", "Surrogate group 2"))
    for col, (arrays, labels) in enumerate(zip((source.subjects, shuffled), names)):
        for g, (curves, label) in enumerate(zip(arrays, labels)):
            for i, curve in enumerate(curves):
                axes[0, col].plot(source.x, curve, color=COLORS[g], alpha=0.58, linewidth=1.0,
                                  label=f"{label} (n={len(curves)})" if i == 0 else None)
            mean, sem = curves.mean(axis=0), curves.std(axis=0, ddof=1)/np.sqrt(len(curves))
            axes[1, col].fill_between(source.x, mean-sem, mean+sem, color=COLORS[g], alpha=0.15, linewidth=0)
            axes[1, col].plot(source.x, mean, color=COLORS[g], linewidth=2.6, label=label)
        for row in range(2):
            axes[row, col].legend(loc="best")
            axes[row, col].margins(x=0.025)
    titles = ["A  Original subject curves", "B  One random reassignment",
              "C  Original means and SEM", "D  Reassigned means and SEM"]
    for ax, title in zip(axes.flat, titles):
        panel(ax, title, args.stimulation_label, args.experimental_response_label)
        ax.tick_params(labelbottom=True)
    footer(fig, "Repeated recordings are averaged within subject and condition. Shading: mean \u00b1 SEM.\n"
                "The surrogate groups are an illustration, not a separately tested comparison.", args)
    save_figure(fig, 1, args)
    csv_out(args.cache_dir / "figure_1_subject_curves.csv", curves_long(source.x, source.subjects, source.groups, "original") +
            curves_long(source.x, shuffled, names[1], "reassigned"))
    short = ("Curve-wise randomization. (A) Original subject-level input-output curves. (B) One random reassignment "
             "of complete curves, preserving group sizes. (C,D) Corresponding group means \u00b1 SEM. "
             "Recordings are averaged within each subject-condition before analysis.")
    full = (short + f" Data: {source.path.name}; {source.groups[0]} n={len(a)}, {source.groups[1]} n={len(b)}. "
            "Colours identify original or surrogate groups, not individual subject identity across panels. "
            "The surrogate assignment was fixed by the random seed and not selected by its p-value. "
            "Figure 2 evaluates only the original group assignment. ANH is not identified as a control condition here.")
    return {"title": "Curve-wise randomization of experimental data", "short": short, "full": full}


def figure2(source: Curves, data: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Notebook cell 5: null statistic histogram and repeat-estimation histograms.

    MC variability is numerical uncertainty for ONE fixed experimental dataset,
    not a distribution over new biological experiments. Sampling remains MC even
    when the full distribution is available as an exact reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.745, wspace=0.30)
    counts = data["stability_counts"]
    exact = data["exact_p"]
    status = (f"Exact reference: p = {exact:.4f} from {data['total_allocations']:,} distinct allocations"
              if exact is not None else "Repeated estimates assess Monte Carlo precision on one fixed dataset")
    header(fig, "Permutation inference and numerical precision",
           f"Original {source.groups[0]} vs {source.groups[1]} grouping | L2 = {data['observed_statistic']:.3f} | "
           f"MC p = {data['histogram_p']:.4f}", status, args)
    values = np.asarray(data["histogram_statistics"])
    axes[0].hist(values, bins=40, color=COLORS[0], alpha=0.78, label=f"Random allocations (B={len(values):,})")
    axes[0].axvline(data["observed_statistic"], color=COLORS[1], linewidth=2.2, label="Observed statistic")
    panel(axes[0], "A  Randomization distribution", r"Squared distance, $L^2$", "Random allocations (count)")
    axes[0].legend(loc="best")
    pseries = [np.asarray(v) for v in data["stability_pvalues"]]
    low = max(0.0, min(float(v.min()) for v in pseries) - 0.005)
    high = min(1.0, max(float(v.max()) for v in pseries) + 0.005)
    if high <= low:
        high = min(1.0, low+0.01)
    bins = np.linspace(low, high, 21)
    for i, (count, values) in enumerate(zip(counts, pseries)):
        axes[1].hist(values, bins=bins, histtype="step", linewidth=1.9, color=COLORS[i % len(COLORS)],
                     label=f"B = {count:,}")
    if exact is not None:
        axes[1].axvline(exact, color="0.25", linestyle="--", linewidth=1.2, label=f"Exact p = {exact:.4f}")
    panel(axes[1], "B  Repeated p-value estimates", "Estimated permutation p-value", "Repeated analyses (count)")
    axes[1].legend(loc="best")
    footer(fig, f"{data['stability_repeats']} independent re-estimates per B on the same data. "
                "B is the number of random allocations, not the number of subjects.\n"
                "Ties are included; Monte Carlo p-values use the +1 correction.", args)
    save_figure(fig, 2, args)
    csv_out(args.cache_dir / "figure_2_null_statistics.csv", [{"draw": i+1, "squared_l2": v} for i, v in enumerate(data["histogram_statistics"])])
    csv_out(args.cache_dir / "figure_2_repeated_pvalues.csv", [{"permutations": c, "repeat": i+1, "p_value": v}
            for c, values in zip(counts, data["stability_pvalues"]) for i, v in enumerate(values)])
    short = ("Permutation-based inference. (A) Squared-L2 distances after random reassignment of complete subject "
             "curves; the vertical line marks the original-group statistic. (B) Repeated p-value estimates at "
             "different permutation counts B show Monte Carlo variability on the same experimental dataset.")
    full = (short + f" The original statistic is {data['observed_statistic']:.9g}; its Monte Carlo p-value is "
            f"{data['histogram_p']:.6g} from B={len(data['histogram_statistics'])}. "
            f"Panel B uses B={counts} and {data['stability_repeats']} independent estimates per B. "
            "MC p=(1+inclusive exceedance count)/(B+1). "
            + (f"The exact reference p={exact:.9g} includes all {data['total_allocations']} allocations and ties. "
               "For this fixed dataset, uniformly resampling the full statistic list is equivalent in distribution "
               "to uniformly drawing allocations with replacement. " if exact is not None else "No exact reference was computed. ")
            + "L2 is the integral of the squared difference between piecewise-linear means, not the L2 norm. "
              "Its units are response-units squared times stimulation-current units. "
              "This precision experiment does not measure biological reproducibility or power.")
    return {"title": "Permutation inference and numerical precision", "short": short, "full": full}


def figure3(source: Curves, parameters: dict[str, Any], data: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Notebook cell 7, ORIGINAL ORDER: experimental, no difference, difference.

    Thin synthetic curves are RECORDINGS (n*m per group), as in the notebook.
    The subsequent statistical simulation averages m recordings per subject.
    This retains the old variance reduction instead of silently changing models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.0), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.745, wspace=0.23)
    header(fig, "Experimental and artificial input-output curves",
           f"Source: {source.path.name} | examples: n = {args.subjects} subjects/group, m = {args.recordings} recordings/subject",
           "Archive generator: shared gain/noise model; pooled null and separate PRE/POST templates", args)
    null = np.asarray(data["null"]).reshape(2, -1, len(source.x))
    alt = np.asarray(data["alternative"]).reshape(2, -1, len(source.x))
    panels = [source.raw, null, alt]
    names = [f"Control ({source.groups[0]})", f"LTP ({source.groups[1]})"]
    templates = [v.mean(axis=0) for v in source.raw]
    pooled = np.concatenate(source.raw).mean(axis=0)
    titles = ["A  Experimental recordings", "B  Null: same population", "C  Alternative: control vs LTP"]
    for j, (ax, arrays) in enumerate(zip(axes, panels)):
        for g, curves in enumerate(arrays):
            ax.plot(source.x, curves.T, color=COLORS[g], alpha=0.18, linewidth=0.75)
            label = names[g] if j != 1 else f"Null group {g+1}"
            ax.plot(source.x, curves.mean(axis=0), color=COLORS[g], linewidth=2.6,
                    label=f"{label}: {len(curves)} curves")
        if j == 1:
            ax.plot(source.x, pooled, color=COLORS[2], linewidth=1.8, linestyle="--", label="Shared true template")
        elif j == 2:
            for g in range(2):
                ax.plot(source.x, templates[g], color=COLORS[g], linestyle="--", linewidth=1.6)
        panel(ax, titles[j], args.stimulation_label)
        ax.legend(loc="best")
        ax.margins(x=0.025)
    axes[0].set_ylabel(args.simulation_response_label)
    footer(fig, "Thin lines: individual recordings. Solid lines: sample means. Dashed lines: generating templates.\n"
                f"Statistical tests average the {args.recordings} recordings assigned to each artificial subject; "
                "the two null samples are independently generated.", args)
    save_figure(fig, 3, args)
    csv_out(args.cache_dir / "figure_3_displayed_curves.csv", curves_long(source.x, source.raw, source.groups, "experimental") +
            curves_long(source.x, null, ["Null 1", "Null 2"], "null") + curves_long(source.x, alt, source.groups, "alternative"))
    write_json(args.cache_dir / "figure_3_simulation_parameters.json", parameters)
    short = ("Artificial input-output curves. (A) Original control/PRE and LTP/POST recordings. (B) Two independent "
             "samples from a common template and variability distribution. (C) Two samples generated around separate "
             "control and LTP templates. Thin lines show recordings, solid lines sample means, and dashed lines generating templates.")
    full = (short + f" Each artificial group contains n={args.subjects} subjects with m={args.recordings} independent "
            f"recordings per subject. Gains follow a shifted exponential with location={parameters['gain_location']:.9g} "
            f"and scale={parameters['gain_scale']:.9g}; additive Gaussian noise has stimulus-specific SDs read from "
            "input_data/archive_variability.json. No subject-shared random effect or cubic detrending is added. Recording-weighted source "
            "means supply the templates, matching to_fake(). The raw recordings are shown here; the tests use within-subject "
            "means. These seeded examples are not selected by significance and do not recover the unrecorded seeds of the notebook. "
            "Legacy figure-4 cache settings are unknown and must not be inferred from this illustration.")
    return {"title": "Experimental and artificial input-output curves", "short": short, "full": full}


def figure4(data: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Notebook cell 9: null/alternative p histograms with reversed log x-axis.

    All strictly positive values are plotted without clipping. Exact zeros have
    an adjacent non-log zero category; they are included in all summary counts.
    Geometric bins preserve notebook layout, not uniform-width density semantics.
    """
    legacy = data["origin"] == "legacy_notebook_pickles"
    arrays = [validate_pvalues(data[s], s) for s in ("null", "alternative")]
    positive = np.concatenate([v[v > 0] for v in arrays])
    floor = min(0.0005, float(positive.min()) * 0.8) if len(positive) else 0.0005
    floor = max(float(np.nextafter(0.0, 1.0)), floor)
    bins = np.geomspace(floor, 1.0, 51)
    fig = plt.figure(figsize=(12.8, 6.6))
    outer = fig.add_gridspec(1, 2, left=0.075, right=0.98, bottom=0.30, top=0.745, wspace=0.26)
    axes, zero_axes = [], []
    max_height = max(max(np.histogram(v[v > 0], bins=bins)[0].max(), np.count_nonzero(v == 0)) for v in arrays)
    for j in range(2):
        inner = outer[j].subgridspec(1, 2, width_ratios=[12, 1.2], wspace=0.08)
        ax = fig.add_subplot(inner[0])
        zero_ax = fig.add_subplot(inner[1], sharey=ax)
        axes.append(ax)
        zero_axes.append(zero_ax)
    if legacy:
        status = "LEGACY PICKLE RESULTS - values unchanged; generating B, n and seed are not recorded"
    else:
        low = args.mode == "test" or min(len(v) for v in arrays) < 2000
        status = ("TEST / LOW-PRECISION REGENERATION - not final validation" if low else
                  "Regenerated archive model; corrected p-values; subject-level analysis")
    detail = f"Null: R = {len(arrays[0]):,} experiments | Alternative: R = {len(arrays[1]):,} experiments"
    if not legacy:
        mode_label = "exact" if data["permutation_mode"] == "exact" else "Monte Carlo"
        detail += f" | n = {data['subjects']} | {mode_label}: {data['allocations_evaluated_per_test']:,} allocations/test"
    header(fig, "P-value distributions under the null and alternative", detail, status, args)
    rates = []
    for j, (ax, za, values, scenario) in enumerate(zip(axes, zero_axes, arrays, ("null", "alternative"))):
        ax.hist(values[values > 0], bins=bins, color=COLORS[j], alpha=0.78)
        ax.set_xscale("log")
        ax.set_xlim(1.18, floor)
        ax.set_ylim(0, max(1.0, max_height*1.12))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        panel(ax, "A  Null experiments" if j == 0 else "B  Alternative experiments",
              "p-value (log scale; decreasing)", "Simulated experiments (count)")
        for alpha, style in zip((0.05, 0.01, 0.001), ("--", ":", "-.")):
            ax.axvline(alpha, color="0.25", linestyle=style, linewidth=1)
        handles = [Line2D([0], [0], color="0.25", linestyle=style, label=f"alpha = {alpha:g}")
                   for alpha, style in zip((0.05, 0.01, 0.001), ("--", ":", "-."))]
        ax.legend(handles=handles, loc="upper right", fontsize=8.5*args.font_scale,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
        zeros = int(np.count_nonzero(values == 0))
        za.bar([0], [zeros], width=0.65, color=COLORS[j], alpha=0.78)
        za.set_xticks([0], ["0"])
        za.set_xlim(-0.7, 0.7)
        za.tick_params(axis="y", left=False, labelleft=False)
        za.spines["left"].set_visible(False)
        za.set_xlabel("p = 0", fontsize=9*args.font_scale)
        za.text(0, zeros+max(0.6, max_height*0.015), str(zeros), ha="center", va="bottom", fontsize=9*args.font_scale)
        za.set_title("Zero\nbin", fontsize=8.5*args.font_scale, pad=9)
        texts = []
        for alpha in (0.05, 0.01, 0.001):
            k = int(np.count_nonzero(values <= alpha))
            lo, hi = binomial_ci(k, len(values))
            rates.append({"scenario": scenario, "origin": data["origin"], "alpha": alpha, "experiments": len(values),
                          "rejections": k, "rejection_rate": k/len(values), "ci_low": lo, "ci_high": hi,
                          "zero_values_included": zeros})
            texts.append(f"{alpha:g}: {100*k/len(values):.1f}%")
        label = "False-positive fraction" if scenario == "null" else "Detection fraction"
        ax.text(0.52, -0.31, label + " (p <= alpha)\n" + " | ".join(texts), transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5*args.font_scale, linespacing=1.5)
    footer(fig, "Zero values have a separate non-log bin and remain in all counts. No p-values are clipped or discarded.\n"
                + ("Legacy values cannot be repaired from the p-value lists alone; regenerate before final statistical claims."
                   if legacy else "Geometric-bin counts are not a uniform-density diagnostic. Performance is conditional on this simulation model."), args)
    save_figure(fig, 4, args)
    csv_out(args.cache_dir / "figure_4_pvalues.csv", [{"scenario": scenario, "experiment": i+1, "p_value": float(p), "origin": data["origin"]}
            for scenario, values in zip(("null", "alternative"), arrays) for i, p in enumerate(values)])
    csv_out(args.cache_dir / "figure_4_rejection_rates.csv", rates)
    short = ("Null and alternative p-value distributions. (A) Independent null samples from a common population. "
             "(B) Samples generated around different control/LTP templates. Dashed reference lines mark alpha=0.05, "
             "0.01 and 0.001. Positive p-values use a reversed logarithmic axis; zeros occupy a separate non-log bin.")
    if legacy:
        short += " Previously saved notebook p-values are displayed unchanged; their generating-run settings are unverified."
    full = (short + f" Counts: R_null={len(arrays[0])}, R_alternative={len(arrays[1])}. "
            + ("These are the original notebook pickle values, not a new analysis. Their generating sample sizes, "
               "permutation counts and seeds are unverified. The archive's old p-value function excluded ties and "
               "did not use the +1 correction; stored values have not been adjusted post hoc. " if legacy else
               f"This run regenerated the archive model with n={data['subjects']} subjects/group and "
               f"m={data['recordings']} independent recordings/subject, averaged before testing. "
               f"Requested B={data['permutations']}; actual evaluation: {data['permutation_mode']}, "
               f"{data['allocations_evaluated_per_test']} allocations/test. New p-values include ties; MC tests "
               "use the +1 correction and exhaustive tests include all assignments. ")
            + f"There are {int(np.count_nonzero(arrays[0] == 0))} null and {int(np.count_nonzero(arrays[1] == 0))} "
              "alternative zeros. All enter the reported p<=alpha rejection proportions. "
              "Positive-value bins have equal log widths, so raw count heights are not a test of uniform density. "
              "Pointwise exact binomial intervals for rejection proportions are exported in the rates CSV. "
              "This plot concerns the permutation method only, not ANOVA or sample-size dependence.")
    return {"title": "P-value distributions under the null and alternative", "short": short, "full": full}


def write_explanations(captions: dict[int, dict[str, str]], args: argparse.Namespace, provenance: dict[str, Any]) -> None:
    for number,caption in captions.items():
        (args.out/f"{STEMS[number]}_caption.txt").write_text(caption['full']+"\n",encoding='utf-8')
    for filename, key in (("captions.txt", "full"), ("poster_captions.txt", "short")):
        text = "\n\n".join(f"Figure {i}. {c['title']}\n{c[key]}" for i, c in sorted(captions.items()))
        (args.out / filename).write_text(text + "\n", encoding="utf-8")
    notes = f"""NOTEBOOK POSTER FIGURES: METHODS AND CACHE NOTES

Figure mapping: notebook cells 3 -> figure 1; 5 -> figure 2; 7 -> figure 3;
9 -> figure 4. The panel order and geometry are retained. Plot styling follows
analysis/poster_validation_suite.py. The workflow uses shared subject-level modules in scripts/.
See examples/ for the separate, ID-matched paired and independent multi-group workflows.

This execution: mode={args.mode}; regenerate={args.regenerate}; seed={args.seed}.
Requested figures: {args.figures}. Cache directory: {args.cache_dir}.
Generation settings (DO NOT attribute these to legacy pickle runs):
  B={args.permutations} permutations per NEW simulation test;
  R={args.experiments} NEW experiments per null/alternative scenario;
  n={args.subjects} artificial subjects/group; m={args.recordings} recordings/subject;
  Figure 2 histogram B={args.histogram_permutations}; precision B={args.stability_counts};
  precision re-estimates per B={args.stability_repeats}.

Reuse precedence: compatible managed cache only; --legacy-pickles explicitly selects the old pair.
Uncached small pieces are computed on first use and then persisted. Missing
simulation p-value pickles never silently launch the large simulation.
--regenerate replaces only this script's own managed caches. A normal rerun resumes
any matching partial simulation cache. Settings/source mismatches stop with an
explanation. DPI, formats, labels, font scaling and output folder are plot-only.
Old scripts, CSVs, notebook, original dump pickles and previous poster files are
not edited. Do not run two writers against one cache directory.

Legacy p-values may include zeros and use an older convention. No mathematical
repair is possible from the lists alone without the original test counts and
statistics. This script preserves them, displays zeros explicitly, and exports
all raw values. A legacy histogram is not validated simply because it is redrawn
at high resolution or because --mode final was selected.

Fresh real-data and simulation tests: squared L2 of piecewise-linear mean curves;
subject-level reassignment; ties included. MC p=(1+count)/(B+1), exact p=count/all.
Figure 2 holds the biological data FIXED and always uses MC for its precision
experiment, even if a full exact reference is available. No significance statement
is made about the one surrogate example in figure 1.

Fresh simulations retain the ARCHIVE population model: separate recording-
weighted templates under H1, one pooled template under H0, common shifted-
exponential gains and independent Gaussian residuals with stimulus-specific SDs.
Three independently distorted recordings per artificial subject are averaged by
default. This reduces their variance and is not a subject-shared random effect.
This differs from the corrected within-group calibration in analysis/poster_validation_suite.py.
Do not mix results from the two population models without identifying them.

The null groups are independent samples from the same distribution, not duplicated
curves. Source PRE/POST curves are templates; these are independent synthetic-group
experiments, not validated paired-design analyses. ANH/RES labels are kept as given;
no control designation is invented for the CA1 example. Actual response units and
the biological meaning of subject IDs must be confirmed from acquisition metadata.

The number of permutations is not the number of experiments or the number of
subjects. Low-R test runs are only execution checks. Exact tests have finite
p-value resolution; increasing requested B beyond the complete allocation space
does not create new distinct assignments. Pointwise binomial intervals quantify
simulation error conditional on the fixed model, not uncertainty in source templates.

Reproducibility sources: run_manifest.json; exported figure-specific CSV files;
managed primitive-numeric pickles; source and cache SHA-256 hashes. No caches are
silently relabeled with modern parameters. Use only trusted pickles; the reader
rejects arbitrary Python object globals. All script settings and core equations
are documented in the source docstrings.
"""
    (args.out / "methods_and_cache_notes.txt").write_text(notes, encoding="utf-8")
    write_json(args.cache_dir / "run_manifest.json", {"created_utc": now(), "script_sha256": sha256(Path(__file__)),
        "engine": ENGINE_VERSION, "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "settings": vars(args),
        "provenance": provenance, "cache_hashes": {p.name: sha256(p) for p in args.cache_dir.glob("*.pkl")},
        "figure_files": [f"{STEMS[i]}.{f}" for i in captions for f in args.formats]})


def self_test() -> None:
    x = np.asarray([0.0, 0.3, 1.0])
    assert np.isclose(l2_squared(np.ones(3)*3, np.zeros(3), x), 9)
    assert np.isclose(l2_squared(x, np.zeros(3), x), 1/3)
    chol = integration_cholesky(x)
    a = np.asarray([[0.0, 0.1, 0.5], [0, 0.3, 0.9]])
    b = np.asarray([[0.0, 0.4, 1.2], [0, 0.6, 1.5]])
    t, values = allocation_statistics(a, b, chol, 1, rng_for(1), True)
    assert len(values) == 6
    assert np.isclose(t, l2_squared(a.mean(axis=0), b.mean(axis=0), x))
    pooled = np.concatenate([a, b])
    independent = []
    for idx in itertools.combinations(range(4), 2):
        first = list(idx); second = [i for i in range(4) if i not in idx]
        independent.append(l2_squared(pooled[first].mean(axis=0), pooled[second].mean(axis=0), x))
    assert np.allclose(values, independent, atol=1e-14)
    assert permutation_test(a*0, b*0, chol, 2, rng_for(1))[1] == 1
    assert permutation_test(a*0, b*0, chol, 20, rng_for(1))[1] == 1
    assert permutation_test(a, b, chol, 3, rng_for(9)) == permutation_test(a, b, chol, 3, rng_for(9))
    import io
    try:
        NumericUnpickler(io.BytesIO(pickle.dumps(os.system))).load()
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError("Unsafe pickle global was accepted")
    assert NumericUnpickler(io.BytesIO(pickle.dumps([0.0, 0.1, 1.0]))) .load() == [0.0, 0.1, 1.0]
    print("SELF-TEST PASSED: piecewise-linear integration, all allocations, ties, deterministic RNG, restricted pickle loading.")


def parse_arguments(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, help="Clean project root; detected from the package location.")
    p.add_argument("--experimental-data", type=Path, help="CA1 CSV; default input_data/stress_response_ca1.csv relative to project.")
    p.add_argument("--simulation-data", type=Path, help="Template CSV; default input_data/ltp_paired.csv.")
    p.add_argument("--experimental-groups", nargs=2, default=DEFAULT_EXPERIMENTAL_GROUPS, metavar=("GROUP1", "GROUP2"))
    p.add_argument("--simulation-groups", nargs=2, default=DEFAULT_SIMULATION_GROUPS, metavar=("CONTROL", "LTP"))
    p.add_argument("--mode", choices=PRESETS, default=DEFAULT_MODE)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--regenerate", dest="regenerate", action="store_true", help="Generate new data into NEW managed pickles; original caches untouched.")
    g.add_argument("--use-pickle", dest="regenerate", action="store_false", help="Reuse compatible saved data (default).")
    p.set_defaults(regenerate=not USE_PICKLE_FILES)
    p.add_argument("--legacy-pickles", action="store_true", help="Force ORIGINAL p-value pair for figure 4, ignoring regenerated results.")
    p.add_argument("--cache-dir", type=Path, help="Managed cache directory; default dumps/notebook/<mode>.")
    p.add_argument("--out", type=Path, help="Output directory; default figures/notebook/<mode>.")
    p.add_argument("--figures", nargs="+", type=int, choices=[1, 2, 3, 4], default=DEFAULT_FIGURES)
    p.add_argument("--permutations", type=int, help="B for each newly simulated experiment; ignored for legacy p-values.")
    p.add_argument("--experiments", type=int, help="R newly simulated experiments per scenario; ignored for legacy p-values.")
    p.add_argument("--histogram-permutations", type=int, help="B for the real-data statistic histogram, Figure 2A.")
    p.add_argument("--stability-counts", nargs="+", type=int, help="Permutation counts B for Figure 2B.")
    p.add_argument("--stability-repeats", type=int, help="Independent p-value re-estimates for each B in Figure 2B.")
    p.add_argument("--subjects", type=int, default=DEFAULT_SUBJECTS_PER_GROUP)
    p.add_argument("--recordings", type=int, default=DEFAULT_RECORDINGS_PER_SUBJECT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=DEFAULT_FORMATS)
    p.add_argument("--dpi", type=int)
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--experimental-response-label", default=EXPERIMENTAL_RESPONSE_LABEL)
    p.add_argument("--simulation-response-label", default=SIMULATION_RESPONSE_LABEL)
    p.add_argument("--stimulation-label", default=STIMULATION_LABEL)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    for key, default in PRESETS[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.legacy_pickles and args.regenerate:
        p.error("--legacy-pickles and --regenerate are mutually exclusive.")
    for key in ("permutations", "experiments", "histogram_permutations", "stability_repeats", "recordings", "dpi"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive.")
    if args.subjects < 2 or args.seed < 0 or args.font_scale <= 0 or not all(v > 0 for v in args.stability_counts):
        p.error("Require subjects >= 2, seed >= 0, positive font-scale and stability-counts.")
    if len(set(args.stability_counts)) != len(args.stability_counts):
        p.error("--stability-counts must be distinct.")
    if len(set(args.experimental_groups)) != 2 or len(set(args.simulation_groups)) != 2:
        p.error("Each group pair must contain two different labels.")
    args.figures = sorted(set(args.figures))
    args.formats = list(dict.fromkeys(args.formats))
    if args.self_test:
        return args
    args.project = (args.project or ROOT).expanduser().resolve()
    for key, default in (("experimental_data", "input_data/stress_response_ca1.csv"),
                         ("simulation_data", "input_data/ltp_paired.csv")):
        value = getattr(args,key) or Path(default)
        setattr(args,key,value.resolve() if value.is_absolute() else (args.project/value).resolve())
    args.cache_dir = args.cache_dir or args.project/"dumps"/"notebook"/("legacy" if args.legacy_pickles else args.mode)
    args.out = args.out or args.project/"figures"/"notebook"/("legacy" if args.legacy_pickles else args.mode)
    args.cache_dir = args.cache_dir.expanduser()
    args.out = args.out.expanduser()
    if not args.cache_dir.is_absolute(): args.cache_dir=args.project/args.cache_dir
    if not args.out.is_absolute(): args.out=args.project/args.out
    args.cache_dir=args.cache_dir.resolve();args.out=args.out.resolve()
    if args.cache_dir in (args.project,args.project/"input_data",args.project/"dumps"/"legacy"):
        p.error("Use a dedicated managed cache subfolder, never source data or historical dumps.")
    return args


def main(argv=None) -> None:
    args = parse_arguments(argv)
    if args.self_test:
        self_test()
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    configure_plots(args.font_scale)
    captions, provenance = {}, {}
    # Never silently start a large simulation or present unknown historical settings.
    if 4 in args.figures and not args.regenerate:
        if args.legacy_pickles:
            if not all((args.project/"dumps"/"legacy"/name).exists() for name in LEGACY_FILENAMES):
                raise FileNotFoundError("Historical pickle pair is missing.")
        elif not (args.cache_dir/"simulation_pvalues.pkl").exists():
            raise FileNotFoundError("Managed p-values absent. Run --regenerate --mode test (or final). "
                                    "Use --legacy-pickles only to display unchanged historical results.")
    if any(i in args.figures for i in (1, 2)):
        source = load_csv(args.experimental_data, tuple(args.experimental_groups))
        provenance["experimental_data"] = {"path": str(source.path), "sha256": sha256(source.path),
                                           "groups": source.groups, "n": [len(v) for v in source.subjects]}
        if 1 in args.figures:
            captions[1] = figure1(source, experimental_example(source, args), args)
        if 2 in args.figures:
            inf = inference_data(source, args)
            captions[2] = figure2(source, inf, args)
            provenance["inference"] = {"exact_p": inf["exact_p"], "p_mc": inf["histogram_p"],
                                       "statistic": inf["observed_statistic"], "created_utc": inf["created_utc"]}
    if any(i in args.figures for i in (3, 4)):
        source = load_csv(args.simulation_data, tuple(args.simulation_groups))
        parameters = archive_parameters(args.project, len(source.x))
        provenance["simulation_templates"] = {"path": str(source.path), "sha256": sha256(source.path),
                                             "parameters": parameters, "groups": source.groups}
        if 3 in args.figures:
            captions[3] = figure3(source, parameters, example_simulations(source, parameters, args), args)
        if 4 in args.figures:
            values = pvalue_data(source, parameters, args)
            captions[4] = figure4(values, args)
            provenance["pvalues"] = {k: v for k, v in values.items() if k not in
                                     ("null", "alternative", "null_statistics", "alternative_statistics")}
    write_explanations(captions, args, provenance)
    print(f"\nDone: {args.out}\nCaptions: captions.txt / poster_captions.txt\n"
          f"Cache: {args.cache_dir}\nOriginal archive files were not changed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Matching simulation checkpoints resume on the next run WITHOUT --regenerate.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, pickle.UnpicklingError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
