"""Empirically informed populations, retaining the two supplied calibration modes.

within-group: pooled within-condition gains/residuals; one generated curve per
independent subject. archive: unchanged historical parameters and three
independent recordings averaged per subject. No hidden switch between models.
Templates remain fixed across sample-size conditions. Source paired recordings
are templates, not reclassified as independent biological observations.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
from .subject_results import SubjectResults
from .cache import sha256_file

@dataclass
class SourceData:
    x: np.ndarray
    labels: tuple[str, ...]
    subjects: tuple[np.ndarray, ...]
    raw: tuple[np.ndarray, ...]
    ids: tuple[list[str], ...]
    row_counts: tuple[int, ...]
    ignored_groups: list[str]
    path: Path

    @property
    def groups(self) -> tuple[str, ...]:
        return self.labels


def load_source(path: Path, *labels: str) -> SourceData:
    data=SubjectResults.from_csv(path)
    groups=tuple(labels) if labels else tuple(data.get_group_ids())
    arrays,ids=data.subject_arrays(groups)
    raw=tuple(np.stack([r.responses for r in data.results if r.group_id==g]) for g in groups)
    return SourceData(data.currents,groups,arrays,raw,ids,tuple(map(len,raw)),
                      sorted(set(data.get_group_ids())-set(groups)),Path(path).resolve())


def load_csv(path: Path, groups: tuple[str, ...]) -> SourceData:
    return load_source(path,*groups)


def archive_parameters(project: Path, k: int) -> dict[str,Any]:
    path=Path(project)/'input_data'/'archive_variability.json'
    params=json.loads(path.read_text(encoding='utf-8'))
    sd=np.asarray(params['noise_sd'],dtype=float)
    if sd.shape!=(k,) or not np.isfinite(sd).all() or np.any(sd<0):
        raise ValueError('Archive noise profile does not match the requested grid.')
    if not np.isfinite(params['gain_location']) or not np.isfinite(params['gain_scale']) or params['gain_scale']<0:
        raise ValueError('Invalid historical gain parameters.')
    return {**params,'file_sha256':sha256_file(path)}


def read_archive_constants(project: Path, k: int):
    params=archive_parameters(project,k)
    return params['gain_location'],params['gain_scale'],np.asarray(params['noise_sd']),params

@dataclass
class Model:
    x: np.ndarray
    templates: np.ndarray
    null_template: np.ndarray
    gain_location: float
    gain_scale: float
    noise_sd: np.ndarray
    recordings_per_subject: int
    calibration_gains: np.ndarray
    calibration_residuals: np.ndarray
    description: str
    source_parameters: dict[str, Any]


def calibrate(source: SourceData, model_name: str, project: Path) -> Model:
    """Default calibration removes the condition effect before pooling deviations."""
    if model_name not in ("within-group", "archive"):
        raise ValueError("Unknown calibration model.")
    subject_templates = np.stack([y.mean(axis=0) for y in source.subjects])
    gains, residuals = [], []
    for y, template in zip(source.subjects, subject_templates):
        denominator = float(template @ template)
        if denominator <= np.finfo(float).tiny:
            raise ValueError("Cannot calibrate multiplicative gain from an all-zero group template.")
        a = y @ template / denominator
        e = y - a[:, None] * template
        e -= e.mean(axis=0)                 # Explicit within-condition centering.
        gains.append(a)
        residuals.append(e)
    pooled_gains = np.concatenate(gains)
    pooled_residuals = np.concatenate(residuals, axis=0)
    if model_name == "archive":
        location, scale, sd, parameters = read_archive_constants(project, len(source.x))
        if not np.array_equal(source.x, np.asarray(parameters["stimulus_grid"])):
            raise ValueError("Historical model requires its original stimulation grid.")
        templates = np.stack([y.mean(axis=0) for y in source.raw])
        null_template = np.concatenate(source.raw, axis=0).mean(axis=0)
        m = 3
        description = ("Archive generator: fixed shifted-exponential gain and stimulus-specific "
                       "Gaussian noise; three independent recordings averaged per simulated subject.")
    else:
        # For an unconstrained shifted exponential, the MLE location is min(A)
        # and its scale is mean(A)-min(A). Normalize to E[A]=1 explicitly.
        gain_mean = float(pooled_gains.mean())
        if gain_mean <= 0 or pooled_gains.min() < 0:
            raise ValueError("Nonpositive empirical gains: the positive-gain model is unsuitable for this dataset.")
        location = float(pooled_gains.min()) / gain_mean
        scale = float(pooled_gains.mean() - pooled_gains.min()) / gain_mean
        if scale <= np.finfo(float).eps:
            scale = 0.0                    # Deterministic gain if all gains coincide.
            location = 1.0
        sd = pooled_residuals.std(axis=0, ddof=1)
        templates = subject_templates
        null_template = np.concatenate(source.subjects, axis=0).mean(axis=0)
        m = 1
        parameters = {"gain_fit": "shifted exponential MLE, normalized to mean one",
                      "residual_sd_ddof": 1, "residual_detrending": "none",
                      "deviations": "each subject relative to its own condition template"}
        description = ("Within-condition calibration: pooled within-group gain/residual variability; "
                       "one generated curve per independent simulated subject.")
    return Model(source.x, templates, null_template, location, scale, sd, m,
                 pooled_gains, pooled_residuals, description, parameters)


def simulate_experiment(model: Model, n: int, scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Return one independent group of n subject curves for each template."""
    if scenario not in ("null", "alternative") or n < 2:
        raise ValueError("Use a null/alternative scenario and at least two subjects per group.")
    templates = model.templates if scenario == "alternative" else np.repeat(model.null_template[None, :], len(model.templates), axis=0)
    m, k = model.recordings_per_subject, len(model.x)
    a = model.gain_location + rng.exponential(model.gain_scale, size=(len(templates), n, m, 1))
    noise = rng.normal(size=(len(templates), n, m, k)) * model.noise_sd
    y = (a * templates[:, None, None, :] + noise).mean(axis=2)
    return tuple(y)


def simulated_recordings(source: SourceData, parameters: dict[str, Any], n: int, m: int,
                         alternative: bool, rng: np.random.Generator) -> np.ndarray:
    if not np.array_equal(source.x, np.asarray(parameters["stimulus_grid"])):
        raise ValueError("Historical variability parameters require the original stimulus grid.")
    templates = np.stack([v.mean(axis=0) for v in source.raw])
    if not alternative:
        template = np.concatenate(source.raw).mean(axis=0)
        templates = np.stack([template, template])
    gain = parameters["gain_location"] + rng.exponential(parameters["gain_scale"], size=(2, n, m, 1))
    noise = rng.normal(size=(2, n, m, len(source.x))) * np.asarray(parameters["noise_sd"])
    return gain * templates[:, None, None, :] + noise
