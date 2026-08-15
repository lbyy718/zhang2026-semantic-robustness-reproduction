"""Analyze the common-semantic-endpoint mechanism diagnostics.

The statistical unit is the training seed.  Channel repeats are summarized
within a seed before any across-seed mean or standard deviation is computed.
The script deliberately keeps clean semantic errors in the system result:
their failure power is zero.  Clean-correct samples on which PGA does not find
a failure have infinite (right-censored) failure power.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "factorial"
    / "factorial_mechanism_v1"
    / "diagnostics"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "analysis"
CELLS = ("R0", "R1", "C0", "C1")
COMPARISONS = (
    ("R1_minus_R0", "R0", "R1", "reconstruction"),
    ("C1_minus_C0", "C0", "C1", "classification"),
)
POWER_GRID_DB = np.arange(-40.0, 1.0, 1.0)
ACCEPTED_REGISTRY_STATES = {"completed", "skipped_complete"}
EPSILON = 1e-30


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"Missing integer field {field!r}.")
    return int(value)


def as_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"Missing floating-point field {field!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite value in {field!r}: {value!r}.")
    return parsed


def optional_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def as_bool(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE"):
        return False
    raise ValueError(f"Invalid Boolean value in {field!r}: {value!r}.")


def _resolve_recorded_path(value: str, input_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (input_root / path).resolve()


def load_validated_diagnostics(
    input_root: Path,
    *,
    expected_jobs: int = 36,
    enforce_standard_design: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load diagnostics after independently checking registry, manifests, hashes.

    ``expected_jobs`` and ``enforce_standard_design`` are parameters so a small
    synthetic registry can exercise the same validation path in unit tests.
    Production defaults require the complete 4 x 3 x 3 design.
    """

    root = input_root.resolve()
    registry_path = root / "registry.csv"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Mechanism registry is missing: {registry_path}")
    registry = read_csv(registry_path)
    if len(registry) != expected_jobs:
        raise ValueError(
            f"Expected {expected_jobs} diagnostic jobs, found {len(registry)}."
        )
    identities: set[tuple[str, int, int, int]] = set()
    all_rows: list[dict[str, str]] = []
    rows_per_job: set[int] = set()
    job_index_sets: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(dict)
    channel_seeds: dict[tuple[int, int], set[int]] = defaultdict(set)

    for registry_row in registry:
        status = registry_row.get("status", "")
        if status not in ACCEPTED_REGISTRY_STATES:
            raise ValueError(
                f"Registry job {registry_row.get('cell')}/"
                f"{registry_row.get('training_seed')}/"
                f"{registry_row.get('repeat_index')} has status {status!r}."
            )
        cell = registry_row["cell"].upper()
        seed = int(registry_row["training_seed"])
        repeat = int(registry_row["repeat_index"])
        channel_seed = int(registry_row["channel_seed"])
        identity = (cell, seed, repeat, channel_seed)
        if identity in identities:
            raise ValueError(f"Duplicate registry identity: {identity}")
        identities.add(identity)
        output_dir = _resolve_recorded_path(registry_row["output_dir"], root)
        diagnostic_path = output_dir / "diagnostics.csv"
        manifest_path = output_dir / "manifest.json"
        if not diagnostic_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Final artifacts are missing for {identity}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Manifest is not completed for {identity}.")
        if bool(manifest.get("skip_pga")) or bool(manifest.get("skip_spectral")):
            raise ValueError(f"PGA or spectral diagnostics were skipped for {identity}.")
        if manifest.get("diagnostics_csv_sha256") != sha256(diagnostic_path):
            raise ValueError(f"Diagnostics hash mismatch for {identity}.")
        rows = read_csv(diagnostic_path)
        if len(rows) != int(manifest.get("rows", -1)):
            raise ValueError(f"Diagnostics row-count mismatch for {identity}.")
        rows_per_job.add(len(rows))
        indices: set[int] = set()
        for row in rows:
            row_identity = (
                row.get("cell", "").upper(),
                as_int(row, "training_seed"),
                as_int(row, "repeat_index"),
                as_int(row, "channel_seed"),
            )
            if row_identity != identity:
                raise ValueError(
                    f"CSV identity {row_identity} does not match registry {identity}."
                )
            dataset_index = as_int(row, "dataset_index")
            if dataset_index in indices:
                raise ValueError(f"Duplicate dataset index in job {identity}.")
            indices.add(dataset_index)
            # Validate the fields required by every downstream result now.
            as_bool(row, "clean_semantic_correct")
            as_float(row, "semantic_failure_margin")
            as_float(row, "semantic_margin_gradient_l2")
            as_float(row, "semantic_linearized_distance_l2")
            as_float(row, "semantic_spectral_norm")
            as_bool(row, "pga_success")
        job_index_sets[seed, repeat][cell] = indices
        channel_seeds[seed, repeat].add(channel_seed)
        all_rows.extend(rows)

    if len(rows_per_job) != 1:
        raise ValueError(f"Jobs have unequal diagnostic row counts: {rows_per_job}")
    for seed_repeat, by_cell in job_index_sets.items():
        sets = list(by_cell.values())
        if len(sets) > 1 and any(value != sets[0] for value in sets[1:]):
            raise ValueError(f"Dataset indices are not paired at {seed_repeat}.")
        if len(channel_seeds[seed_repeat]) != 1:
            raise ValueError(f"Channel seed differs across cells at {seed_repeat}.")

    cells = sorted({identity[0] for identity in identities})
    seeds = sorted({identity[1] for identity in identities})
    repeats = sorted({identity[2] for identity in identities})
    if enforce_standard_design:
        if set(cells) != set(CELLS) or len(seeds) != 3 or repeats != [0, 1, 2]:
            raise ValueError(
                "The production analysis requires R0/R1/C0/C1, three training "
                "seeds, and repeats 0/1/2."
            )
        expected = {
            (cell, seed, repeat)
            for cell in CELLS
            for seed in seeds
            for repeat in repeats
        }
        observed = {(cell, seed, repeat) for cell, seed, repeat, _ in identities}
        if observed != expected:
            raise ValueError("The 4 x 3 x 3 mechanism design is incomplete.")

    batch_manifest_path = root / "batch_manifest.json"
    if batch_manifest_path.is_file():
        batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        if batch_manifest.get("status") != "completed":
            raise ValueError("Batch manifest is not completed.")
    metadata = {
        "jobs": len(registry),
        "rows": len(all_rows),
        "rows_per_job": next(iter(rows_per_job)),
        "cells": cells,
        "training_seeds": seeds,
        "repeat_indices": repeats,
    }
    return all_rows, metadata


def failure_power_per_use(
    row: dict[str, Any], *, unresolved_policy: str = "upper"
) -> float:
    """Return a PGA-estimated failure threshold under one censoring policy.

    ``upper`` treats a clean-correct unsuccessful PGA run as surviving every
    plotted budget (infinite threshold), yielding an optimistic empirical
    robust-accuracy bound. ``lower`` treats it as non-robust at every positive
    plotted budget (zero threshold), yielding a deliberately conservative
    bound. Neither curve is a certified robustness guarantee.
    """

    if unresolved_policy not in {"lower", "upper"}:
        raise ValueError("unresolved_policy must be 'lower' or 'upper'.")

    if not as_bool(row, "clean_semantic_correct"):
        return 0.0
    if not as_bool(row, "pga_success"):
        return math.inf if unresolved_policy == "upper" else 0.0
    power = as_float(row, "pga_attack_power_per_channel_use")
    if power < 0:
        raise ValueError("PGA power cannot be negative.")
    return power


def robust_accuracy(
    rows: Sequence[dict[str, Any]],
    budgets: np.ndarray,
    *,
    unresolved_policy: str = "upper",
) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot calculate robust accuracy from no rows.")
    thresholds = np.asarray(
        [
            failure_power_per_use(row, unresolved_policy=unresolved_policy)
            for row in rows
        ],
        dtype=float,
    )
    return np.asarray([(thresholds > budget).mean() for budget in budgets], dtype=float)


def pga_empirical_robust_accuracy_bounds(
    rows: Sequence[dict[str, Any]], budgets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative/optimistic empirical bounds for unresolved PGA runs."""

    lower = robust_accuracy(rows, budgets, unresolved_policy="lower")
    upper = robust_accuracy(rows, budgets, unresolved_policy="upper")
    if np.any(lower > upper):
        raise RuntimeError("PGA empirical lower bound exceeded upper bound.")
    return lower, upper


def normalized_auc(curve: np.ndarray, grid_db: np.ndarray) -> float:
    if len(curve) != len(grid_db) or len(curve) < 2:
        raise ValueError("AUC requires aligned curves with at least two points.")
    segment_areas = 0.5 * (curve[:-1] + curve[1:]) * np.diff(grid_db)
    return float(segment_areas.sum() / (grid_db[-1] - grid_db[0]))


def retention_intersection(
    curve: np.ndarray,
    grid_db: np.ndarray,
    clean_accuracy: float,
    retained_fraction: float,
) -> tuple[float | None, str]:
    target = retained_fraction * clean_accuracy
    if curve[0] <= target:
        return None, "censored_below_grid"
    crossing = np.flatnonzero(curve <= target)
    if not len(crossing):
        return None, "censored_above_grid"
    index = int(crossing[0])
    left_x, right_x = float(grid_db[index - 1]), float(grid_db[index])
    left_y, right_y = float(curve[index - 1]), float(curve[index])
    if left_y == right_y:
        return right_x, "observed"
    fraction = (target - left_y) / (right_y - left_y)
    return left_x + fraction * (right_x - left_x), "observed"


def common_clean_correct_indices(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[str] = CELLS,
) -> dict[tuple[int, int], set[int]]:
    correct: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for row in rows:
        if as_bool(row, "clean_semantic_correct"):
            correct[
                as_int(row, "training_seed"),
                as_int(row, "repeat_index"),
                row["cell"].upper(),
            ].add(as_int(row, "dataset_index"))
    keys = sorted({(as_int(row, "training_seed"), as_int(row, "repeat_index")) for row in rows})
    result: dict[tuple[int, int], set[int]] = {}
    for seed, repeat in keys:
        cell_sets = [correct[seed, repeat, cell.upper()] for cell in cells]
        result[seed, repeat] = set.intersection(*cell_sets)
    return result


def _mean_optional(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(selected) if selected else None


def _geometric_repeat_mean(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [optional_float(row, field) for row in rows]
    positive = [value for value in values if value is not None and value > 0]
    return math.exp(statistics.mean(math.log(value) for value in positive)) if positive else None


def _pearson_log_distance_attack(rows: Sequence[dict[str, Any]]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        distance = optional_float(row, "semantic_linearized_distance_l2")
        attack_sq = optional_float(row, "pga_attack_power_total_l2_sq")
        if (
            as_bool(row, "clean_semantic_correct")
            and as_bool(row, "pga_success")
            and distance is not None
            and distance > 0
            and attack_sq is not None
            and attack_sq > 0
        ):
            pairs.append((math.log(distance), 0.5 * math.log(attack_sq)))
    if len(pairs) < 3:
        return None
    left = np.asarray([pair[0] for pair in pairs])
    right = np.asarray([pair[1] for pair in pairs])
    if float(left.std()) == 0 or float(right.std()) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def fisher_z_mean_correlations(values: Iterable[float | None]) -> float | None:
    """Average repeat-level correlations in Fisher-z space."""

    selected = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not selected:
        return None
    clipped = np.clip(np.asarray(selected, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(clipped).mean()))


def build_robust_curve_outputs(
    rows: Sequence[dict[str, Any]],
    grid_db: np.ndarray = POWER_GRID_DB,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return PGA-estimated curve bounds and one summary per training seed."""

    budgets = 10.0 ** (grid_db / 10.0)
    pairwise_common = {
        "reconstruction": common_clean_correct_indices(rows, ("R0", "R1")),
        "classification": common_clean_correct_indices(rows, ("C0", "C1")),
    }
    four_cell_common = common_clean_correct_indices(rows, CELLS)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            row["cell"].upper(),
            as_int(row, "training_seed"),
            as_int(row, "repeat_index"),
        ].append(row)
    seeds = sorted({key[1] for key in grouped})
    repeats = sorted({key[2] for key in grouped})
    scopes = ("all", "pairwise_common_clean_correct", "four_cell_common_clean_correct")
    curve_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    seed_curves: dict[tuple[str, int, str, str], np.ndarray] = {}

    for cell in CELLS:
        objective = "reconstruction" if cell.startswith("R") else "classification"
        for seed in seeds:
            repeat_curves: dict[str, dict[str, list[np.ndarray]]] = {
                scope: {"lower": [], "upper": []} for scope in scopes
            }
            repeat_clean: list[float] = []
            repeat_pairwise_n: list[int] = []
            repeat_four_n: list[int] = []
            repeat_success: dict[str, list[float]] = {scope: [] for scope in scopes}
            repeat_spectral_convergence: list[float] = []
            repeat_correlations: list[float | None] = []
            repeat_margin_geomean: list[float | None] = []
            repeat_gradient_geomean: list[float | None] = []
            repeat_distance_geomean: list[float | None] = []
            repeat_spectral_geomean: list[float | None] = []
            repeat_attack_geomean: list[float | None] = []
            for repeat in repeats:
                selected = grouped[cell, seed, repeat]
                if not selected:
                    raise ValueError(f"Missing repeat {cell}/{seed}/{repeat}.")
                pairwise_ids = pairwise_common[objective][seed, repeat]
                four_ids = four_cell_common[seed, repeat]
                scoped_rows = {
                    "all": selected,
                    "pairwise_common_clean_correct": [
                        row
                        for row in selected
                        if as_int(row, "dataset_index") in pairwise_ids
                    ],
                    "four_cell_common_clean_correct": [
                        row
                        for row in selected
                        if as_int(row, "dataset_index") in four_ids
                    ],
                }
                if not scoped_rows["pairwise_common_clean_correct"]:
                    raise ValueError(f"No pairwise common-correct samples at {cell}/{seed}/{repeat}.")
                if not scoped_rows["four_cell_common_clean_correct"]:
                    raise ValueError(f"No four-cell common-correct samples at {seed}/{repeat}.")
                for scope, scope_rows in scoped_rows.items():
                    lower, upper = pga_empirical_robust_accuracy_bounds(
                        scope_rows, budgets
                    )
                    repeat_curves[scope]["lower"].append(lower)
                    repeat_curves[scope]["upper"].append(upper)
                    clean_correct = [
                        row for row in scope_rows if as_bool(row, "clean_semantic_correct")
                    ]
                    repeat_success[scope].append(
                        statistics.mean(as_bool(row, "pga_success") for row in clean_correct)
                        if clean_correct
                        else 0.0
                    )
                correct_rows = [row for row in selected if as_bool(row, "clean_semantic_correct")]
                repeat_clean.append(len(correct_rows) / len(selected))
                repeat_pairwise_n.append(len(scoped_rows["pairwise_common_clean_correct"]))
                repeat_four_n.append(len(scoped_rows["four_cell_common_clean_correct"]))
                repeat_spectral_convergence.append(
                    statistics.mean(
                        as_bool(row, "semantic_spectral_converged_20_30")
                        for row in selected
                    )
                )
                repeat_correlations.append(_pearson_log_distance_attack(selected))
                repeat_margin_geomean.append(
                    _geometric_repeat_mean(correct_rows, "semantic_failure_margin")
                )
                repeat_gradient_geomean.append(
                    _geometric_repeat_mean(correct_rows, "semantic_margin_gradient_l2")
                )
                repeat_distance_geomean.append(
                    _geometric_repeat_mean(correct_rows, "semantic_linearized_distance_l2")
                )
                repeat_spectral_geomean.append(
                    _geometric_repeat_mean(selected, "semantic_spectral_norm")
                )
                successful = [
                    row
                    for row in correct_rows
                    if as_bool(row, "pga_success")
                    and (optional_float(row, "pga_attack_power_per_channel_use") or 0) > 0
                ]
                repeat_attack_geomean.append(
                    _geometric_repeat_mean(successful, "pga_attack_power_per_channel_use")
                )

            averaged: dict[str, dict[str, np.ndarray]] = {
                scope: {
                    bound: np.mean(np.stack(repeat_curves[scope][bound]), axis=0)
                    for bound in ("lower", "upper")
                }
                for scope in scopes
            }
            for scope in scopes:
                for bound in ("lower", "upper"):
                    seed_curves[cell, seed, scope, bound] = averaged[scope][bound]
                for index, power_db in enumerate(grid_db):
                    curve_rows.append(
                        {
                            "aggregation_level": "training_seed",
                            "cell": cell,
                            "objective": objective,
                            "training_seed": seed,
                            "channel_repeats_averaged": len(repeats),
                            "sample_scope": scope,
                            "attack_power_per_use_db": float(power_db),
                            "pga_empirical_robust_accuracy_lower_bound": float(averaged[scope]["lower"][index]),
                            "pga_empirical_robust_accuracy_upper_bound": float(averaged[scope]["upper"][index]),
                            "lower_bound_sd_across_training_seeds": "",
                            "upper_bound_sd_across_training_seeds": "",
                        }
                    )
            clean_accuracy = statistics.mean(repeat_clean)
            summary: dict[str, Any] = {
                "cell": cell,
                "objective": objective,
                "training_seed": seed,
                "channel_repeats_averaged": len(repeats),
                "samples_per_repeat": len(grouped[cell, seed, repeats[0]]),
                "pairwise_common_clean_correct_samples_per_repeat_mean": statistics.mean(repeat_pairwise_n),
                "four_cell_common_clean_correct_samples_per_repeat_mean": statistics.mean(repeat_four_n),
                "clean_semantic_accuracy_repeat_mean": clean_accuracy,
                "pga_success_rate_among_all_clean_correct_repeat_mean": statistics.mean(repeat_success["all"]),
                "pga_success_rate_pairwise_common_repeat_mean": statistics.mean(repeat_success["pairwise_common_clean_correct"]),
                "pga_success_rate_four_cell_common_repeat_mean": statistics.mean(repeat_success["four_cell_common_clean_correct"]),
                "semantic_failure_margin_geomean_repeat_mean": _mean_optional(repeat_margin_geomean),
                "semantic_margin_gradient_l2_geomean_repeat_mean": _mean_optional(repeat_gradient_geomean),
                "semantic_linearized_distance_l2_geomean_repeat_mean": _mean_optional(repeat_distance_geomean),
                "semantic_spectral_norm_geomean_repeat_mean": _mean_optional(repeat_spectral_geomean),
                "spectral_convergence_rate_repeat_mean": statistics.mean(repeat_spectral_convergence),
                "pga_power_per_use_geomean_successful_complete_case_repeat_mean": _mean_optional(repeat_attack_geomean),
                "pearson_ln_linear_distance_vs_ln_pga_l2_fisher_z_repeat_mean": fisher_z_mean_correlations(repeat_correlations),
                "operational_grid_db": "[-40, 0] inclusive, 1 dB step; trapezoidal approximation",
            }
            for scope in scopes:
                scope_clean = clean_accuracy if scope == "all" else 1.0
                for bound in ("lower", "upper"):
                    curve = averaged[scope][bound]
                    summary[
                        f"pga_operational_auc_normalized_{scope}_{bound}_bound"
                    ] = normalized_auc(curve, grid_db)
                    for retention in (0.90, 0.75):
                        power, status = retention_intersection(
                            curve, grid_db, scope_clean, retention
                        )
                        label = int(retention * 100)
                        prefix = f"pga_operational_retention_{label}_{scope}_{bound}_bound"
                        summary[f"{prefix}_power_db"] = "" if power is None else power
                        summary[f"{prefix}_status"] = status
            seed_summaries.append(summary)

    for cell in CELLS:
        objective = "reconstruction" if cell.startswith("R") else "classification"
        for scope in scopes:
            matrices = {
                bound: np.stack(
                    [seed_curves[cell, seed, scope, bound] for seed in seeds]
                )
                for bound in ("lower", "upper")
            }
            for index, power_db in enumerate(grid_db):
                curve_rows.append(
                    {
                        "aggregation_level": "across_training_seeds",
                        "cell": cell,
                        "objective": objective,
                        "training_seed": "",
                        "channel_repeats_averaged": len(repeats),
                        "sample_scope": scope,
                        "attack_power_per_use_db": float(power_db),
                        "pga_empirical_robust_accuracy_lower_bound": float(matrices["lower"][:, index].mean()),
                        "pga_empirical_robust_accuracy_upper_bound": float(matrices["upper"][:, index].mean()),
                        "lower_bound_sd_across_training_seeds": (
                            float(matrices["lower"][:, index].std(ddof=1)) if len(seeds) > 1 else ""
                        ),
                        "upper_bound_sd_across_training_seeds": (
                            float(matrices["upper"][:, index].std(ddof=1)) if len(seeds) > 1 else ""
                        ),
                    }
                )
    return curve_rows, seed_summaries


def paired_log_identity_diagnostics(
    off_row: dict[str, Any], on_row: dict[str, Any]
) -> dict[str, float | None]:
    def log_value(row: dict[str, Any], field: str) -> float:
        value = as_float(row, field)
        if value <= 0:
            raise ValueError(f"{field} must be positive for the log-distance identity.")
        return math.log(value)

    delta_margin = log_value(on_row, "semantic_failure_margin") - log_value(
        off_row, "semantic_failure_margin"
    )
    delta_gradient = log_value(on_row, "semantic_margin_gradient_l2") - log_value(
        off_row, "semantic_margin_gradient_l2"
    )
    delta_distance = log_value(on_row, "semantic_linearized_distance_l2") - log_value(
        off_row, "semantic_linearized_distance_l2"
    )
    attack_delta: float | None = None
    if as_bool(off_row, "pga_success") and as_bool(on_row, "pga_success"):
        off_attack = optional_float(off_row, "pga_attack_power_total_l2_sq")
        on_attack = optional_float(on_row, "pga_attack_power_total_l2_sq")
        if off_attack is not None and on_attack is not None and off_attack > 0 and on_attack > 0:
            attack_delta = 0.5 * math.log(on_attack / off_attack)
    off_spectral = optional_float(off_row, "semantic_spectral_norm")
    on_spectral = optional_float(on_row, "semantic_spectral_norm")
    spectral_delta = (
        math.log(on_spectral / off_spectral)
        if off_spectral is not None
        and on_spectral is not None
        and off_spectral > 0
        and on_spectral > 0
        else None
    )
    return {
        "delta_ln_margin": delta_margin,
        "delta_ln_gradient": delta_gradient,
        "delta_ln_linearized_distance": delta_distance,
        "delta_ln_distance_from_margin_minus_gradient": delta_margin - delta_gradient,
        "identity_residual": delta_distance - (delta_margin - delta_gradient),
        "complete_case_delta_ln_pga_attack_l2": attack_delta,
        "delta_ln_semantic_spectral_norm": spectral_delta,
    }


def build_paired_effects(
    rows: Sequence[dict[str, Any]], seed_summaries: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["cell"].upper(),
            as_int(row, "training_seed"),
            as_int(row, "repeat_index"),
            as_int(row, "dataset_index"),
        ): row
        for row in rows
    }
    seeds = sorted({as_int(row, "training_seed") for row in rows})
    repeats = sorted({as_int(row, "repeat_index") for row in rows})
    summary_lookup = {
        (row["cell"], int(row["training_seed"])): row for row in seed_summaries
    }
    output: list[dict[str, Any]] = []
    for comparison, off_cell, on_cell, objective in COMPARISONS:
        pairwise_common = common_clean_correct_indices(rows, (off_cell, on_cell))
        for seed in seeds:
            repeat_effects: list[dict[str, float | None]] = []
            paired_count = 0
            success_mode_counts = {
                "both_success": 0,
                "off_only_success": 0,
                "on_only_success": 0,
                "neither_success": 0,
            }
            success_mode_repeat_rates: dict[str, list[float]] = {
                key: [] for key in success_mode_counts
            }
            for repeat in repeats:
                identity_diagnostics: list[dict[str, float | None]] = []
                repeat_modes = {key: 0 for key in success_mode_counts}
                for dataset_index in sorted(pairwise_common[seed, repeat]):
                    off_row = lookup[off_cell, seed, repeat, dataset_index]
                    on_row = lookup[on_cell, seed, repeat, dataset_index]
                    off_success = as_bool(off_row, "pga_success")
                    on_success = as_bool(on_row, "pga_success")
                    if off_success and on_success:
                        mode = "both_success"
                    elif off_success:
                        mode = "off_only_success"
                    elif on_success:
                        mode = "on_only_success"
                    else:
                        mode = "neither_success"
                    repeat_modes[mode] += 1
                    success_mode_counts[mode] += 1
                    identity_diagnostics.append(
                        paired_log_identity_diagnostics(off_row, on_row)
                    )
                paired_count += len(identity_diagnostics)
                if not identity_diagnostics:
                    raise ValueError(
                        f"No pairwise common-correct rows for {comparison}/{seed}/{repeat}."
                    )
                for mode, count in repeat_modes.items():
                    success_mode_repeat_rates[mode].append(
                        count / len(identity_diagnostics)
                    )
                repeat_effects.append(
                    {
                        key: _mean_optional(item[key] for item in identity_diagnostics)
                        for key in identity_diagnostics[0]
                    }
                )
            row: dict[str, Any] = {
                "comparison": comparison,
                "objective": objective,
                "training_seed": seed,
                "channel_repeats_averaged": len(repeats),
                "pairwise_common_clean_correct_sample_repeat_pairs": paired_count,
            }
            for mode, count in success_mode_counts.items():
                row[f"pga_{mode}_sample_repeat_pairs"] = count
                row[f"pga_{mode}_rate_repeat_mean"] = statistics.mean(
                    success_mode_repeat_rates[mode]
                )
            for key in repeat_effects[0]:
                row[f"mean_{key}"] = _mean_optional(
                    repeat_effect[key] for repeat_effect in repeat_effects
                )
            off_summary = summary_lookup[off_cell, seed]
            on_summary = summary_lookup[on_cell, seed]
            for scope in ("all", "pairwise_common_clean_correct"):
                for bound in ("lower", "upper"):
                    field = f"pga_operational_auc_normalized_{scope}_{bound}_bound"
                    row[f"delta_{field}"] = float(on_summary[field]) - float(
                        off_summary[field]
                    )
            row["delta_clean_semantic_accuracy"] = float(
                on_summary["clean_semantic_accuracy_repeat_mean"]
            ) - float(off_summary["clean_semantic_accuracy_repeat_mean"])
            output.append(row)
    return output


def _pooled_sd(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    numerator = (len(left) - 1) * statistics.variance(left) + (
        len(right) - 1
    ) * statistics.variance(right)
    return math.sqrt(numerator / (len(left) + len(right) - 2))


def greedy_margin_match(
    off_rows: Sequence[dict[str, Any]],
    on_rows: Sequence[dict[str, Any]],
    *,
    caliper_scale: float = 0.2,
) -> list[tuple[dict[str, Any], dict[str, Any], float, float]]:
    """Globally greedy, no-replacement nearest-neighbor log-margin matching."""

    eligible_off = [
        row
        for row in off_rows
        if as_bool(row, "clean_semantic_correct")
        and as_float(row, "semantic_failure_margin") > 0
    ]
    eligible_on = [
        row
        for row in on_rows
        if as_bool(row, "clean_semantic_correct")
        and as_float(row, "semantic_failure_margin") > 0
    ]
    if not eligible_off or not eligible_on:
        return []
    off_log = [math.log(as_float(row, "semantic_failure_margin")) for row in eligible_off]
    on_log = [math.log(as_float(row, "semantic_failure_margin")) for row in eligible_on]
    pooled = _pooled_sd(off_log, on_log)
    caliper = caliper_scale * pooled
    tolerance = max(caliper, 1e-12)
    candidates = sorted(
        (
            abs(off_log[left_index] - on_log[right_index]),
            left_index,
            right_index,
        )
        for left_index in range(len(eligible_off))
        for right_index in range(len(eligible_on))
        if abs(off_log[left_index] - on_log[right_index]) <= tolerance
    )
    used_off: set[int] = set()
    used_on: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    for distance, left_index, right_index in candidates:
        if left_index in used_off or right_index in used_on:
            continue
        used_off.add(left_index)
        used_on.add(right_index)
        matches.append(
            (
                eligible_off[left_index],
                eligible_on[right_index],
                distance,
                caliper,
            )
        )
    return matches


def build_margin_matches(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            row["cell"].upper(),
            as_int(row, "training_seed"),
            as_int(row, "repeat_index"),
            as_int(row, "class_label"),
        ].append(row)
    seeds = sorted({as_int(row, "training_seed") for row in rows})
    repeats = sorted({as_int(row, "repeat_index") for row in rows})
    classes = sorted({as_int(row, "class_label") for row in rows})
    output: list[dict[str, Any]] = []
    for comparison, off_cell, on_cell, objective in COMPARISONS:
        for seed in seeds:
            for repeat in repeats:
                for class_label in classes:
                    matches = greedy_margin_match(
                        grouped[off_cell, seed, repeat, class_label],
                        grouped[on_cell, seed, repeat, class_label],
                    )
                    for match_index, (off_row, on_row, distance, caliper) in enumerate(matches):
                        identity_diagnostics = paired_log_identity_diagnostics(
                            off_row, on_row
                        )
                        output.append(
                            {
                                "comparison": comparison,
                                "objective": objective,
                                "training_seed": seed,
                                "repeat_index": repeat,
                                "class_label": class_label,
                                "stratum_match_index": match_index,
                                "off_cell": off_cell,
                                "on_cell": on_cell,
                                "off_dataset_index": as_int(off_row, "dataset_index"),
                                "on_dataset_index": as_int(on_row, "dataset_index"),
                                "off_ln_margin": math.log(as_float(off_row, "semantic_failure_margin")),
                                "on_ln_margin": math.log(as_float(on_row, "semantic_failure_margin")),
                                "absolute_ln_margin_distance": distance,
                                "caliper_ln_margin": caliper,
                                **identity_diagnostics,
                            }
                        )
    return output


def summarize_margin_matching(
    match_rows: Sequence[dict[str, Any]],
    *,
    source_rows: Sequence[dict[str, Any]],
    minimum_pairs: int = 200,
    smd_limit: float = 0.1,
) -> dict[str, dict[str, Any]]:
    """Audit post-treatment log-margin matching without causal language."""

    expected_seeds = sorted({as_int(row, "training_seed") for row in source_rows})
    expected_repeats = sorted({as_int(row, "repeat_index") for row in source_rows})

    def smd_for(selected: Sequence[dict[str, Any]]) -> float | None:
        if not selected:
            return None
        off = [float(row["off_ln_margin"]) for row in selected]
        on = [float(row["on_ln_margin"]) for row in selected]
        pooled = _pooled_sd(off, on)
        if pooled == 0:
            return 0.0 if statistics.mean(on) == statistics.mean(off) else math.inf
        return (statistics.mean(on) - statistics.mean(off)) / pooled

    output: dict[str, dict[str, Any]] = {}
    for comparison, off_cell, on_cell, _objective in COMPARISONS:
        selected = [row for row in match_rows if row["comparison"] == comparison]
        eligible_by_stratum: dict[tuple[str, int, int, int], int] = defaultdict(int)
        for row in source_rows:
            cell = row["cell"].upper()
            if cell not in {off_cell, on_cell}:
                continue
            if not as_bool(row, "clean_semantic_correct"):
                continue
            if as_float(row, "semantic_failure_margin") <= 0:
                continue
            eligible_by_stratum[
                cell,
                as_int(row, "training_seed"),
                as_int(row, "repeat_index"),
                as_int(row, "class_label"),
            ] += 1
        seed_audit: dict[str, Any] = {}
        all_seed_balance_qualified = True
        all_repeat_coverage = True
        for seed in expected_seeds:
            seed_selected = [
                row for row in selected if int(row["training_seed"]) == seed
            ]
            eligible_off = sum(
                count
                for (cell, row_seed, _repeat, _label), count in eligible_by_stratum.items()
                if cell == off_cell and row_seed == seed
            )
            eligible_on = sum(
                count
                for (cell, row_seed, _repeat, _label), count in eligible_by_stratum.items()
                if cell == on_cell and row_seed == seed
            )
            capacity = sum(
                min(
                    eligible_by_stratum[off_cell, seed, repeat, label],
                    eligible_by_stratum[on_cell, seed, repeat, label],
                )
                for repeat in expected_repeats
                for label in range(10)
            )
            covered_repeats = sorted(
                {int(row["repeat_index"]) for row in seed_selected}
            )
            repeat_coverage = covered_repeats == expected_repeats
            seed_smd = smd_for(seed_selected)
            seed_balance = seed_smd is not None and abs(seed_smd) < smd_limit
            all_seed_balance_qualified = all_seed_balance_qualified and seed_balance
            all_repeat_coverage = all_repeat_coverage and repeat_coverage
            seed_audit[str(seed)] = {
                "eligible_off": eligible_off,
                "eligible_on": eligible_on,
                "eligible_pair_capacity_within_class_repeat": capacity,
                "matched_pairs": len(seed_selected),
                "match_rate_of_pair_capacity": (
                    len(seed_selected) / capacity if capacity else 0.0
                ),
                "covered_repeats": covered_repeats,
                "all_expected_repeats_covered": repeat_coverage,
                "matched_log_margin_smd": seed_smd,
                "absolute_smd_below_limit": seed_balance,
            }

        overall_smd = smd_for(selected)
        total_capacity = sum(
            int(audit["eligible_pair_capacity_within_class_repeat"])
            for audit in seed_audit.values()
        )
        seed_effects: dict[str, list[float]] = defaultdict(list)
        for seed in expected_seeds:
            by_repeat = [row for row in selected if int(row["training_seed"]) == seed]
            repeats = sorted({int(row["repeat_index"]) for row in by_repeat})
            for field in (
                "delta_ln_gradient",
                "delta_ln_linearized_distance",
                "complete_case_delta_ln_pga_attack_l2",
                "delta_ln_semantic_spectral_norm",
            ):
                repeat_means = [
                    _mean_optional(
                        row[field]
                        for row in by_repeat
                        if int(row["repeat_index"]) == repeat
                    )
                    for repeat in repeats
                ]
                mean = _mean_optional(repeat_means)
                if mean is not None:
                    seed_effects[field].append(mean)
        effect_summary: dict[str, Any] = {}
        for field, values in seed_effects.items():
            effect_summary[f"{field}_mean_across_training_seeds"] = statistics.mean(values)
            effect_summary[f"{field}_sd_across_training_seeds"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        output[comparison] = {
            "analysis_type": "post-treatment log-margin matched sensitivity analysis",
            "causal_interpretation": False,
            "eligible_off": sum(
                count
                for (cell, _seed, _repeat, _label), count in eligible_by_stratum.items()
                if cell == off_cell
            ),
            "eligible_on": sum(
                count
                for (cell, _seed, _repeat, _label), count in eligible_by_stratum.items()
                if cell == on_cell
            ),
            "eligible_pair_capacity_within_class_seed_repeat": total_capacity,
            "matched_sample_repeat_pairs": len(selected),
            "match_rate_of_pair_capacity": (
                len(selected) / total_capacity if total_capacity else 0.0
            ),
            "matched_log_margin_smd": overall_smd,
            "per_training_seed_audit": seed_audit,
            "qualification_rule": (
                f"pairs >= {minimum_pairs}; overall and every-seed absolute matched "
                f"log-margin SMD < {smd_limit}; exactly three training seeds; every "
                "seed covers every expected channel repeat"
            ),
            "qualified": (
                len(selected) >= minimum_pairs
                and overall_smd is not None
                and abs(overall_smd) < smd_limit
                and len(expected_seeds) == 3
                and all(len([row for row in selected if int(row["training_seed"]) == seed]) > 0 for seed in expected_seeds)
                and all_repeat_coverage
                and all_seed_balance_qualified
            ),
            "inference_unit": "training seed; channel repeats averaged within seed",
            **effect_summary,
        }
    return output


def _seed_metric_summary(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return {
        "mean": statistics.mean(values) if values else None,
        "sd_across_training_seeds": statistics.stdev(values) if len(values) > 1 else None,
        "training_seeds": len(values),
    }


def build_summary(
    metadata: dict[str, Any],
    seed_summaries: Sequence[dict[str, Any]],
    paired_effects: Sequence[dict[str, Any]],
    matching: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    system_results: dict[str, Any] = {}
    for cell in CELLS:
        selected = [row for row in seed_summaries if row["cell"] == cell]
        system_results[cell] = {
            field: _seed_metric_summary(selected, field)
            for field in (
                "clean_semantic_accuracy_repeat_mean",
                "pga_success_rate_among_all_clean_correct_repeat_mean",
                "pga_success_rate_pairwise_common_repeat_mean",
                "pga_operational_auc_normalized_all_lower_bound",
                "pga_operational_auc_normalized_all_upper_bound",
                "pga_operational_auc_normalized_pairwise_common_clean_correct_lower_bound",
                "pga_operational_auc_normalized_pairwise_common_clean_correct_upper_bound",
                "pga_operational_auc_normalized_four_cell_common_clean_correct_lower_bound",
                "pga_operational_auc_normalized_four_cell_common_clean_correct_upper_bound",
                "semantic_failure_margin_geomean_repeat_mean",
                "semantic_margin_gradient_l2_geomean_repeat_mean",
                "semantic_linearized_distance_l2_geomean_repeat_mean",
                "semantic_spectral_norm_geomean_repeat_mean",
                "spectral_convergence_rate_repeat_mean",
                "pearson_ln_linear_distance_vs_ln_pga_l2_fisher_z_repeat_mean",
            )
        }
    paired_summary: dict[str, Any] = {}
    for comparison, *_ in COMPARISONS:
        selected = [row for row in paired_effects if row["comparison"] == comparison]
        paired_summary[comparison] = {
            field: _seed_metric_summary(selected, field)
            for field in (
                "mean_delta_ln_margin",
                "mean_delta_ln_gradient",
                "mean_delta_ln_linearized_distance",
                "mean_complete_case_delta_ln_pga_attack_l2",
                "mean_delta_ln_semantic_spectral_norm",
                "pga_both_success_rate_repeat_mean",
                "pga_off_only_success_rate_repeat_mean",
                "pga_on_only_success_rate_repeat_mean",
                "pga_neither_success_rate_repeat_mean",
                "delta_pga_operational_auc_normalized_all_lower_bound",
                "delta_pga_operational_auc_normalized_all_upper_bound",
                "delta_pga_operational_auc_normalized_pairwise_common_clean_correct_lower_bound",
                "delta_pga_operational_auc_normalized_pairwise_common_clean_correct_upper_bound",
            )
        }
    return {
        "status": "completed",
        "input_validation": metadata,
        "endpoint": "shared CIFAR-10 semantic misclassification endpoint",
        "pga_empirical_bound_policy_not_certified": {
            "clean_semantic_error": 0.0,
            "clean_correct_and_pga_success": "PGA attack power per channel use",
            "clean_correct_and_pga_unsuccessful_lower_bound": (
                "treated as non-robust at every positive plotted budget; deliberately conservative"
            ),
            "clean_correct_and_pga_unsuccessful_upper_bound": (
                "treated as surviving every plotted budget (infinity); optimistic right-censored bound"
            ),
            "certified_robustness": False,
        },
        "pga_empirical_robust_accuracy_power_grid_db": [float(value) for value in POWER_GRID_DB],
        "operational_auc_and_retention": (
            "descriptive summaries on the fixed [-40, 0] dB grid; AUC uses a "
            "trapezoidal approximation to stepwise empirical curves"
        ),
        "primary_within_task_scope": "pairwise_common_clean_correct",
        "four_cell_scope": "cross-task descriptive only",
        "inference_unit": "training seed; all channel repeats averaged within seed",
        "pga_estimated_empirical_system_results": system_results,
        "paired_local_identity_diagnostics": paired_summary,
        "post_treatment_margin_matched_sensitivity": matching,
        "interpretation_limits": [
            "PGA is retained as the single attack solver; solver-optimality is not established.",
            "The lower/upper PGA envelopes are conservative/optimistic empirical conventions, not certified bounds on true robustness.",
            "Jacobian and margin-gradient quantities are local, not a global Lipschitz constant.",
            "The log-distance equation is an algebraic identity, not a causal mechanism decomposition.",
            "Margin matching conditions on a post-training variable and is sensitivity analysis, not causal adjustment.",
            "Cross-task semantic evaluation uses a shared failure event but different system mappings.",
            "The four-cell common-correct scope is descriptive; within-task contrasts use pairwise common-correct samples.",
            "Only three independent training seeds are available.",
        ],
    }


def _across_curve_lookup(curve_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        if row["aggregation_level"] == "across_training_seeds":
            result[row["cell"], row["sample_scope"]].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: float(row["attack_power_per_use_db"]))
    return result


def draw_figures(
    rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    seed_summaries: Sequence[dict[str, Any]],
    paired_effects: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    colors = {"R0": "tab:orange", "R1": "tab:blue", "C0": "tab:red", "C1": "tab:green"}
    lookup = _across_curve_lookup(curve_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, (objective, cells) in zip(
        axes, (("Reconstruction", ("R0", "R1")), ("Classification", ("C0", "C1")))
    ):
        for cell in cells:
            for scope, line_style, scope_label in (
                ("all", "-", "all samples"),
                (
                    "pairwise_common_clean_correct",
                    "--",
                    "conditional: pairwise common clean-correct",
                ),
            ):
                selected = lookup[cell, scope]
                x = np.asarray([float(row["attack_power_per_use_db"]) for row in selected])
                lower = np.asarray(
                    [float(row["pga_empirical_robust_accuracy_lower_bound"]) for row in selected]
                )
                upper = np.asarray(
                    [float(row["pga_empirical_robust_accuracy_upper_bound"]) for row in selected]
                )
                if np.allclose(lower, upper, rtol=1e-9, atol=1e-12):
                    axis.plot(
                        x,
                        upper,
                        line_style,
                        color=colors[cell],
                        label=f"{cell}: {scope_label}",
                    )
                else:
                    axis.fill_between(x, lower, upper, color=colors[cell], alpha=0.08)
                    axis.plot(
                        x,
                        upper,
                        line_style,
                        color=colors[cell],
                        label=f"{cell}: {scope_label} (upper)",
                    )
                    axis.plot(
                        x,
                        lower,
                        ":",
                        color=colors[cell],
                        alpha=0.8,
                        label=f"{cell}: {scope_label} (lower)",
                    )
        axis.set_title(f"{objective}: PGA-estimated empirical envelope")
        axis.set_xlabel("PGA power per channel use (dB)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=6.5)
    axes[0].set_ylabel("PGA-estimated empirical robust accuracy (not certified)")
    fig.tight_layout()
    fig.savefig(output / "robust_accuracy_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fields = (
        ("mean_delta_ln_margin", r"$\Delta\,\ln$ margin"),
        ("mean_delta_ln_gradient", r"$\Delta\,\ln$ gradient"),
        ("mean_delta_ln_linearized_distance", r"$\Delta\,\ln$ linearized distance"),
        (
            "mean_complete_case_delta_ln_pga_attack_l2",
            "$\\Delta\\,\\ln$ PGA L2\n(complete cases)",
        ),
    )
    for axis, (comparison, *_rest) in zip(axes, COMPARISONS):
        selected = [row for row in paired_effects if row["comparison"] == comparison]
        x = np.arange(len(fields))
        for index, row in enumerate(selected):
            values = [
                float(row[field]) if row.get(field) not in (None, "") else math.nan
                for field, _ in fields
            ]
            offset = (index - (len(selected) - 1) / 2) * 0.12
            axis.scatter(
                x + offset,
                values,
                s=42,
                alpha=0.75,
                label=f"seed {row['training_seed']}",
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, [label for _, label in fields], rotation=20, ha="right")
        axis.set_title(comparison.replace("_minus_", " - "))
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("On - off (identity/sensitivity; natural-log units)")
    fig.suptitle("Local distance identity diagnostics (not a causal decomposition)")
    fig.tight_layout()
    fig.savefig(output / "margin_gradient_identity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(CELLS))
    distributions: list[list[float]] = []
    for cell in CELLS:
        distributions.append(
            [
                math.log(as_float(row, "semantic_linearized_distance_l2"))
                for row in rows
                if row["cell"].upper() == cell
                and as_bool(row, "clean_semantic_correct")
                and as_float(row, "semantic_linearized_distance_l2") > 0
            ]
        )
    axis.boxplot(distributions, positions=positions, widths=0.55, showfliers=False)
    for position, cell in zip(positions, CELLS):
        seed_medians = []
        for seed in sorted({as_int(row, "training_seed") for row in rows}):
            selected = [
                math.log(as_float(row, "semantic_linearized_distance_l2"))
                for row in rows
                if row["cell"].upper() == cell
                and as_int(row, "training_seed") == seed
                and as_bool(row, "clean_semantic_correct")
                and as_float(row, "semantic_linearized_distance_l2") > 0
            ]
            seed_medians.append(statistics.median(selected))
        axis.scatter([position] * len(seed_medians), seed_medians, s=45, color=colors[cell], zorder=3)
    axis.set_xticks(positions, CELLS)
    axis.set_ylabel("ln linearized boundary distance (clean-correct only)")
    axis.set_title(
        "Pooled repeated measures (not independent); dots = training-seed medians"
    )
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "linearized_distance_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    for position, cell in zip(positions, CELLS):
        values = [
            math.log(float(row["semantic_spectral_norm_geomean_repeat_mean"]))
            for row in seed_summaries
            if row["cell"] == cell
        ]
        axis.scatter([position] * len(values), values, s=55, color=colors[cell])
        axis.plot([position - 0.16, position + 0.16], [statistics.mean(values)] * 2, color="black")
    axis.set_xticks(positions, CELLS)
    axis.set_ylabel(
        "ln semantic Jacobian spectral norm\n"
        "(mean over repeats of per-repeat geometric means)"
    )
    axis.set_title(
        "Implicit local estimates (not global G; compare within task only)\n"
        "Dots = training seeds; black line = seed mean"
    )
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "semantic_spectral_norm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, cells, title in zip(
        axes,
        (("R0", "R1"), ("C0", "C1")),
        (
            "Reconstruction systems: PGA empirical complete cases",
            "Classification systems: PGA empirical complete cases",
        ),
    ):
        for cell in cells:
            selected = [
                row
                for row in rows
                if row["cell"].upper() == cell
                and as_bool(row, "clean_semantic_correct")
                and as_bool(row, "pga_success")
                and as_float(row, "semantic_linearized_distance_l2") > 0
                and as_float(row, "pga_attack_power_total_l2_sq") > 0
            ]
            # Deterministic thinning keeps vector output and rendering manageable.
            selected = selected[:: max(1, len(selected) // 3000)]
            x = [math.log(as_float(row, "semantic_linearized_distance_l2")) for row in selected]
            y = [0.5 * math.log(as_float(row, "pga_attack_power_total_l2_sq")) for row in selected]
            axis.scatter(x, y, s=7, alpha=0.12, color=colors[cell], label=cell)
        axis.set_xlabel("ln linearized distance (clean-correct)")
        axis.set_ylabel("ln PGA attack L2 (successful complete cases)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("PGA step size = 0.1; horizontal bands reflect attack quantization")
    fig.tight_layout()
    fig.savefig(output / "linearized_distance_vs_pga.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return command


def main() -> int:
    args = parser().parse_args()
    rows, metadata = load_validated_diagnostics(args.input_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    curve_rows, seed_summaries = build_robust_curve_outputs(rows)
    paired_effects = build_paired_effects(rows, seed_summaries)
    margin_matches = build_margin_matches(rows)
    matching_summary = summarize_margin_matching(
        margin_matches, source_rows=rows
    )
    summary = build_summary(metadata, seed_summaries, paired_effects, matching_summary)
    write_csv(output / "seed_summary.csv", seed_summaries)
    write_csv(output / "paired_effects.csv", paired_effects)
    write_csv(output / "robust_accuracy_curves.csv", curve_rows)
    write_csv(output / "margin_matched.csv", margin_matches)
    write_json(output / "analysis_summary.json", summary)
    draw_figures(rows, curve_rows, seed_summaries, paired_effects, output)
    print(f"mechanism analysis: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
