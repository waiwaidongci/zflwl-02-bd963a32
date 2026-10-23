#!/usr/bin/env python3
"""Reference solver for a two-campaign latent-cure assignment."""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {
    "schema",
    "gas_constant_j_mol_k",
    "activation_kj_mol",
    "runs",
    "fit_tolerance",
}
RUN_KEYS = {"id", "sample", "profile", "conversion"}
RUN_ID = re.compile(r"C-[A-Z0-9]{4}\Z")
SAMPLE_ID = re.compile(r"S-[A-Z0-9]{4}\Z")
LOG_PREFACTOR_LOW = 4.0
LOG_PREFACTOR_HIGH = 14.0


class Refused(ValueError):
    pass


def _number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise Refused(f"{label} is outside its allowed range")
    return number


def parse_input(path: Path) -> tuple[float, float, float, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level keys do not match latent-cure-v1")
    if payload.get("schema") != "latent-cure-v1":
        raise Refused("unsupported schema")
    gas_constant = _number(
        payload.get("gas_constant_j_mol_k"), "gas_constant_j_mol_k", 8.0, 9.0
    )
    activation = _number(
        payload.get("activation_kj_mol"), "activation_kj_mol", 40.0, 220.0
    )
    fit_tolerance = _number(payload.get("fit_tolerance"), "fit_tolerance", 1e-6, 1e-3)
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 8:
        raise Refused("runs must contain exactly 8 entries")

    runs: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    sample_counts: dict[str, int] = {}
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict) or set(raw) != RUN_KEYS:
            raise Refused(f"run {index} has an invalid key set")
        run_id = raw.get("id")
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise Refused(f"run {index} has an invalid id")
        if run_id in seen_runs:
            raise Refused("run ids must be unique")
        seen_runs.add(run_id)
        sample = raw.get("sample")
        if not isinstance(sample, str) or SAMPLE_ID.fullmatch(sample) is None:
            raise Refused(f"run {index} has an invalid sample")
        sample_counts[sample] = sample_counts.get(sample, 0) + 1
        raw_profile = raw.get("profile")
        if not isinstance(raw_profile, list) or not 3 <= len(raw_profile) <= 5:
            raise Refused(f"{run_id} profile has an invalid length")
        profile: list[tuple[float, float]] = []
        for point_index, point in enumerate(raw_profile):
            if not isinstance(point, list) or len(point) != 2:
                raise Refused(f"{run_id} profile point {point_index} is invalid")
            time = _number(point[0], "profile time", 0.0, 600.0)
            temperature = _number(point[1], "profile temperature", 300.0, 450.0)
            profile.append((time, temperature))
        if profile[0][0] != 0.0:
            raise Refused(f"{run_id} profile must start at zero")
        if any(profile[i][0] <= profile[i - 1][0] for i in range(1, len(profile))):
            raise Refused(f"{run_id} profile times must strictly increase")
        conversion = _number(raw.get("conversion"), f"{run_id} conversion", 0.01, 0.99)
        runs.append(
            {
                "id": run_id,
                "sample": sample,
                "profile": profile,
                "conversion": conversion,
            }
        )

    if len(sample_counts) != 4 or any(count != 2 for count in sample_counts.values()):
        raise Refused("exactly four samples must each occur twice")
    return gas_constant, activation, fit_tolerance, runs


def _temperature(profile: list[tuple[float, float]], time: float) -> float:
    for index in range(len(profile) - 1):
        start, start_temp = profile[index]
        end, end_temp = profile[index + 1]
        if start <= time <= end:
            return start_temp + (time - start) / (end - start) * (end_temp - start_temp)
    return profile[-1][1]


def _conversion(
    conversion_parameter: float, initial_intermediate: float, rate_time: float
) -> float:
    del conversion_parameter
    return 1.0 - math.exp(-rate_time) * (
        1.0 + (1.0 - initial_intermediate) * rate_time
    )


def _integrated_rate(
    profile: list[tuple[float, float]], gas_constant: float, activation: float, log_a: float
) -> float:
    total = 0.0
    for index in range(len(profile) - 1):
        start, start_temp = profile[index]
        end, end_temp = profile[index + 1]
        pieces = 24
        step = (end - start) / pieces
        for piece in range(pieces):
            left = _temperature(profile, start + piece * step)
            middle = _temperature(profile, start + (piece + 0.5) * step)
            right = _temperature(profile, start + (piece + 1) * step)
            exponent = -activation * 1000.0 / gas_constant
            total += step * (
                10.0**log_a * math.exp(exponent / left)
                + 4.0 * 10.0**log_a * math.exp(exponent / middle)
                + 10.0**log_a * math.exp(exponent / right)
            ) / 6.0
    return total


def _infer_seed(conversion: float, rate_time: float) -> float | None:
    if rate_time <= 1e-12 or rate_time > 40.0:
        return None
    try:
        value = ((1.0 - conversion) * math.exp(rate_time) - 1.0) / rate_time
    except OverflowError:
        return None
    seed = 1.0 - value
    if not math.isfinite(seed) or not -0.02 <= seed <= 0.52:
        return None
    return seed


def _evaluate(
    log_a: float,
    gas_constant: float,
    activation: float,
    runs: list[dict[str, Any]],
    sample_names: list[str],
) -> tuple[float, list[int], list[float]]:
    rate_times = [
        _integrated_rate(run["profile"], gas_constant, activation, log_a)
        for run in runs
    ]
    seeds = [_infer_seed(run["conversion"], rt) for run, rt in zip(runs, rate_times)]
    if any(seed is None for seed in seeds):
        return math.inf, [], []

    sample_runs = {
        sample: [index for index, run in enumerate(runs) if run["sample"] == sample]
        for sample in sample_names
    }
    best: tuple[float, list[int], list[float]] | None = None
    for first_group in itertools.combinations(sample_names, 2):
        first = set(first_group)
        groups = [first, set(sample_names) - first]
        assignment = [0 if sample in first else 1 for sample in sample_names]
        campaign_seeds = []
        valid = True
        for group in groups:
            values = [
                seeds[index]
                for sample in group
                for index in sample_runs[sample]
            ]
            seed = math.fsum(values) / len(values)
            if not 0.02 <= seed <= 0.48:
                valid = False
            campaign_seeds.append(seed)
        if not valid or abs(campaign_seeds[0] - campaign_seeds[1]) < 0.02:
            continue
        errors = []
        for index, run in enumerate(runs):
            campaign = assignment[sample_names.index(run["sample"])]
            predicted = _conversion(
                log_a, campaign_seeds[campaign], rate_times[index]
            )
            errors.append(predicted - run["conversion"])
        sse = math.fsum(error * error for error in errors)
        if best is None or sse < best[0]:
            best = (sse, assignment, campaign_seeds)
    if best is None:
        return math.inf, [], []
    return best


def _golden(
    function: Any, low: float, high: float, iterations: int = 80
) -> tuple[float, float]:
    inverse = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - inverse * (high - low)
    right = low + inverse * (high - low)
    left_value = function(left)
    right_value = function(right)
    for _ in range(iterations):
        if left_value < right_value:
            high = right
            right = left
            right_value = left_value
            left = high - inverse * (high - low)
            left_value = function(left)
        else:
            low = left
            left = right
            left_value = right_value
            right = low + inverse * (high - low)
            right_value = function(right)
    candidate = (low + high) / 2.0
    return candidate, function(candidate)


def fit(
    gas_constant: float,
    activation: float,
    fit_tolerance: float,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_names = sorted({run["sample"] for run in runs})
    coarse = []
    value = LOG_PREFACTOR_LOW
    while value <= LOG_PREFACTOR_HIGH + 1e-12:
        sse, assignment, seeds = _evaluate(
            value, gas_constant, activation, runs, sample_names
        )
        coarse.append((sse, value, assignment, seeds))
        value += 0.01
    coarse.sort(key=lambda item: item[0])
    best = coarse[0]
    low = max(LOG_PREFACTOR_LOW, best[1] - 0.02)
    high = min(LOG_PREFACTOR_HIGH, best[1] + 0.02)
    refined_log_a, _ = _golden(
        lambda candidate: _evaluate(
            candidate, gas_constant, activation, runs, sample_names
        )[0],
        low,
        high,
    )
    sse, assignment, seeds = _evaluate(
        refined_log_a, gas_constant, activation, runs, sample_names
    )
    rms = math.sqrt(sse / len(runs))
    if rms > fit_tolerance:
        raise Refused("latent-cure network cannot reach the requested fit tolerance")

    order = sorted(range(2), key=lambda index: seeds[index])
    remap = {old: new for new, old in enumerate(order)}
    result_map = {
        sample_names[index]: remap[assignment[index]]
        for index in range(len(sample_names))
    }
    return {
        "campaigns": [
            {"initial_intermediate": round(seeds[index], 6)} for index in order
        ],
        "log10_prefactor": round(refined_log_a, 6),
        "rms_conversion": round(rms, 6),
        "sample_campaigns": result_map,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        gas_constant, activation, fit_tolerance, runs = parse_input(source)
        result = fit(gas_constant, activation, fit_tolerance, runs)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
