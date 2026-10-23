from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("DUAL_CURE_ENTRY", "/app/predict.py")
R = 8.314462618
ACTIVATION_KJ = 85.0
TRUE_LOG_A = 8.8
TRUE_SEEDS = [0.08, 0.33]
PROFILES = [
    [[0.0, 380.0], [300.0, 380.0], [600.0, 380.0]],
    [[0.0, 400.0], [240.0, 400.0], [480.0, 400.0]],
    [[0.0, 410.0], [150.0, 410.0], [300.0, 410.0]],
    [[0.0, 390.0], [400.0, 390.0], [500.0, 390.0]],
]
SAMPLES = ["S-K7Q2", "S-M4R8", "S-T9C1", "S-B6N5"]
PROFILE_PAIRS = [(0, 2), (1, 3), (1, 0), (3, 2)]


def _run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _conversion(profile: list[list[float]], seed: float, log_a: float) -> float:
    rate = 10.0**log_a * math.exp(
        -ACTIVATION_KJ * 1000.0 / (R * profile[-1][1])
    )
    rate_time = rate * profile[-1][0]
    return 1.0 - math.exp(-rate_time) * (
        1.0 + (1.0 - seed) * rate_time
    )


def _campaign_map(rotated: bool = False) -> dict[str, int]:
    groups = [1, 1, 0, 0] if rotated else [0, 0, 1, 1]
    return dict(zip(SAMPLES, groups))


def _payload(rotated: bool = False) -> dict[str, object]:
    campaign_map = _campaign_map(rotated)
    runs = []
    for sample_index, sample in enumerate(SAMPLES):
        seed = TRUE_SEEDS[campaign_map[sample]]
        pair = PROFILE_PAIRS[(sample_index + (1 if rotated else 0)) % 4]
        for run_number, profile_index in enumerate(pair):
            profile = PROFILES[profile_index]
            suffix = (sample_index * 7 + run_number * 5 + 11 + (17 if rotated else 0)) % 10000
            runs.append(
                {
                    "conversion": round(_conversion(profile, seed, TRUE_LOG_A), 10),
                    "id": f"C-{suffix:04d}",
                    "profile": [[point[0], point[1]] for point in profile],
                    "sample": sample,
                }
            )
    return {
        "activation_kj_mol": ACTIVATION_KJ,
        "fit_tolerance": 0.000001,
        "gas_constant_j_mol_k": R,
        "runs": runs,
        "schema": "latent-cure-v1",
    }


def _endpoint_rms(result: dict[str, object], payload: dict[str, object]) -> float:
    errors = []
    for run in payload["runs"]:
        campaign = result["sample_campaigns"][run["sample"]]
        seed = result["campaigns"][campaign]["initial_intermediate"]
        predicted = _conversion(run["profile"], seed, result["log10_prefactor"])
        errors.append((predicted - run["conversion"]) ** 2)
    return math.sqrt(math.fsum(errors) / len(errors))


@pytest.fixture(scope="session")
def main_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("latent-cure-main")
    source = root / "runs.json"
    output = root / "result.json"
    payload = _payload()
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    return {
        "source": source,
        "output": output,
        "payload": payload,
        "completed": completed,
        "result": result,
        "truth": _campaign_map(),
    }


@pytest.fixture(scope="session")
def rotated_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("latent-cure-rotated")
    source = root / "runs.json"
    output = root / "result.json"
    payload = _payload(rotated=True)
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    return {
        "source": source,
        "output": output,
        "payload": payload,
        "completed": completed,
        "result": result,
        "truth": _campaign_map(rotated=True),
    }


def test_gate_hidden_assignment_closes_all_endpoints(
    main_case: dict[str, object],
) -> None:
    """题面：「拟合把全部记录的终点转化率压到 `fit_tolerance` 的均方根误差以内后」"""
    # CURE-CONTEXT CURE-ENV CURE-CLI CURE-RMS CURE-OUTPUT CURE-SENSING
    completed = main_case["completed"]
    result = main_case["result"]
    payload = main_case["payload"]
    assert completed.returncode == 0
    assert result["sample_campaigns"] == main_case["truth"]
    assert result["rms_conversion"] <= payload["fit_tolerance"]
    assert _endpoint_rms(result, payload) <= payload["fit_tolerance"]


def test_unlabelled_samples_recover_hidden_campaign_partition(
    main_case: dict[str, object],
) -> None:
    """题面：「样品编号本身不透露批次。」"""
    # CURE-MAP
    completed = main_case["completed"]
    result = main_case["result"]
    assert completed.returncode == 0
    assert set(result["sample_campaigns"]) == set(SAMPLES)
    assert result["sample_campaigns"] == _campaign_map()
    assert len(set(result["sample_campaigns"].values())) == 2


def test_shared_rate_recovers_prefactor(main_case: dict[str, object]) -> None:
    """题面：「两步反应共用同一个速率常数 `k = A * exp(-E/(R*T))`」"""
    # CURE-RATE CURE-STATE
    completed = main_case["completed"]
    result = main_case["result"]
    assert completed.returncode == 0
    assert result["log10_prefactor"] == pytest.approx(TRUE_LOG_A, abs=0.0001)


def test_latent_intermediate_state_is_resolved_per_campaign(
    main_case: dict[str, object],
) -> None:
    """题面：「`initial_intermediate` 的取值范围是 0.02 到 0.48。」"""
    # CURE-LATENT CURE-SEED-RANK CURE-CAMPAIGN-FIELDS
    completed = main_case["completed"]
    result = main_case["result"]
    seeds = [item["initial_intermediate"] for item in result["campaigns"]]
    assert completed.returncode == 0
    assert seeds == sorted(seeds)
    assert seeds[0] == pytest.approx(TRUE_SEEDS[0], abs=0.0001)
    assert seeds[1] == pytest.approx(TRUE_SEEDS[1], abs=0.0001)


def test_rotated_assignment_still_uses_seed_order_for_campaign_numbers(
    rotated_case: dict[str, object],
) -> None:
    """题面：「`campaigns` 按 `initial_intermediate` 从小到大排列」"""
    # CURE-ORDER
    completed = rotated_case["completed"]
    result = rotated_case["result"]
    seeds = [item["initial_intermediate"] for item in result["campaigns"]]
    assert completed.returncode == 0
    assert result["sample_campaigns"] == rotated_case["truth"]
    assert seeds == sorted(seeds)
    assert seeds[0] == pytest.approx(TRUE_SEEDS[0], abs=0.0001)


def test_extra_top_level_key_is_refused_without_output(tmp_path: Path) -> None:
    """题面：「JSON 顶层恰好包含五个字段。」"""
    # CURE-ROOT CURE-TOPFIELDS CURE-TOPFIELDS-02 CURE-REFUSE CURE-STRUCTURE
    payload = _payload()
    payload["operator"] = "unexpected"
    source = tmp_path / "extra.json"
    output = tmp_path / "extra-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_seven_runs_are_refused(tmp_path: Path) -> None:
    """题面：「`runs` 固定为八条记录。」"""
    # CURE-RUNCOUNT
    payload = _payload()
    payload["runs"] = payload["runs"][:7]
    source = tmp_path / "run-count.json"
    output = tmp_path / "run-count-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_sample_observed_once_is_refused(tmp_path: Path) -> None:
    """题面：「四个样品各出现两次。」"""
    # CURE-SAMPLE
    payload = _payload()
    payload["runs"] = [
        run for run in payload["runs"] if run["sample"] != SAMPLES[-1]
    ] + [
        run for run in payload["runs"] if run["sample"] == SAMPLES[-1]
    ][:1]
    source = tmp_path / "sample-count.json"
    output = tmp_path / "sample-count-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_nonmonotone_profile_time_is_refused(tmp_path: Path) -> None:
    """题面：「第一点时间为 0。」"""
    # CURE-PROFILE
    payload = _payload()
    payload["runs"][0]["profile"][1][0] = 0.0
    source = tmp_path / "profile.json"
    output = tmp_path / "profile-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_conversion_is_refused_as_non_numeric(tmp_path: Path) -> None:
    """题面：「数组中的布尔值不算数值。」"""
    # CURE-NUMBER
    payload = _payload()
    payload["runs"][2]["conversion"] = True
    source = tmp_path / "boolean.json"
    output = tmp_path / "boolean-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_wrong_schema_is_refused(tmp_path: Path) -> None:
    """题面：「`schema` 写作 `\"latent-cure-v1\"`。」"""
    # CURE-SCHEMA
    payload = _payload()
    payload["schema"] = "dual-cure-v3"
    source = tmp_path / "schema.json"
    output = tmp_path / "schema-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_activation_above_public_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`activation_kj_mol` 沿用上一轮质检结果，只接受 40 至 220 kJ/mol 的登记值。」"""
    # CURE-ACTIVATION
    payload = _payload()
    payload["activation_kj_mol"] = 221.0
    source = tmp_path / "activation.json"
    output = tmp_path / "activation-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_fit_tolerance_above_public_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`fit_tolerance` 接受 0.000001 至 0.001 的登记值。」"""
    # CURE-FIT-RANGE CURE-PUBLIC-REFUSE
    payload = _payload()
    payload["fit_tolerance"] = 0.0011
    source = tmp_path / "tolerance.json"
    output = tmp_path / "tolerance-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_output_uses_compact_sorted_json_bytes(main_case: dict[str, object]) -> None:
    """题面：「文件写成无空白 JSON。」"""
    # CURE-FORMAT
    output = main_case["output"]
    completed = main_case["completed"]
    raw = output.read_bytes()
    canonical = (
        json.dumps(
            json.loads(raw),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )
    assert completed.returncode == 0
    assert raw == canonical


def test_repeated_fit_produces_identical_output(
    main_case: dict[str, object], tmp_path: Path
) -> None:
    """题面：「同一份输入再次运行时字节应当一致」"""
    # CURE-REPEAT CURE-NOTESTS
    second = tmp_path / "repeat-result.json"
    repeated = _run(main_case["source"], second)
    assert repeated.returncode == 0
    assert second.read_bytes() == main_case["output"].read_bytes()
