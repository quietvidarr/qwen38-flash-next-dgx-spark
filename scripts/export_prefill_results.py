#!/usr/bin/env python3
"""Export a bounded, public-safe receipt from a completed study. No model imports.

Pass --study with an operator-selected directory to regenerate the two results.
Without --study, validate the committed receipt and its deterministic report.
Private source locations are never inferred or embedded in exported artifacts.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "prefill-screen.json"
REPORT = ROOT / "results" / "prefill-screen.md"
ARMS = ("A", "B", "C")
CHECKPOINT = "55a732e0c4c3d4614bc42b68493bb930d9b02c0a"
BASE = "609ca419abecebdc5a059498a613680bd3aa847f"
PR212 = "f28b49f6e91b1789e25345875d61c5e663c19b1a"
PR252 = "7ebfd533d72b827cd23661e307f8d4eaa0d1fb59"
HASH = re.compile(r"[0-9a-f]{64}\Z")
ROW_FIELDS = (
    "suite", "fixture", "page_state", "max_tokens", "draft", "transport",
    "prompt_tokens", "prompt_sha256", "prompt_token_ids_sha256", "sampling",
    "raw_completion_tokens", "completion_tokens", "terminal_eos", "finish_reason",
    "overshoot_tokens", "ttft_seconds", "elapsed_seconds", "decode_seconds",
    "first_chunk_tokens", "decode_tokens", "decode_tps", "decode_metric_kind",
    "prefill_seconds", "prefill_metric_kind", "prefill_tps_first_sample_inclusive",
    "prompt_tokens_per_ttft_proxy", "draft_tokens", "accepted_tokens", "eligible",
    "output_sha256", "peak_torch_memory_allocated_bytes",
)
STAT_FIELDS = ("prefill_s", "decode_s", "rounds", "drafts", "cached", "min_rows", "drafted", "accepted")
KERNEL_FIELDS = ("enabled", "prompt_dispatch_calls", "prompt_rows", "prompt_pairs")
AHEAD_FIELDS = ("enabled", "submitted", "completed", "busy_skips", "errors", "future_tokens", "row_ids", "pages_advised", "in_flight")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def pick(value, fields):
    return {key: value[key] for key in fields}


def hash_inventory(inventory):
    """Keep every digest without exposing source filesystem names.

    Stable opaque artifact IDs hash each receipt's relative source locator.
    The original locator is NOT exported; duplicate content remains represented.
    """
    result = []
    for locator, digest in sorted(inventory.items()):
        require(HASH.fullmatch(digest), "Invalid source artifact digest")
        result.append({"artifact_id": sha(locator.encode("utf-8")), "sha256": digest})
    require(len({r["artifact_id"] for r in result}) == len(result), "Artifact ID collision")
    return result


def treatment(row):
    result = {}
    for label, module, fields in (
        ("prompt_kernel", "tensorfold.cuda.exl3.kernel212", KERNEL_FIELDS),
        ("selective_read_ahead", "tensorfold.families.qwen4_exp.cuda.ngram_ahead", AHEAD_FIELDS),
    ):
        before = pick(row["treatment_before"][module], fields)
        after = pick(row["treatment_after"][module], fields)
        result[label] = {"before": before, "after": after,
                         "delta": {k: after[k] - before[k] for k in fields if k != "enabled"}}
    return result


def export_row(row, index):
    ids = row["raw_output_token_ids"]
    chunks = row["emission_chunks"]
    require(len(ids) == row["raw_completion_tokens"] == sum(c["n"] for c in chunks), "Emission count mismatch")
    require(row["eligible"] and not row["errors"] and row["overshoot_tokens"] == 0, "Ineligible measured row")
    require(row["engine_stats"]["cached"] == 0, "Measured row reused prefix cache")
    require(row["prompt_tokens"] > 0 and 0 < len(ids) <= row["max_tokens"], "Invalid actual count")
    require(row["first_chunk_tokens"] == chunks[0]["n"], "First chunk count mismatch")
    require(row["decode_tokens"] == len(ids) - chunks[0]["n"], "Decode numerator mismatch")
    require(math.isclose(row["prefill_tps_first_sample_inclusive"], row["prompt_tokens"] / row["prefill_seconds"], rel_tol=1e-12), "Prefill denominator mismatch")
    require(sha(row["output_text"].encode("utf-8")) == row["output_sha256"], "Output hash mismatch")
    if row["decode_tokens"]:
        interval = (chunks[-1]["time_ns"] - chunks[0]["time_ns"]) / 1e9
        require(math.isclose(interval, row["decode_seconds"], rel_tol=1e-12), "Decode interval mismatch")
        require(math.isclose(row["decode_tps"], row["decode_tokens"] / interval, rel_tol=1e-12), "Decode rate mismatch")
    else:
        require(row["decode_tps"] is None and row["decode_seconds"] is None, "Single emission has decode estimate")
    result = pick(row, ROW_FIELDS)
    result["cell"] = index
    result["raw_output_token_ids_sha256"] = sha(canonical(ids))
    result["engine_stats"] = pick(row["engine_stats"], STAT_FIELDS)
    result["process_major_fault_delta"] = row["telemetry_after"]["major_faults"] - row["telemetry_before"]["major_faults"]
    result["treatment"] = treatment(row)
    result["prefix_cache_clear"] = pick(row["cache_eviction"], ("serial_entries", "concurrent_kept_before", "concurrent_kept_after"))
    advice = row["page_advice"]
    result["page_advice"] = None if advice is None else {
        "enabled": advice["advice_enabled"], "cold_state_verified": advice["cold_state_verified"],
        "range_count": len(advice["ranges"]),
        "advised_bytes": sum(r["length"] for r in advice["ranges"]),
        "all_advice_succeeded": all(r["advice_succeeded"] for r in advice["ranges"]),
        "error_count": len(advice["errors"]),
    }
    return result


def export_smoke(row):
    response = row["response"]
    choice = response["choices"][0]
    message = choice["message"]
    native = response["tensorfold"]
    result = pick(row, ("label", "http_status", "elapsed_seconds", "passed"))
    result.update({"finish_reason": choice["finish_reason"],
                   "usage": response["usage"], "engine_stats": pick(native, STAT_FIELDS),
                   "raw_output_token_ids_sha256": sha(canonical(native["token_ids"])),
                   "raw_emitted_tokens": len(native["token_ids"]),
                   "output_sha256": sha((message.get("content") or "").encode("utf-8")),
                   "reasoning_sha256": sha((message.get("reasoning_content") or "").encode("utf-8")),
                   "reasoning_characters": len(message.get("reasoning_content") or "")})
    require(result["passed"] and result["http_status"] == 200, "Smoke failed")
    return result


def comparison(before, after):
    result = pick(before, ("cell", "suite", "fixture", "page_state", "prompt_tokens"))
    result["token_ids_equal"] = before["raw_output_token_ids_sha256"] == after["raw_output_token_ids_sha256"]
    result["output_text_hash_equal"] = before["output_sha256"] == after["output_sha256"]
    for key in ("ttft_seconds", "elapsed_seconds", "prefill_seconds", "prefill_tps_first_sample_inclusive", "prompt_tokens_per_ttft_proxy", "decode_tps"):
        a, b = before[key], after[key]
        result[key] = {"before": a, "after": b, "change_percent": None if a is None or b is None else 100 * (b / a - 1)}
    return result


def public_artifacts():
    """Read existing public assets only. Do not create or edit media or patches."""
    patch_path = "patches/prompt-expert-kernels.patch"
    patch = (ROOT / patch_path).read_bytes()
    patch_sha = sha(patch)
    require(HASH.fullmatch(patch_sha), "Invalid public patch digest")
    changed = re.findall(rb"^diff --git a/(.+?) b/(.+)$", patch, re.M)
    require(len(changed) == 19 and (b"pyproject.toml", b"pyproject.toml") in changed, "Public patch packaging boundary changed")
    require(b"def test_f16_prompt_tiles_give_the_sliced_bits" not in patch and b"def test_exl3_prompt_gemm_tiles_keep_the_bits" not in patch, "Public tests include unrun dense GEMM scope")
    image_path = "results/prefill-screen.png"
    image = (ROOT / image_path).read_bytes()
    require(image[:8] == b"\x89PNG\r\n\x1a\n" and image[12:16] == b"IHDR", "Invalid PNG header")
    width, height = struct.unpack(">II", image[16:24])
    return {
        "winning_B_patch": {"path": patch_path, "sha256": patch_sha, "bytes": len(patch),
                            "files": 19, "measured_runtime_and_test_files": 18,
                            "packaging_only_files": 1,
                            "test_source_note": "The public CUDA test file retains the seven executed expert/prompt cases. Unrun F16/dense GEMM tests from the broader original PR are omitted; this test-only scope reduction does not change runtime code.",
                            "packaging_note": "Post-study pyproject.toml package-data fix includes legacy212 sources. Not a new GPU measurement or new runtime-code treatment.",
                            "basis": "Patch against the public vision-prefix-patched TensorFold base; PR252 is rejected and not shipped.",
                            "public_launch_flag": "--prefill-kernels", "default_enabled": False},
        "image": {"path": image_path, "sha256": sha(image), "bytes": len(image),
                  "width": width, "height": height,
                  "scope": "Minimal A/B 32k/64k cold-advice-unverified table; rounded display values use unrounded source rates for gains."},
    }


def build(study):
    receipts = {}

    def raw(relative, label):
        value = (study / relative).read_bytes()
        receipts[label] = {"sha256": sha(value), "bytes": len(value)}
        return value

    def load(relative, label):
        return json.loads(raw(relative, label))

    analysis = load("analysis/RESULTS.json", "analysis_result")
    panel = load("panel/RESULTS.json", "panel_result")
    block = load("block/RESULTS.json", "block_result")
    integration = load("integration/RESULTS.json", "combined_preparation_result")
    kernel = load("kernel212/RESULTS.json", "prompt_kernel_preparation_result")
    ahead = load("readahead252/RESULTS.json", "read_ahead_preparation_result")
    restored = load("restoration/RESULTS.json", "restoration_result")
    current = load("restoration/runs/final-current-check.json", "restoration_final_check")
    block_protocol = load("protocol.json", "block_protocol_file")
    protocol = panel["protocol"]
    require(block_protocol == block["protocol"], "Block protocol receipts disagree")
    require(block_protocol["panel_protocol_sha256"] == panel["protocol_sha256"], "Block/panel protocol link mismatch")
    require(sha(canonical(protocol)) == panel["protocol_sha256"], "Protocol canonical hash mismatch")
    require(protocol["checkpoint_revision"] == CHECKPOINT, "Checkpoint changed")
    require(kernel["source"]["incumbent_head"] == BASE and kernel["source"]["head"] == PR212, "Prompt kernel source changed")
    require(ahead["incumbent"]["base_commit"] == BASE and ahead["provenance"]["head"] == PR252, "Read-ahead source changed")
    require(analysis["finished"] and block["completed"], "Study incomplete")
    logs = {}
    for relative, label, count in (
        ("integration/runs/combined-cpu.log", "combined_cpu_tests", 31),
        ("panel/runs/cpu-preparation.log", "collector_cpu_tests", 17),
        ("block/runs/qwen-prefill-compile-b.log", "targeted_gpu_tests", 7),
    ):
        log = raw(relative, label).decode("utf-8")
        require((f"Ran {count} tests" in log and "\nOK" in log) if "cpu" in label else f"{count} passed" in log, "Preparation test receipt mismatch")
        logs[label] = count
    raw_rows = {}
    arms = {}
    for arm in ARMS:
        contents = raw(Path("arms") / arm / "runs" / "rows.jsonl", f"arm_{arm}_rows")
        require(sha(contents) == analysis["arms"][arm]["source_row_sha256"], "Raw rows do not match analysis")
        raw_rows[arm] = [json.loads(line) for line in contents.splitlines()]
        cells = [export_row(r, i) for i, r in enumerate(raw_rows[arm])]
        require(Counter(r["suite"] for r in cells) == Counter(prefill=10, decode=6, witness=10), "Panel count mismatch")
        smoke = [export_smoke(json.loads(line)) for line in raw(Path("arms") / arm / "runs" / "smoke.jsonl", f"arm_{arm}_smoke").splitlines()]
        require(len(smoke) == 9, "Smoke count mismatch")
        witnesses = []
        for fixture in ("speed/code", "speed/math", "speed/prose", "speed/structured", "prefill/32000-code"):
            pair = [r for r in cells if r["suite"] == "witness" and r["fixture"] == fixture]
            require(len(pair) == 2 and {r["draft"] for r in pair} == {False, True}, "Incomplete witness pair")
            serial = next(r for r in pair if not r["draft"])
            drafted = next(r for r in pair if r["draft"])
            equal = serial["raw_output_token_ids_sha256"] == drafted["raw_output_token_ids_sha256"]
            require(equal, "Same-runtime witness mismatch")
            witnesses.append({"fixture": fixture, "serial_cell": serial["cell"], "drafted_cell": drafted["cell"],
                              "identical": equal, "actual_tokens": serial["raw_completion_tokens"],
                              "serial_token_ids_sha256": serial["raw_output_token_ids_sha256"],
                              "drafted_token_ids_sha256": drafted["raw_output_token_ids_sha256"]})
        arms[arm] = {"cells": cells, "same_runtime_witnesses": witnesses, "smoke": smoke}
    measured = block["arms"]["A"]["runtime_source_hashes"]
    require(all(block["arms"][a]["runtime_source_hashes"] == measured for a in ARMS), "Measured runtime sources differ across arms")
    for locator, digest in measured.items():
        require(sha((study / "runtime" / "src" / locator).read_bytes()) == digest, "Measured runtime source no longer matches receipt")
    # The complete preparation inventory and measured Python inventory are distinct.
    # Never label the earlier preparation snapshot as the measured final runtime.
    originals212 = {name: value["sha256"] for name, value in kernel["source"]["kernel_files_exact_head"].items()}
    for name, digest in originals212.items():
        require(sha((study / "runtime" / "src" / "tensorfold" / "cuda" / "exl3" / name).read_bytes()) == digest, "Original prompt kernel artifact changed")
    comparisons = {a: [comparison(x, y) for x, y in zip(arms["A"]["cells"], arms[a]["cells"])] for a in ("B", "C")}
    for arm in ("B", "C"):
        require(all(r["token_ids_equal"] for r in comparisons[arm]), "Bounded A/treatment token mismatch")
        require(all(x["raw_output_token_ids"] == y["raw_output_token_ids"] for x, y in zip(raw_rows["A"], raw_rows[arm])), "Raw token identity mismatch")
        for row, recorded in zip(comparisons[arm], analysis["comparisons"][arm]):
            require(row["fixture"] == recorded["fixture"] and row["page_state"] == recorded["page_state"], "Analysis order mismatch")
            for key in ("ttft_seconds", "elapsed_seconds", "prefill_seconds", "prefill_tps_first_sample_inclusive", "decode_tps"):
                require(row[key]["before"] == recorded[key]["before"] and row[key]["after"] == recorded[key]["after"], "Analysis measurement mismatch")
    losses = {a: [r["cell"] for r in comparisons[a] if r["suite"] == "decode" and r["decode_tps"]["change_percent"] is not None and r["decode_tps"]["change_percent"] < -3] for a in ("B", "C")}
    require(losses == {"B": [], "C": [10, 11, 12, 13]}, "Candidate verdict changed")
    require(current["completed"] and current["same_image"] and current["source_and_backend_settings_unchanged"] and current["original_unit_definition_restored"] and not current["experiment_running"], "Restoration final check failed")
    require(restored["completed"] and restored["source_unchanged"] and current["generation"]["finish_reason"] == "stop", "Restoration incomplete")
    minimum_cpu = min(kernel["cpu_safety"]["min_boundary_mem_available"], ahead["safety"]["min_mem_available_bytes_across_recorded_cpu_attempts"], integration["minimum_reserve_bytes"])
    result = {
        "schema_version": 1,
        "study": "bounded_fresh_prefill_screen",
        "hardware": "single NVIDIA GB10, DGX Spark",
        "identity": {
            "checkpoint_revision": CHECKPOINT, "checkpoint_format": "EXL3 4.05bpw_h6_ng6 language checkpoint",
            "runtime_base_commit": BASE,
            "tokenizer_files": protocol["tokenizer_files"],
            "engine_settings": protocol["engine_settings"], "sampling": protocol["sampling"],
            "serving_settings": pick(protocol["app_settings"], ("default_thinking", "max_tokens", "vision_max_images", "context_window")),
        },
        "attribution": {
            "contribution": "Measured integration and bounded qualification, not invention of the upstream mechanisms.",
            "prompt_kernel": {"pull_request": 212, "head": PR212, "author": kernel["source"]["author"],
                              "scope": "Original expert prompt kernels and whole prompt window; unrelated F16/dense GEMM changes excluded.",
                              "original_source_artifact_sha256": originals212,
                              "original_patch_sha256": kernel["source"]["patch_sha256"]},
            "selective_read_ahead": {"pull_request": 252, "head": PR252, "author": ahead["provenance"]["author"],
                                     "original_source_artifacts": hash_inventory(ahead["provenance"]["original_source_hashes"]),
                                     "scope": "Selective future n-gram page advice, adapted to one bounded worker; startup page policy unchanged."},
        },
        "design": {
            "repetitions": protocol["repetitions"], "arm_order": list(ARMS),
            "arm_definitions": {"A": "Incumbent control, both study switches off", "B": "Prompt kernel on; selective read-ahead off", "C": "B plus selective read-ahead, additive"},
            "rows_per_arm": {"prefill": 10, "decode": 6, "witness": 10},
            "page_cold_verified": False, "isolated_gpu_prefill_throughput": False,
            "metric_definitions": protocol["metric_definitions"],
            "decode_neutral_screen_percent": 3,
            "fixtures": [pick(f, ("id", "prompt_sha256", "token_ids_sha256", "prompt_tokens")) for f in protocol["fixtures"]],
            "raw_token_hash_encoding": "SHA256 of UTF-8 JSON integer array, sorted keys, comma/colon separators, no whitespace or newline",
            "source_artifact_id_encoding": "SHA256 of original receipt-relative source locator UTF-8 bytes; locator withheld",
            "smoke_source_sha256": protocol["smoke_policy"]["source_sha256"],
        },
        "preparation": {"passed_test_counts": logs,
                        "evidence_levels": "31 combined CPU contracts; 17 collector CPU/schema tests; 7 native targeted GPU kernel tests. CPU dispatch mocks are not CUDA proof.",
                        "protocol_canonical_sha256": panel["protocol_sha256"],
                        "collector_source_artifacts": hash_inventory(protocol["source_hashes"]),
                        "combined_preparation_source_artifacts": hash_inventory(integration["runtime_source_hashes"]),
                        "measured_runtime_python_source_artifacts": hash_inventory(measured)},
        "arms": arms, "comparisons": comparisons,
        "verification": {"A_B_token_matches": 26, "A_C_token_matches": 26, "matched_cells_per_comparison": 26,
                         "serial_drafted_witness_pairs_per_arm": 5, "smoke_cases_passed_per_arm": 9,
                         "all_primary_rows_zero_reused_prefix_tokens": True},
        "decision": {"B": "provisionally useful candidate", "C": "rejected: unexplained short decode losses beyond 3%",
                     "decode_loss_cells_beyond_3_percent": losses,
                     "selected_candidate": "B", "promotion": "Separate operator decision; study ended on restored incumbent"},
        "safety": {
            "reserve_floor_bytes": block["policy"]["floor_bytes"],
            "host_swap_growth_limit_bytes": block["policy"]["max_host_swap_growth_bytes"],
            "baseline_reset_between_arms": block["policy"]["baseline_reset_between_arms"],
            "minimum_cpu_preparation_boundary_reserve_bytes": minimum_cpu,
            "minimum_cpu_preparation_boundary_reserve_GiB": minimum_cpu / 2**30,
            "minimum_continuous_block_reserve_bytes": block["minimum_reserve_bytes"],
            "minimum_continuous_block_reserve_GiB": block["minimum_reserve_bytes"] / 2**30,
            "fixed_host_swap_baseline_bytes": block["fixed_host_swap_baseline_bytes"],
            "maximum_host_swap_growth_bytes": block["maximum_host_swap_growth_bytes"],
            "maximum_experiment_cgroup_swap_bytes": block["maximum_experiment_cgroup_swap_bytes"],
            "own_cgroup_swap_max_bytes": 0, "hard_safety_abort": analysis["safety"]["hard_safety_abort"],
        },
        "restoration": {
            "completed": restored["completed"], "source_unchanged": restored["source_unchanged"],
            "original_unit_definition_restored": current["original_unit_definition_restored"],
            "same_image": current["same_image"], "source_and_backend_settings_unchanged": current["source_and_backend_settings_unchanged"],
            "experiment_running": current["experiment_running"],
            "final_recorded_route_generation_verified": True,
            "generation_finish_reason": current["generation"]["finish_reason"],
            "generation_usage": current["generation"]["usage"],
            "generation_output_sha256": sha(current["generation"]["answer"].encode("utf-8")),
            "generation_reasoning_characters": current["generation"]["reasoning_characters"],
            "vision_prefix_smoke": [pick(r, ("label", "http_status", "answer_sha256", "prompt_tokens", "cached_tokens", "elapsed_seconds", "finish_reason")) for r in restored["vision_prefix_smoke"]],
            "scope": "Completed study's final recorded check, not a new live endpoint probe by this exporter.",
        },
        "notes": [
            "One pass per condition, fixed A/B/C order. Practical screen, not statistical equivalence, confidence intervals, or a causal explanation of every timing difference.",
            "Cold-advice-unverified means clean file-scoped DONTNEED advice, not verified eviction. Warm is an immediately repeated prompt with prefix cache cleared, not whole-table residency.",
            "Native prefill time/rate includes the first sample. TTFT includes first callback and EOS. Neither is isolated GPU prefill throughput.",
            "Decode is post-first-emission-chunk runtime emission throughput; subtract the entire first chunk and include terminal EOS. Single emitted EOS has no decode estimate.",
            "The nominal 32k code/prose prompts actually contain 32020/32196 tokens. The 32k code decode cell emits one EOS despite cap400; no sustained decode estimate.",
            "Four short decode domains reach the 400-token cap with length finish. Timing and token identity do not prove finished-answer quality.",
            "All 26 bounded cells match A/B and A/C; five serial/drafted pairs match within each arm. This is not a universal lossless or quality claim.",
            "Short fixtures retain their historical custom no-thinking rendering; the serving application remains default-thinking and is tested separately.",
            "Kernel counters are successful Python dispatches, not asynchronous CUDA completions or graph replay. Read-ahead advice completion is not page residency evidence.",
            "Preparation and measured-source inventories are separate snapshots. Opaque artifact IDs retain all source hashes without disclosing source filesystem locations.",
            "Only hashes and counts are published for prompt bytes, emitted token IDs, outputs and reasoning. Raw study text/media, request identifiers, endpoints and host metadata are withheld.",
        ],
        "publication_artifacts": public_artifacts(),
        "source_receipts": receipts,
    }
    workspace_extra = block["arms"]["B"]["capacity_plan"]["cache_workspace_bytes_estimate"] - block["arms"]["A"]["capacity_plan"]["cache_workspace_bytes_estimate"]
    require(workspace_extra == 480285400, "Measured workspace delta changed")
    result["publication_artifacts"]["winning_B_patch"]["measured_capacity_workspace_extra_bytes"] = workspace_extra
    result["preparation"]["block_source_artifacts"] = hash_inventory(block_protocol["source_hashes"])
    validate(result)
    return result


def validate(result):
    require(result["identity"]["checkpoint_revision"] == CHECKPOINT, "Public checkpoint mismatch")
    require(result["identity"]["runtime_base_commit"] == BASE, "Public runtime mismatch")
    require(set(result["arms"]) == set(ARMS), "Missing public arm")
    expected_artifacts = public_artifacts()
    expected_artifacts["winning_B_patch"]["measured_capacity_workspace_extra_bytes"] = 480285400
    require(result["publication_artifacts"] == expected_artifacts, "Public patch/image bytes no longer match receipt")
    for arm in ARMS:
        cells = result["arms"][arm]["cells"]
        require(len(cells) == 26 and [r["cell"] for r in cells] == list(range(26)), "Missing or duplicate public cells")
        require(Counter(r["suite"] for r in cells) == Counter(prefill=10, decode=6, witness=10), "Public panel count mismatch")
        require(len(result["arms"][arm]["same_runtime_witnesses"]) == 5, "Public witness count mismatch")
        require(len(result["arms"][arm]["smoke"]) == 9, "Public smoke count mismatch")
        require(all(r["engine_stats"]["cached"] == 0 and r["eligible"] for r in cells), "Public fresh-prefix gate failed")
        require(all(HASH.fullmatch(r["raw_output_token_ids_sha256"]) and HASH.fullmatch(r["output_sha256"]) for r in cells), "Public output hashes invalid")
        require(all(r["identical"] and r["serial_token_ids_sha256"] == r["drafted_token_ids_sha256"] for r in result["arms"][arm]["same_runtime_witnesses"]), "Public witness identity failed")
    for arm in ("B", "C"):
        require(result["comparisons"][arm] == [comparison(x, y) for x, y in zip(result["arms"]["A"]["cells"], result["arms"][arm]["cells"])], "Public comparison mismatch")
        require(all(r["token_ids_equal"] for r in result["comparisons"][arm]), "Public token identity failed")
    # Whitelist-only construction is the primary privacy boundary. This second
    # gate catches accidental host metadata or raw-response fields in later edits.
    blocked_keys = {"request_id", "task_id", "pid", "container_pid", "container_id", "route", "effective_unit", "output_text", "reasoning_content", "raw_output_token_ids", "body", "messages"}

    def walk(value):
        if isinstance(value, dict):
            require(not blocked_keys.intersection(value), "Private or raw content field in public receipt")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            require(not value.startswith("/") and not re.search(r"(?:https?://|ssh://|[A-Za-z]:\\)", value), "Filesystem or endpoint location in public receipt")
        elif isinstance(value, float):
            require(math.isfinite(value), "Nonfinite public metric")

    walk(result)
    require(result["safety"]["maximum_host_swap_growth_bytes"] == 0 and result["safety"]["maximum_experiment_cgroup_swap_bytes"] == 0, "Public safety mismatch")


def report(result):
    arms = result["arms"]
    lines = [
        "# Measured fresh-prefill screen", "",
        "**B is the provisionally useful prompt-kernel integration. C is rejected** because all four short decode domains lost 6.06% to 7.76% versus A, beyond the frozen 3% screening margin. C is B plus selective read-ahead, not an independent control. The study ended with the incumbent restored and generation-verified.", "",
        "One NVIDIA GB10 / DGX Spark. EXL3 4.05bpw_h6_ng6 language checkpoint, fixed revision `" + CHECKPOINT + "`; TensorFold base `" + BASE + "`. Greedy sampling, BF16 KV, explicit 262144 context, two stream slots, CUDA graphs, MTP depth6/confidence0.7. Every measured row reused zero prefix tokens.", "",
        "Measured integration, not invention: original TensorFold PR212 prompt expert kernels by **" + result["attribution"]["prompt_kernel"]["author"] + "**, head `" + PR212 + "`; PR252 selective n-gram read-ahead by **" + result["attribution"]["selective_read_ahead"]["author"] + "**, head `" + PR252 + "`. PR212 unrelated F16/dense GEMM changes were excluded. Full source artifact hashes and source-receipt hashes are in [prefill-screen.json](prefill-screen.json), with source filesystem locations withheld.", "",
        "Published B is opt-in via `--prefill-kernels`, default off. The public patch contains the 18 measured runtime/test files plus one **packaging-only** `pyproject.toml` fix for legacy212 source data. That fix is not a new GPU sweep. Measured B adds **480285400 bytes** of capacity-plan workspace; PR252/C is not shipped. Public patch SHA256: `" + result["publication_artifacts"]["winning_B_patch"]["sha256"] + "`.", "",
        "The existing [32k/64k before/after PNG](prefill-screen.png) is recorded by actual SHA256, byte size and dimensions in the JSON. Its rounded values represent only cold-advice-unverified A/B rows, not warm rows or C.", "",
        "## All prefill cells", "",
        "Each A/B/C value is **first-sample-inclusive prompt tok/s / TTFT seconds**. Changes compare the rate with A. JSON retains native prefill seconds, request wall seconds, output/count/hash evidence, faults, draft counters and treatment engagement for every cell.", "",
        "| Actual prompt / domain | Page state | A rate / TTFT | B rate / TTFT | C rate / TTFT | B vs A | C vs A |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i in range(10):
        row = arms["A"]["cells"][i]
        state = "cold-advice-unverified" if row["page_state"] == "cold-advice-unverified" else "warm, prefix-uncached"
        values = [f"{arms[a]['cells'][i]['prefill_tps_first_sample_inclusive']:.2f} / {arms[a]['cells'][i]['ttft_seconds']:.3f}" for a in ARMS]
        changes = [f"{result['comparisons'][a][i]['prefill_tps_first_sample_inclusive']['change_percent']:+.2f}%" for a in ("B", "C")]
        lines.append(f"| {row['prompt_tokens']} {row['fixture'].split('-')[-1]} | {state} | " + " | ".join(values + changes) + " |")
    lines += ["", "## All decode cells", "",
              "Post-first-emission-chunk tok/s, EOS included. The first chunk is subtracted in full. All arms have matching emitted counts and finish reasons. `n/a` is not a throughput estimate.", "",
              "| Actual prompt / domain | Cap / emitted / finish | A tok/s | B tok/s | C tok/s | B vs A | C vs A |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for i in range(10, 16):
        row = arms["A"]["cells"][i]
        values = ["n/a" if arms[a]["cells"][i]["decode_tps"] is None else f"{arms[a]['cells'][i]['decode_tps']:.3f}" for a in ARMS]
        changes = ["n/a" if result["comparisons"][a][i]["decode_tps"]["change_percent"] is None else f"{result['comparisons'][a][i]['decode_tps']['change_percent']:+.2f}%" for a in ("B", "C")]
        lines.append(f"| {row['prompt_tokens']} {row['fixture']} | {row['max_tokens']} / {row['raw_completion_tokens']} / {row['finish_reason']} | " + " | ".join(values + changes) + " |")
    safety = result["safety"]
    lines += ["", "## Bounded correctness, preparation and safety", "",
              "- **26/26 A/B and 26/26 A/C emitted-token sequences match**, including EOS. Each arm retains all 10 prefill, 6 decode and 10 witness rows, plus 9 passed serving smokes.",
              "- **Five serial/drafted witness pairs match in each arm**: code, math, prose, structured output and nominal 32k code. The last witness emits one EOS, not a sustained decode sample.",
              "- **31 combined CPU tests + 17 collector CPU tests + 7 native targeted GPU tests passed.** CPU mocks, CUDA kernel tests and full-model measured rows are separate evidence levels.",
              f"- Reserve minima: **{safety['minimum_cpu_preparation_boundary_reserve_GiB']:.2f} GiB** across recorded CPU preparation boundaries, **{safety['minimum_continuous_block_reserve_GiB']:.2f} GiB** in the continuously supervised block. Frozen floor 7 GiB; host swap growth limit 1 GiB. One fixed block baseline, never reset between arms. Maximum host swap growth **0 bytes**; maximum owned-cgroup swap **0 bytes**, swap.max=0. No hard safety abort.",
              "- Original service definition, image, source and backend settings were restored. The final retained route check confirms a stopped experiment, a completed generation with stop finish, default thinking and image-prefix-cache behavior. This exporter did not run a new live endpoint probe.", "",
              "## Limits and interpretation", "",
              "One pass per condition in fixed A/B/C order. B's measured decode changes are within the 3% practical margin; this is **not statistical equivalence**. Cold-advice-unverified is file-scoped clean-page advice, not independently verified cold residency. Warm means immediately repeated prompt with prefix state cleared, not a resident whole table. Both native prefill rate and prompt-tokens/TTFT are first-sample/emission proxies, not isolated GPU prefill throughput.", "",
              "Nominal 32k prompts are actually 32020 code / 32196 prose tokens. The 32k-code decode cell emits **one EOS at cap400**, so there is no decode estimate. Four short domains reach cap400 with length finish; these are timing probes, **not finished-answer quality proof**. Historical short fixtures use their retained no-thinking rendering; serving default-thinking is tested separately.", "",
              "Matched bounded token IDs do not establish universal losslessness or answer quality. Python kernel dispatch counters do not measure CUDA completion/graph replay; completed read-ahead advice does not prove residency. C's short decode loss is retained without inventing a cause. JSON contains hashes rather than prompt/output/reasoning text or raw token arrays; no raw study media, host identifiers, source locations or endpoints are exported.", "",
              "## Reproduce the export", "", "```sh", "python scripts/export_prefill_results.py", "python scripts/export_prefill_results.py --study STUDY_DIRECTORY --check", "python scripts/export_prefill_results.py --study STUDY_DIRECTORY", "```", "",
              "The first command validates the committed JSON and deterministic report. `--study` is optional and supplied by the operator; `--check` compares against the retained source without writing. Regeneration writes only these two result files. No dependencies, GPU access, model loading, Git commits or pushes are performed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Completed study directory selected by the operator")
    parser.add_argument("--check", action="store_true", help="Compare regenerated bytes without writing")
    args = parser.parse_args()
    if args.study is None:
        result = json.loads(RESULT.read_text())
        validate(result)
        require(REPORT.read_text() == report(result), "Committed report differs from receipt")
    else:
        result = build(args.study)
        text, markdown = pretty(result), report(result)
        if args.check:
            require(RESULT.read_text() == text and REPORT.read_text() == markdown, "Committed export differs from study")
        else:
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            RESULT.write_text(text)
            REPORT.write_text(markdown)
    print("PASS: A/B/C each 26 cells (10 prefill + 6 decode + 10 witness), 9 smokes; A/B 26/26, A/C 26/26; 5 serial/drafted pairs per arm")
    print("PASS: source hashes, metric denominators, zero prefix reuse, deterministic report, public allowlist; B useful, C rejected")


if __name__ == "__main__":
    main()
