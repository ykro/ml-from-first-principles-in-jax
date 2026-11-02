"""The Outside Judge: Deterministic ML First-Principles Verification & Benchmark Runner.

Executes all 5 Machine Learning from First Principles experiments in JAX,
enforces deterministic mathematical and convergence assertions, and writes
`benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.

The quality metrics (MSE, loss, accuracy, weight norms) come from synthetic data with
fixed seeds and repeat exactly on CPU. The timings are medians after a warmup and vary
run to run. Every row also prints a fairer reference next to the naive baseline (see
"Cómo leer estas cifras" in the README).
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_rules_vs_learned_weights")
exp02 = importlib.import_module("experiments.02_gradient_descent_geometry")
exp03 = importlib.import_module("experiments.03_logistic_classification_cross_entropy")
exp04 = importlib.import_module("experiments.04_two_layer_mlp_from_scratch")
exp05 = importlib.import_module("experiments.05_overfitting_and_weight_decay")


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 ML experiments and enforce deterministic assertions."""
    load_before = _load_average()
    r01 = exp01.run_experiment()
    r02 = exp02.run_experiment()
    r03 = exp03.run_experiment()
    r04 = exp04.run_experiment()
    r05 = exp05.run_experiment()

    # Experiment 01: Hardcoded Rules vs Learned Linear Weights
    assert r01["mse_reduction_vs_rules"] >= 15.0, (
        f"Exp01 MSE reduction vs rules must be >= 15x, got {r01['mse_reduction_vs_rules']}x"
    )
    assert r01["max_weight_error"] < 0.25, (
        f"Exp01 failed to recover true weights accurately: {r01['max_weight_error']}"
    )
    assert r01["learned_vs_closed_form_ratio"] < 1.10, (
        f"Exp01 gradient descent should land within 10% of the closed-form fit: "
        f"{r01['learned_vs_closed_form_ratio']}"
    )

    # Experiment 02: Gradient Descent Geometry (SGD vs Momentum vs Adam)
    assert r02["adam_final_loss"] < r02["momentum_final_loss"] < r02["sgd_final_loss"], (
        "Exp02 expected Adam < Momentum < SGD final loss on ill-conditioned surface"
    )
    assert r02["adam_steps_to_0_05"] is not None, "Exp02 Adam must reach loss < 0.05"
    assert r02["sgd_steps_to_0_05"] is None or (
        r02["adam_steps_to_0_05"] < r02["sgd_steps_to_0_05"]
    ), "Exp02 expected Adam to reach loss < 0.05 in fewer steps than SGD"
    sweep = r02["lr_sweep"]
    assert sweep["adam"]["final_loss"] < sweep["momentum"]["final_loss"], (
        "Exp02 Adam should still win with every optimizer at its best learning rate"
    )
    assert sweep["momentum"]["final_loss"] < sweep["sgd"]["final_loss"], (
        "Exp02 Momentum should beat SGD with every optimizer at its best learning rate"
    )
    assert r02["speedup_vs_eager"] >= 20.0, (
        f"Exp02 scan vs eager speedup too low: {r02['speedup_vs_eager']}x"
    )
    assert r02["speedup_vs_jit_step_loop"] >= 1.0, (
        f"Exp02 scan slower than a Python loop of @jit steps: {r02['speedup_vs_jit_step_loop']}x"
    )

    # Experiment 03: Softmax, LogSumExp & Cross-Entropy
    assert r03["naive_overflow_nan"] is True, "Exp03 naive softmax should overflow on 120.0"
    assert r03["stable_logsumexp_finite"] is True, "Exp03 stable log_softmax must remain finite"
    assert r03["max_diff_vs_jax_nn"] < 1e-5, f"Exp03 parity failed: {r03['max_diff_vs_jax_nn']}"
    assert r03["final_accuracy"] >= 95.0, f"Exp03 train accuracy < 95%: {r03['final_accuracy']}%"
    assert r03["test_accuracy"] >= 95.0, f"Exp03 test accuracy < 95%: {r03['test_accuracy']}%"

    # Experiment 04: Two-Layer MLP from Scratch
    assert r04["linear_collapse_max_diff"] < 1e-4, (
        f"Exp04 linear collapse identity failed: {r04['linear_collapse_max_diff']}"
    )
    assert r04["linear_stack_accuracy"] < 60.0, (
        f"Exp04 linear stack without activation should fail: {r04['linear_stack_accuracy']}%"
    )
    assert r04["mlp_gelu_accuracy"] >= 97.0, (
        f"Exp04 GELU MLP must exceed 97% train accuracy: {r04['mlp_gelu_accuracy']}%"
    )
    assert r04["mlp_gelu_test_accuracy"] >= 97.0, (
        f"Exp04 GELU MLP must exceed 97% test accuracy: {r04['mlp_gelu_test_accuracy']}%"
    )
    assert r04["zero_init_distinct_hidden_units"] == 1, (
        "Exp04 zero init should leave every hidden unit identical"
    )
    assert r04["he_init_distinct_hidden_units"] == r04["hidden_dim"], (
        "Exp04 random init should give every hidden unit its own weights"
    )

    # Experiment 05: Overfitting & Weight Decay
    assert r05["overfit_val_to_train_ratio"] >= 4.0, (
        f"Exp05 unregularized model lacked train/val gap: {r05['overfit_val_to_train_ratio']}x"
    )
    assert r05["val_mse_reduction_ratio"] >= 3.0, (
        f"Exp05 L2 weight decay must reduce validation MSE >= 3x: {r05['val_mse_reduction_ratio']}x"
    )
    assert r05["test_mse_reduction_ratio"] >= 3.0, (
        f"Exp05 must also reduce held-out test MSE >= 3x: {r05['test_mse_reduction_ratio']}x"
    )
    assert r05["regularized_weight_norm"] < r05["overfit_weight_norm"], (
        "Exp05 L2 weight decay must shrink parameter L2 norm"
    )

    return {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "timing": "median ms after a warmup; varies run to run",
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_device": str(jax.devices()[0]),
        },
        "experiments": {
            "01_rules_vs_learned_weights": r01,
            "02_gradient_descent_geometry": r02,
            "03_logistic_classification_cross_entropy": r03,
            "04_two_layer_mlp_from_scratch": r04,
            "05_overfitting_and_weight_decay": r05,
        },
        "all_verifications_passed": True,
    }


def _steps(value: int | None) -> str:
    return f"step {value}" if value is not None else "never"


def main() -> None:
    summary = run_all_verifications()
    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "latest_results.json"
    out_txt = bench_dir / "terminal_output.txt"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    env = summary["environment"]
    e01 = summary["experiments"]["01_rules_vs_learned_weights"]
    e02 = summary["experiments"]["02_gradient_descent_geometry"]
    e03 = summary["experiments"]["03_logistic_classification_cross_entropy"]
    e04 = summary["experiments"]["04_two_layer_mlp_from_scratch"]
    e05 = summary["experiments"]["05_overfitting_and_weight_decay"]
    sw = e02["lr_sweep"]
    t05 = e05["test_mse"]

    def load1(values: list[float]) -> str:
        return f"{values[0]:.2f}" if values else "n/a"

    gap01 = (e01["learned_vs_closed_form_ratio"] - 1.0) * 100.0
    rows = [
        (
            "01. Rules vs learned (MSE)",
            f"rules {e01['rules_mse']:.4f}",
            f"learned {e01['learned_mse']:.4f}",
            f"{e01['mse_reduction_vs_rules']:.2f}x lower",
            f"fair: closed-form least squares {e01['closed_form_mse']:.4f} "
            f"(learned is {gap01:.1f}% above it)",
        ),
        (
            f"02. Optimizers, {e02['n_steps']} steps",
            f"SGD {e02['sgd_final_loss']:.4f}",
            f"Adam {e02['adam_final_loss']:.6f}",
            f"<0.05 at {_steps(e02['adam_steps_to_0_05'])}",
            f"fair: best lr each -> SGD {_steps(sw['sgd']['steps_to_0_05'])}, "
            f"Momentum {_steps(sw['momentum']['steps_to_0_05'])}, "
            f"Adam {_steps(sw['adam']['steps_to_0_05'])}",
        ),
        (
            "02. Same work, timing",
            f"eager {e02['eager_loop_ms']:.2f} ms",
            f"scan {e02['compiled_scan_ms']:.2f} ms",
            f"{e02['speedup_vs_eager']:.2f}x",
            f"fair: Python loop of @jit step {e02['jit_step_loop_ms']:.2f} ms "
            f"-> scan {e02['speedup_vs_jit_step_loop']:.2f}x faster",
        ),
        (
            "03. LogSumExp + cross-entropy",
            "naive: nan",
            "stable: finite",
            f"diff {e03['max_diff_vs_jax_nn']:.1e}",
            f"accuracy {e03['initial_accuracy']:.2f}% -> train {e03['final_accuracy']:.2f}%, "
            f"held-out test {e03['test_accuracy']:.2f}%",
        ),
        (
            "04. MLP vs linear stack",
            f"linear {e04['linear_stack_accuracy']:.2f}%",
            f"MLP {e04['mlp_gelu_accuracy']:.2f}%",
            "train acc",
            f"held-out test: linear {e04['linear_stack_test_accuracy']:.2f}%, "
            f"MLP {e04['mlp_gelu_test_accuracy']:.2f}%; zero init: "
            f"{e04['zero_init_distinct_hidden_units']} of {e04['hidden_dim']} units distinct",
        ),
        (
            "05. Overfit vs WD + val gate",
            f"val {e05['overfit_val_mse']:.4f}",
            f"val {e05['regularized_val_mse']:.4f}",
            f"{e05['val_mse_reduction_ratio']:.2f}x lower",
            f"fair (test set): {t05['unregularized_final']:.4f} -> "
            f"{t05['weight_decay_plus_gate_best']:.4f}, "
            f"{e05['test_mse_reduction_ratio']:.2f}x lower; "
            f"WD alone {t05['weight_decay_only_final']:.4f}",
        ),
    ]
    width = 92
    lines = [
        "=" * width,
        "THE OUTSIDE JUDGE — ML from First Principles in JAX Verification Report",
        "=" * width,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"NumPy {env['numpy_version']} | Backend: {env['jax_backend']}"
        ),
        (
            f"CPU     : {env['cpu_model']} ({env['cpu_count']} cores) | 1-min load average "
            f"{load1(env['load_average_before'])} before, {load1(env['load_average_after'])} after"
        ),
        "-" * width,
        f"{'Experiment':<30} | {'Baseline':<18} | {'JAX ML':<16} | Result",
        "-" * width,
    ]
    for name, base, jax_ml, result, fair in rows:
        lines.append(f"{name:<30} | {base:<18} | {jax_ml:<16} | {result}")
        lines.append(f"    {fair}")
    lines += [
        "-" * width,
        "Quality metrics are deterministic (fixed seeds). Times are medians after a warmup",
        "and vary run to run.",
        f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
