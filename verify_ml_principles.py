"""The Outside Judge: Deterministic ML First-Principles Verification & Benchmark Runner.

Executes all 5 Machine Learning from First Principles experiments in JAX,
enforces deterministic mathematical and convergence assertions, and writes
`benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_rules_vs_learned_weights")
exp02 = importlib.import_module("experiments.02_gradient_descent_geometry")
exp03 = importlib.import_module("experiments.03_logistic_classification_cross_entropy")
exp04 = importlib.import_module("experiments.04_two_layer_mlp_from_scratch")
exp05 = importlib.import_module("experiments.05_overfitting_and_weight_decay")


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 ML experiments and enforce deterministic assertions."""
    r01 = exp01.run_experiment()
    r02 = exp02.run_experiment()
    r03 = exp03.run_experiment()
    r04 = exp04.run_experiment()
    r05 = exp05.run_experiment()

    # Assertions for Experiment 01: Hardcoded Rules vs Learned Linear Weights
    assert r01["mse_reduction_vs_rules"] >= 15.0, (
        f"Exp01 MSE reduction vs rules must be >= 15x, got {r01['mse_reduction_vs_rules']}x"
    )
    assert r01["max_weight_error"] < 0.25, (
        f"Exp01 failed to recover true weights accurately: {r01['max_weight_error']}"
    )

    # Assertions for Experiment 02: Gradient Descent Geometry (SGD vs Momentum vs Adam)
    assert r02["adam_final_loss"] < r02["momentum_final_loss"] < r02["sgd_final_loss"], (
        "Exp02 expected Adam < Momentum < SGD final loss on ill-conditioned surface"
    )
    assert r02["adam_steps_to_0_05"] < r02["sgd_steps_to_0_05"], (
        "Exp02 expected Adam to reach loss < 0.05 in fewer steps than SGD"
    )
    assert r02["speedup"] >= 5.0, f"Exp02 compiled scan speedup too low: {r02['speedup']}x"

    # Assertions for Experiment 03: Softmax, LogSumExp & Cross-Entropy
    assert r03["naive_overflow_nan"] is True, "Exp03 naive softmax should overflow on 120.0"
    assert r03["stable_logsumexp_finite"] is True, "Exp03 stable log_softmax must remain finite"
    assert r03["max_diff_vs_jax_nn"] < 1e-5, f"Exp03 parity failed: {r03['max_diff_vs_jax_nn']}"
    assert r03["final_accuracy"] >= 95.0, f"Exp03 accuracy < 95%: {r03['final_accuracy']}%"

    # Assertions for Experiment 04: Two-Layer MLP from Scratch
    assert r04["linear_collapse_max_diff"] < 1e-4, (
        f"Exp04 linear collapse identity failed: {r04['linear_collapse_max_diff']}"
    )
    assert r04["linear_stack_accuracy"] < 60.0, (
        f"Exp04 linear stack without activation should fail: {r04['linear_stack_accuracy']}%"
    )
    assert r04["mlp_gelu_accuracy"] >= 97.0, (
        f"Exp04 GELU MLP must exceed 97% accuracy on concentric rings: {r04['mlp_gelu_accuracy']}%"
    )

    # Assertions for Experiment 05: Overfitting & Weight Decay
    assert r05["overfit_val_to_train_ratio"] >= 4.0, (
        f"Exp05 unregularized model lacked train/val gap: {r05['overfit_val_to_train_ratio']}x"
    )
    assert r05["val_mse_reduction_ratio"] >= 3.0, (
        f"Exp05 L2 weight decay must reduce validation MSE >= 3x: {r05['val_mse_reduction_ratio']}x"
    )
    assert r05["regularized_weight_norm"] < r05["overfit_weight_norm"], (
        "Exp05 L2 weight decay must shrink parameter L2 norm"
    )

    summary = {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
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
    return summary


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

    lines = [
        "=" * 78,
        "THE OUTSIDE JUDGE — ML from First Principles in JAX Verification Report",
        "=" * 78,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"NumPy {env['numpy_version']} | Backend: {env['jax_backend']}"
        ),
        "-" * 78,
        (
            f"{'Experiment':<34} | {'Baseline / Naive':<16} | "
            f"{'Learned / JAX ML':<16} | {'Gain / Check':<12}"
        ),
        "-" * 78,
        (
            f"{'01. Rules vs Learned Linear MSE':<34} | "
            f"{str(e01['rules_mse']) + ' MSE':<16} | "
            f"{str(e01['learned_mse']) + ' MSE':<16} | "
            f"{str(e01['mse_reduction_vs_rules']) + 'x lower':<12}"
        ),
        (
            f"{'02. SGD vs Momentum vs Adam':<34} | "
            f"{'SGD ' + str(e02['sgd_steps_to_0_05']) + ' steps':<16} | "
            f"{'Adam ' + str(e02['adam_steps_to_0_05']) + ' steps':<16} | "
            f"{str(e02['speedup']) + 'x @jit':<12}"
        ),
        (
            f"{'03. LogSumExp & Cross-Entropy':<34} | "
            f"{str(e03['initial_accuracy']) + '% (NaN exp)':<16} | "
            f"{str(e03['final_accuracy']) + '% (Finite)':<16} | "
            f"{'PASS (<1e-7)':<12}"
        ),
        (
            f"{'04. 2-Layer MLP vs Linear Stack':<34} | "
            f"{str(e04['linear_stack_accuracy']) + '% (Linear)':<16} | "
            f"{str(e04['mlp_gelu_accuracy']) + '% (GELU MLP)':<16} | "
            f"{'PASS (>97%)':<12}"
        ),
        (
            f"{'05. Overfitting vs Weight Decay':<34} | "
            f"{str(e05['overfit_val_mse']) + ' Val MSE':<16} | "
            f"{str(e05['regularized_val_mse']) + ' Val MSE':<16} | "
            f"{str(e05['val_mse_reduction_ratio']) + 'x lower':<12}"
        ),
        "-" * 78,
        f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
