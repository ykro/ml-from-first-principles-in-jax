"""Deterministic pytest suite for ml-from-first-principles-in-jax."""

from __future__ import annotations

import importlib

exp01 = importlib.import_module("experiments.01_rules_vs_learned_weights")
exp02 = importlib.import_module("experiments.02_gradient_descent_geometry")
exp03 = importlib.import_module("experiments.03_logistic_classification_cross_entropy")
exp04 = importlib.import_module("experiments.04_two_layer_mlp_from_scratch")
exp05 = importlib.import_module("experiments.05_overfitting_and_weight_decay")


def test_01_rules_vs_learned_weights() -> None:
    res = exp01.run_experiment()
    assert res["mse_reduction_vs_rules"] >= 15.0
    assert res["max_weight_error"] < 0.25


def test_02_gradient_descent_geometry() -> None:
    res = exp02.run_experiment()
    assert res["adam_final_loss"] < res["momentum_final_loss"] < res["sgd_final_loss"]
    assert res["adam_steps_to_0_05"] < res["sgd_steps_to_0_05"]
    assert res["speedup"] >= 5.0


def test_03_logistic_classification_cross_entropy() -> None:
    res = exp03.run_experiment()
    assert res["naive_overflow_nan"] is True
    assert res["stable_logsumexp_finite"] is True
    assert res["max_diff_vs_jax_nn"] < 1e-5
    assert res["final_accuracy"] >= 95.0


def test_04_two_layer_mlp_from_scratch() -> None:
    res = exp04.run_experiment()
    assert res["linear_collapse_max_diff"] < 1e-4
    assert res["linear_stack_accuracy"] < 60.0
    assert res["mlp_gelu_accuracy"] >= 97.0


def test_05_overfitting_and_weight_decay() -> None:
    res = exp05.run_experiment()
    assert res["overfit_val_to_train_ratio"] >= 4.0
    assert res["val_mse_reduction_ratio"] >= 3.0
    assert res["regularized_weight_norm"] < res["overfit_weight_norm"]
