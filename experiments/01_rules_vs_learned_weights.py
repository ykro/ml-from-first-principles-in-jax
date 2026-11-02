"""Experiment 01: From Hardcoded if/else Rules to Learned Weights (Linear Regression & MSE).

Demonstrates for Software Engineers:
1. Traditional software engineering encodes domain heuristics manually (`if/else`,
   magic thresholds, hand-tuned multipliers) to estimate system behavior.
2. Machine Learning replaces hand-tuned magic numbers with a parameterized function
   `y_hat = X @ w + b` and an objective `MSE Loss = mean((y_hat - y)^2)`.
3. Using `jax.value_and_grad` inside a `@jax.jit` training loop, the system discovers
   the true underlying weights from historical telemetry (`CPU`, `RAM`, `RPS`, `payload_kb`).

How to read the numbers: the handwritten rule is a naive baseline (guessed coefficients),
so "x lower MSE vs rules" mostly measures how bad the guess was. The fair reference is the
closed-form least-squares fit (the best any linear model can do on this data) and the noise
floor (MSE of the true weights). The data is synthetic and linear by construction.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def hardcoded_latency_rules(features: jax.Array) -> jax.Array:
    """Handwritten heuristic written by an on-call engineer to estimate p99 latency (ms).

    features columns: [cpu_util, ram_gb, rps_k, payload_kb]
    """
    cpu = features[:, 0]
    ram = features[:, 1]
    rps = features[:, 2]
    payload = features[:, 3]

    # Typical brittle rule chain with guessed coefficients and threshold jumps
    base_ms = 12.0 + 18.0 * cpu + 2.5 * ram + 6.0 * rps + 0.8 * payload
    high_load_penalty = jnp.where(cpu > 0.75, 14.0, 0.0)
    large_payload_penalty = jnp.where(payload > 1.5, 9.0, 0.0)
    return base_ms + high_load_penalty + large_payload_penalty


def predict_linear(params: dict[str, jax.Array], features: jax.Array) -> jax.Array:
    """Parameterized linear model: y_hat = X @ w + b."""
    return features @ params["w"] + params["b"]


def mse_loss(params: dict[str, jax.Array], features: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean Squared Error loss function."""
    preds = predict_linear(params, features)
    return jnp.mean(jnp.square(preds - targets))


@jax.jit
def train_step(
    params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    lr: float = 0.08,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Single compiled gradient descent step using jax.value_and_grad."""
    loss, grads = jax.value_and_grad(mse_loss)(params, features, targets)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


def closed_form_least_squares(features: jax.Array, targets: jax.Array) -> tuple[np.ndarray, float]:
    """Best possible linear fit (ordinary least squares in float64): the fair reference."""
    x = np.c_[np.asarray(features, dtype=np.float64), np.ones(features.shape[0])]
    y = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef, float(np.mean(np.square(x @ coef - y)))


def run_experiment(n_samples: int = 2048, n_steps: int = 250) -> dict[str, Any]:
    """Compare handwritten latency rules against learned linear weights in JAX."""
    key = jax.random.key(42)
    k_feat, k_noise = jax.random.split(key)

    # Normalized telemetry features: [cpu_util, ram_gb, rps_k, payload_kb]
    features = jax.random.uniform(k_feat, shape=(n_samples, 4), minval=0.1, maxval=2.0)

    # True data-generating physics of the service (unknown to the engineer)
    true_w = jnp.array([34.5, -4.2, 19.8, 11.4], dtype=jnp.float32)
    true_b = jnp.array(8.5, dtype=jnp.float32)
    noise = 0.6 * jax.random.normal(k_noise, shape=(n_samples,), dtype=jnp.float32)
    targets = features @ true_w + true_b + noise

    # 1. Evaluate handwritten rules
    rule_preds = hardcoded_latency_rules(features)
    rules_mse = float(jnp.mean(jnp.square(rule_preds - targets)))

    # 2. Initialize model parameters at zero (zero prior domain guesses)
    params = {
        "w": jnp.zeros((4,), dtype=jnp.float32),
        "b": jnp.array(0.0, dtype=jnp.float32),
    }
    initial_ml_mse = float(mse_loss(params, features, targets))

    # Warmup @jax.jit compilation (the first call traces and compiles)
    _params_warm, _loss_warm = train_step(params, features, targets)
    _loss_warm.block_until_ready()

    # Run the compiled training loop, timing every step (median per step)
    step_ms = []
    curr_params = params
    for _ in range(n_steps):
        t0 = time.perf_counter()
        curr_params, final_loss = train_step(curr_params, features, targets)
        final_loss.block_until_ready()
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    learned_mse = float(mse_loss(curr_params, features, targets))
    mse_reduction_vs_rules = rules_mse / learned_mse
    max_weight_error = float(jnp.max(jnp.abs(curr_params["w"] - true_w)))

    # 3. Fair references: closed-form least squares and the noise floor of the true weights
    ols_coef, ols_mse = closed_form_least_squares(features, targets)
    noise_floor_mse = float(jnp.mean(jnp.square(features @ true_w + true_b - targets)))

    return {
        "experiment": "01_rules_vs_learned_weights",
        "backend": jax.default_backend(),
        "n_samples": n_samples,
        "n_steps": n_steps,
        "rules_mse": round(rules_mse, 4),
        "initial_ml_mse": round(initial_ml_mse, 4),
        "learned_mse": round(learned_mse, 4),
        "mse_reduction_vs_rules": round(mse_reduction_vs_rules, 2),
        "true_w": [round(float(x), 3) for x in true_w],
        "learned_w": [round(float(x), 3) for x in curr_params["w"]],
        "true_b": round(float(true_b), 3),
        "learned_b": round(float(curr_params["b"]), 3),
        "max_weight_error": round(max_weight_error, 4),
        "closed_form_mse": round(ols_mse, 4),
        "closed_form_w": [round(float(x), 3) for x in ols_coef[:4]],
        "closed_form_b": round(float(ols_coef[4]), 3),
        "learned_vs_closed_form_ratio": round(learned_mse / ols_mse, 3),
        "noise_floor_mse": round(noise_floor_mse, 4),
        "median_step_ms": round(statistics.median(step_ms), 4),
        "timed_steps": n_steps,
    }


def main() -> None:
    res = run_experiment()
    print("=" * 74)
    print("Experiment 01: Hardcoded if/else Rules vs Learned Weights (Linear ML)")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(f"Telemetry Dataset           : {res['n_samples']} samples x 4 features")
    print(f"Handwritten Rules MSE       : {res['rules_mse']:.4f} ms^2 (guessed coefficients)")
    print(f"Initial Un-trained ML MSE   : {res['initial_ml_mse']:.4f} ms^2 (w = 0, b = 0)")
    print(f"Learned MSE ({res['n_steps']} GD steps)  : {res['learned_mse']:.4f} ms^2")
    print(
        f"MSE Reduction vs Rules      : {res['mse_reduction_vs_rules']:.2f}x lower "
        "(naive baseline)"
    )
    print("-" * 74)
    print(f"Closed-Form Least Squares   : {res['closed_form_mse']:.4f} ms^2 (best linear fit)")
    print(f"Noise Floor (true w, b)     : {res['noise_floor_mse']:.4f} ms^2")
    gap_pct = (res["learned_vs_closed_form_ratio"] - 1.0) * 100.0
    print(
        f"Learned / Closed-Form MSE   : {res['learned_vs_closed_form_ratio']:.3f} "
        f"({gap_pct:.1f}% above the best fit)"
    )
    print("-" * 74)
    print(f"True Weights (w)            : {res['true_w']} | b = {res['true_b']}")
    print(f"Learned Weights (w)         : {res['learned_w']} | b = {res['learned_b']}")
    print(f"Max Weight Recovery Error   : {res['max_weight_error']:.4f}")
    print(
        f"Median @jit Train Step      : {res['median_step_ms']:.4f} ms "
        f"({res['timed_steps']} steps)"
    )


if __name__ == "__main__":
    main()
