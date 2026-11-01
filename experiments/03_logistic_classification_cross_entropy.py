"""Experiment 03: From Raw Logits to Probabilities (Softmax, LogSumExp & Cross-Entropy).

Demonstrates for Software Engineers:
1. Why classification models output unconstrained real numbers (`logits`) and map
   them to probability distributions via `Sigmoid` (binary) or `Softmax` (multiclass).
2. Why naive `jnp.exp(logits) / jnp.sum(jnp.exp(logits))` overflows to `inf`/`nan`
   in `float32` as soon as a logit exceeds ~89.0.
3. How the `LogSumExp` identity (`max_z + log(sum(exp(z - max_z)))`) guarantees
   numerical stability for any logit magnitude.
4. End-to-end training of a 3-class traffic anomaly classifier (`Normal`,
   `DDoS Burst`, `Memory Leak`) using Cross-Entropy Loss and `@jax.jit`.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp


def naive_softmax(logits: jax.Array) -> jax.Array:
    """Unstable softmax that exponentiates raw logits directly."""
    exps = jnp.exp(logits)
    return exps / jnp.sum(exps, axis=-1, keepdims=True)


def stable_log_softmax(logits: jax.Array) -> jax.Array:
    """Numerically stable log-softmax using the max-subtraction LogSumExp trick."""
    max_logit = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logit
    logsumexp = max_logit + jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return logits - logsumexp


def cross_entropy_loss(
    params: dict[str, jax.Array],
    features: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Multiclass cross-entropy loss from raw logits."""
    logits = features @ params["W"] + params["b"]
    log_probs = stable_log_softmax(logits)
    n = features.shape[0]
    picked_log_probs = log_probs[jnp.arange(n), labels]
    return -jnp.mean(picked_log_probs)


def compute_accuracy(
    params: dict[str, jax.Array],
    features: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Compute classification accuracy from argmax of logits."""
    logits = features @ params["W"] + params["b"]
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))


@jax.jit
def train_classifier_step(
    params: dict[str, jax.Array],
    features: jax.Array,
    labels: jax.Array,
    lr: float = 0.25,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Compiled multiclass logistic regression training step."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, features, labels)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


def run_experiment(n_per_class: int = 600, n_steps: int = 200) -> dict[str, Any]:
    """Verify LogSumExp numerical stability and train a 3-class anomaly classifier."""
    # 1. Demonstrate float32 overflow in naive softmax vs stable log_softmax
    extreme_logits = jnp.array([[10.0, 20.0, 120.0]], dtype=jnp.float32)
    naive_probs = naive_softmax(extreme_logits)
    naive_has_nan = bool(jnp.any(jnp.isnan(naive_probs)))

    stable_lp = stable_log_softmax(extreme_logits)
    stable_probs = jnp.exp(stable_lp)
    stable_is_finite = bool(
        jnp.all(jnp.isfinite(stable_lp)) and jnp.all(jnp.isfinite(stable_probs))
    )
    max_diff_vs_jax_nn = float(jnp.max(jnp.abs(stable_lp - jax.nn.log_softmax(extreme_logits))))

    # 2. Generate 3-class server telemetry dataset:
    # Class 0: Normal traffic, Class 1: DDoS burst, Class 2: Memory leak
    key = jax.random.key(101)
    k0, k1, k2 = jax.random.split(key, 3)

    centers = jnp.array(
        [
            [-1.8, -1.2, 0.2, -0.8],  # Normal
            [2.2, -0.6, 2.5, 1.4],    # DDoS Burst
            [-0.4, 2.4, -1.5, 1.8],   # Memory Leak
        ],
        dtype=jnp.float32,
    )
    c0 = centers[0] + 0.75 * jax.random.normal(k0, (n_per_class, 4), dtype=jnp.float32)
    c1 = centers[1] + 0.75 * jax.random.normal(k1, (n_per_class, 4), dtype=jnp.float32)
    c2 = centers[2] + 0.75 * jax.random.normal(k2, (n_per_class, 4), dtype=jnp.float32)

    features = jnp.concatenate([c0, c1, c2], axis=0)
    labels = jnp.concatenate(
        [
            jnp.zeros(n_per_class, dtype=jnp.int32),
            jnp.ones(n_per_class, dtype=jnp.int32),
            jnp.full(n_per_class, 2, dtype=jnp.int32),
        ],
        axis=0,
    )

    params = {
        "W": jnp.zeros((4, 3), dtype=jnp.float32),
        "b": jnp.zeros((3,), dtype=jnp.float32),
    }

    initial_loss = float(cross_entropy_loss(params, features, labels))
    initial_acc = float(compute_accuracy(params, features, labels))

    # Warmup @jax.jit
    _p, _l = train_classifier_step(params, features, labels)
    _l.block_until_ready()

    t0 = time.perf_counter()
    curr_params = params
    final_loss = jnp.array(initial_loss)
    for _ in range(n_steps):
        curr_params, final_loss = train_classifier_step(curr_params, features, labels)
    final_loss.block_until_ready()
    train_time_ms = (time.perf_counter() - t0) * 1000.0

    final_acc = float(compute_accuracy(curr_params, features, labels))

    return {
        "experiment": "03_logistic_classification_cross_entropy",
        "backend": jax.default_backend(),
        "n_samples": int(features.shape[0]),
        "naive_overflow_nan": naive_has_nan,
        "stable_logsumexp_finite": stable_is_finite,
        "max_diff_vs_jax_nn": max_diff_vs_jax_nn,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(float(final_loss), 4),
        "initial_accuracy": round(initial_acc * 100.0, 2),
        "final_accuracy": round(final_acc * 100.0, 2),
        "train_time_ms": round(train_time_ms, 4),
    }


def main() -> None:
    res = run_experiment()
    print("=" * 74)
    print("Experiment 03: Softmax, LogSumExp Stability & Cross-Entropy Classification")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(f"Naive exp(120.0) NaN/Inf    : {res['naive_overflow_nan']} (float32 overflow)")
    print(
        f"LogSumExp Trick Finite      : {res['stable_logsumexp_finite']} "
        f"(diff={res['max_diff_vs_jax_nn']:.1e})"
    )
    print("-" * 74)
    print(
        f"Initial Cross-Entropy Loss  : {res['initial_loss']:.4f} "
        f"(Accuracy: {res['initial_accuracy']:.2f}%)"
    )
    print(
        f"Trained Cross-Entropy Loss  : {res['final_loss']:.4f} "
        f"(Accuracy: {res['final_accuracy']:.2f}%)"
    )
    print(f"200 @jit Training Steps     : {res['train_time_ms']:.4f} ms")


if __name__ == "__main__":
    main()
