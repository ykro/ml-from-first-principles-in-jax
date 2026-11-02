"""Experiment 04: Your First Neural Network (2-Layer MLP in 80 Lines of Pure JAX).

Demonstrates for Software Engineers:
1. Why stacking linear layers without a non-linear activation (`(X @ W1) @ W2`)
   collapses algebraically into a single matrix `X @ W_collapsed` and fails on
   non-linear decision boundaries (concentric latency/load rings).
2. Why zero initialization breaks symmetry (all hidden neurons compute identical
   gradients) and how He/Xavier variance scaling with `jax.random.split` solves it.
3. A complete 2-layer MLP (`Linear -> GELU -> Linear -> LogSoftmax`) trained with
   `@jax.jit` and `jax.value_and_grad` from ~33% random accuracy to >98% accuracy.

How to read the numbers: train accuracy is measured on the 1500 points used for the
gradients; test accuracy on 1500 new points from the same rings (different key). The
zero-init run trains the same MLP from all-zero weights to show the symmetry problem.
"""

from __future__ import annotations

import statistics
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def generate_concentric_rings(
    key: jax.Array,
    n_per_class: int = 500,
) -> tuple[jax.Array, jax.Array]:
    """Generate a 3-class non-linear concentric ring dataset in 2D."""
    k_angles, k_radii = jax.random.split(key)
    angles = jax.random.uniform(k_angles, (3, n_per_class), minval=0.0, maxval=2.0 * jnp.pi)
    radii_bases = jnp.array([[0.35], [1.05], [1.85]], dtype=jnp.float32)
    radii = radii_bases + 0.14 * jax.random.normal(k_radii, (3, n_per_class), dtype=jnp.float32)

    x0 = radii * jnp.cos(angles)
    x1 = radii * jnp.sin(angles)
    features = jnp.stack([x0.reshape(-1), x1.reshape(-1)], axis=-1)
    labels = jnp.repeat(jnp.arange(3, dtype=jnp.int32), n_per_class)
    return features, labels


def init_mlp_params(
    key: jax.Array,
    in_dim: int = 2,
    hidden_dim: int = 64,
    out_dim: int = 3,
) -> dict[str, jax.Array]:
    """Initialize a 2-layer MLP using He/Xavier scaling and explicit PRNG splits."""
    k1, k2 = jax.random.split(key)
    w1_scale = jnp.sqrt(2.0 / in_dim)
    w2_scale = jnp.sqrt(2.0 / hidden_dim)
    return {
        "W1": jax.random.normal(k1, (in_dim, hidden_dim), dtype=jnp.float32) * w1_scale,
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "W2": jax.random.normal(k2, (hidden_dim, out_dim), dtype=jnp.float32) * w2_scale,
        "b2": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def forward_linear_stack(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Two linear layers WITHOUT activation (collapses to a single linear map)."""
    hidden = x @ params["W1"] + params["b1"]
    return hidden @ params["W2"] + params["b2"]


def forward_mlp(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Two-layer MLP WITH non-linear GELU activation."""
    hidden = jax.nn.gelu(x @ params["W1"] + params["b1"])
    return hidden @ params["W2"] + params["b2"]


def mlp_loss(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    """Cross-entropy loss for the 2-layer MLP."""
    logits = forward_mlp(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(x.shape[0]), y])


def linear_stack_loss(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    """Cross-entropy loss for the linear stack without activation."""
    logits = forward_linear_stack(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(x.shape[0]), y])


@partial(jax.jit, static_argnames=("n_steps",))
def train_both_models(
    init_params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    n_steps: int = 350,
    lr: float = 0.18,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, jax.Array]:
    """Train the linear stack and the GELU MLP in parallel using jax.lax.scan."""
    mlp_vg = jax.value_and_grad(mlp_loss)
    lin_vg = jax.value_and_grad(linear_stack_loss)

    def step_fn(
        carry: tuple[dict[str, jax.Array], dict[str, jax.Array]],
        _: None,
    ) -> tuple[tuple[dict[str, jax.Array], dict[str, jax.Array]], tuple[jax.Array, jax.Array]]:
        p_lin, p_mlp = carry
        l_lin, g_lin = lin_vg(p_lin, x, y)
        l_mlp, g_mlp = mlp_vg(p_mlp, x, y)
        p_lin_next = jax.tree.map(lambda p, g: p - lr * g, p_lin, g_lin)
        p_mlp_next = jax.tree.map(lambda p, g: p - lr * g, p_mlp, g_mlp)
        return (p_lin_next, p_mlp_next), (l_lin, l_mlp)

    (final_lin, final_mlp), (lin_losses, mlp_losses) = jax.lax.scan(
        step_fn, (init_params, init_params), None, length=n_steps
    )
    return final_lin, final_mlp, lin_losses[-1], mlp_losses[-1]


def accuracy(logits: jax.Array, labels: jax.Array) -> float:
    return float(jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)))


def distinct_hidden_units(w1: jax.Array) -> int:
    """How many hidden neurons have different incoming weights (columns of W1)."""
    return int(jnp.unique(jnp.round(w1.T, 6), axis=0).shape[0])


def run_experiment(n_per_class: int = 500, n_steps: int = 350) -> dict[str, Any]:
    """Verify linear collapse and train the 2-layer GELU MLP on concentric rings."""
    key = jax.random.key(2026)
    k_data, k_init = jax.random.split(key)

    features, labels = generate_concentric_rings(k_data, n_per_class=n_per_class)
    # Held-out rings: same distribution, different key, never used for gradients
    test_x, test_y = generate_concentric_rings(jax.random.key(4040), n_per_class=n_per_class)
    init_params = init_mlp_params(k_init, in_dim=2, hidden_dim=64, out_dim=3)

    # 1. Verify algebraic collapse: (X @ W1) @ W2 == X @ (W1 @ W2)
    unactivated_two_layer = (features @ init_params["W1"]) @ init_params["W2"]
    collapsed_w = init_params["W1"] @ init_params["W2"]
    single_matrix_out = features @ collapsed_w
    linear_collapse_max_diff = float(jnp.max(jnp.abs(unactivated_two_layer - single_matrix_out)))

    init_logits = forward_mlp(init_params, features)
    init_acc = float(jnp.mean((jnp.argmax(init_logits, axis=-1) == labels).astype(jnp.float32)))

    # Warmup (the first call traces and compiles), then median of 5 timed runs
    f_lin, f_mlp, l_lin, l_mlp = train_both_models(init_params, features, labels, n_steps=n_steps)
    l_mlp.block_until_ready()
    run_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = train_both_models(init_params, features, labels, n_steps=n_steps)
        jax.block_until_ready(out)
        run_ms.append((time.perf_counter() - t0) * 1000.0)

    lin_acc = accuracy(forward_linear_stack(f_lin, features), labels)
    mlp_acc = accuracy(forward_mlp(f_mlp, features), labels)
    lin_test_acc = accuracy(forward_linear_stack(f_lin, test_x), test_y)
    mlp_test_acc = accuracy(forward_mlp(f_mlp, test_x), test_y)

    # 2. Symmetry: the same MLP trained from all-zero weights never differentiates its neurons
    zero_params = jax.tree.map(jnp.zeros_like, init_params)
    _, z_mlp, _, _ = train_both_models(zero_params, features, labels, n_steps=n_steps)
    zero_init_acc = accuracy(forward_mlp(z_mlp, features), labels)

    return {
        "experiment": "04_two_layer_mlp_from_scratch",
        "backend": jax.default_backend(),
        "n_samples": int(features.shape[0]),
        "hidden_dim": 64,
        "n_steps": n_steps,
        "linear_collapse_max_diff": linear_collapse_max_diff,
        "initial_accuracy": round(init_acc * 100.0, 2),
        "linear_stack_accuracy": round(lin_acc * 100.0, 2),
        "mlp_gelu_accuracy": round(mlp_acc * 100.0, 2),
        "linear_stack_test_accuracy": round(lin_test_acc * 100.0, 2),
        "mlp_gelu_test_accuracy": round(mlp_test_acc * 100.0, 2),
        "n_test": int(test_x.shape[0]),
        "linear_stack_final_loss": round(float(l_lin), 4),
        "mlp_gelu_final_loss": round(float(l_mlp), 4),
        "he_init_distinct_hidden_units": distinct_hidden_units(f_mlp["W1"]),
        "zero_init_distinct_hidden_units": distinct_hidden_units(z_mlp["W1"]),
        "zero_init_accuracy": round(zero_init_acc * 100.0, 2),
        "median_train_ms": round(statistics.median(run_ms), 4),
    }


def main() -> None:
    res = run_experiment()
    print("=" * 74)
    print("Experiment 04: Two-Layer MLP from Scratch (Linear Collapse vs GELU MLP)")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(f"Non-Linear Dataset          : {res['n_samples']} samples (3 concentric rings)")
    print(
        f"Linear Collapse Max Diff    : {res['linear_collapse_max_diff']:.2e} "
        "((X@W1)@W2 vs X@(W1@W2))"
    )
    print("-" * 74)
    print(f"Initial Random Accuracy     : {res['initial_accuracy']:.2f}%")
    print(
        f"2-Layer Linear (No Activ.)  : train {res['linear_stack_accuracy']:.2f}% | "
        f"test {res['linear_stack_test_accuracy']:.2f}%"
    )
    print(
        f"2-Layer MLP (GELU, 64 hid.) : train {res['mlp_gelu_accuracy']:.2f}% | "
        f"test {res['mlp_gelu_test_accuracy']:.2f}%"
    )
    print("-" * 74)
    print(
        f"He Init (random, scaled)    : {res['he_init_distinct_hidden_units']} of "
        f"{res['hidden_dim']} hidden units distinct"
    )
    print(
        f"Zero Init (all W = 0)       : {res['zero_init_distinct_hidden_units']} of "
        f"{res['hidden_dim']} distinct | train {res['zero_init_accuracy']:.2f}%"
    )
    print(
        f"Median @jit + lax.scan Train: {res['median_train_ms']:.4f} ms "
        f"(2 nets x {res['n_steps']} steps)"
    )


if __name__ == "__main__":
    main()
