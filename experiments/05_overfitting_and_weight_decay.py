"""Experiment 05: The Silent Enemy in ML (Overfitting, Train/Val Split & L2 Weight Decay).

Demonstrates for Software Engineers:
1. Why evaluating a model on the same data used to compute gradients (`train set`)
   is equivalent to running unit tests only on hardcoded mocks.
2. How an over-parameterized model (4 true telemetry signals + 24 spurious noisy
   metrics on a small window of 32 training incidents) memorizes training noise,
   driving `train_mse` near zero while `val_mse` explodes (`overfitting`).
3. How combining a strict `Train/Val Split`, `L2 Weight Decay` (`0.5 * wd * sum(W^2)`),
   and a `Validation Gate` (early stopping on validation loss) suppresses spurious
   weights and reduces validation error.

How to read the numbers: the Validation Gate picks its checkpoint by looking at the
validation set, so the validation MSE of that checkpoint is optimistic. The honest number
is the MSE on a separate test set (1024 incidents, different key) that no decision used.
The ablation trains weight decay alone and the gate alone to show what each one buys.
Validation and test targets carry noise 0.10 (train: 1.15), so their MSE floor is ~0.01
and their MSE mostly measures how far the learned weights are from the true ones.
"""

from __future__ import annotations

import statistics
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def true_incident_weights(n_features: int = 28) -> tuple[jax.Array, jax.Array]:
    """Only the first 4 metrics are causal; the other 24 have true weight 0."""
    true_w = jnp.zeros((n_features,), dtype=jnp.float32)
    true_w = true_w.at[:4].set(jnp.array([1.2, -0.9, 0.8, -0.7], dtype=jnp.float32))
    return true_w, jnp.array(0.5, dtype=jnp.float32)


def generate_test_incidents(
    key: jax.Array,
    n_test: int = 1024,
    n_features: int = 28,
) -> tuple[jax.Array, jax.Array]:
    """Held-out test incidents (same distribution and noise as validation)."""
    k_x, k_n = jax.random.split(key)
    x_test = jax.random.normal(k_x, (n_test, n_features), dtype=jnp.float32)
    true_w, true_b = true_incident_weights(n_features)
    y_test = x_test @ true_w + true_b + 0.10 * jax.random.normal(k_n, (n_test,), dtype=jnp.float32)
    return x_test, y_test


def generate_high_dim_incident_dataset(
    key: jax.Array,
    n_train: int = 32,
    n_val: int = 256,
    n_features: int = 28,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generate 32 training incidents and 256 validation incidents across 28 metrics.

    Only the first 4 metrics carry true causal signal; the remaining 24 metrics
    are spurious uncorrelated dashboard metrics.
    """
    k_xtr, k_xval, k_ntr, k_nval = jax.random.split(key, 4)
    x_train = jax.random.normal(k_xtr, (n_train, n_features), dtype=jnp.float32)
    x_val = jax.random.normal(k_xval, (n_val, n_features), dtype=jnp.float32)

    true_w, true_b = true_incident_weights(n_features)

    y_train = (
        x_train @ true_w
        + true_b
        + 1.15 * jax.random.normal(k_ntr, (n_train,), dtype=jnp.float32)
    )
    y_val = (
        x_val @ true_w
        + true_b
        + 0.10 * jax.random.normal(k_nval, (n_val,), dtype=jnp.float32)
    )
    return x_train, y_train, x_val, y_val


def mse_eval(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    """Pure data Mean Squared Error without regularization penalty."""
    preds = x @ params["w"] + params["b"]
    return jnp.mean(jnp.square(preds - y))


def regularized_loss(
    params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    weight_decay: float,
) -> jax.Array:
    """Data MSE + L2 Weight Decay penalty on weights w (excluding bias b)."""
    data_loss = mse_eval(params, x, y)
    l2_penalty = 0.5 * weight_decay * jnp.sum(jnp.square(params["w"]))
    return data_loss + l2_penalty


@partial(jax.jit, static_argnames=("n_steps",))
def run_training_regime(
    init_params: dict[str, jax.Array],
    x_train: jax.Array,
    y_train: jax.Array,
    x_val: jax.Array,
    y_val: jax.Array,
    weight_decay: float,
    n_steps: int = 800,
    lr: float = 0.05,
) -> dict[str, jax.Array]:
    """Train with Adam + optional L2 Weight Decay and track best validation checkpoint."""
    vg_fn = jax.value_and_grad(regularized_loss)

    m0 = jax.tree.map(jnp.zeros_like, init_params)
    v0 = jax.tree.map(jnp.zeros_like, init_params)
    init_val = mse_eval(init_params, x_val, y_val)

    def body_fn(carry: tuple[Any, ...], step_idx: jax.Array) -> tuple[tuple[Any, ...], jax.Array]:
        params, m, v, best_params, best_val_mse = carry
        _, grads = vg_fn(params, x_train, y_train, weight_decay)

        new_m = jax.tree.map(lambda mi, gi: 0.9 * mi + 0.1 * gi, m, grads)
        new_v = jax.tree.map(lambda vi, gi: 0.999 * vi + 0.001 * jnp.square(gi), v, grads)
        m_hat = jax.tree.map(lambda mi: mi / (1.0 - jnp.power(0.9, step_idx)), new_m)
        v_hat = jax.tree.map(lambda vi: vi / (1.0 - jnp.power(0.999, step_idx)), new_v)
        new_params = jax.tree.map(
            lambda pi, mhi, vhi: pi - lr * mhi / (jnp.sqrt(vhi) + 1e-8),
            params,
            m_hat,
            v_hat,
        )

        tr_mse = mse_eval(new_params, x_train, y_train)
        vl_mse = mse_eval(new_params, x_val, y_val)

        improved = vl_mse < best_val_mse
        next_best_val = jnp.where(improved, vl_mse, best_val_mse)
        next_best_params = jax.tree.map(
            lambda bp, np_: jnp.where(improved, np_, bp),
            best_params,
            new_params,
        )
        carry_out = (new_params, new_m, new_v, next_best_params, next_best_val)
        return carry_out, jnp.stack([tr_mse, vl_mse])

    steps = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    init_carry = (init_params, m0, v0, init_params, init_val)
    final_carry, history = jax.lax.scan(body_fn, init_carry, steps)
    final_params, _, _, best_params, best_val_mse = final_carry

    return {
        "final_train_mse": history[-1, 0],
        "final_val_mse": history[-1, 1],
        "best_val_mse": best_val_mse,
        "final_weight_l2_norm": jnp.linalg.norm(final_params["w"]),
        "best_weight_l2_norm": jnp.linalg.norm(best_params["w"]),
        "final_params": final_params,
        "best_params": best_params,
    }


def run_experiment() -> dict[str, Any]:
    """Compare unregularized overfitting against L2 Weight Decay + Validation Gate."""
    key = jax.random.key(99)
    x_train, y_train, x_val, y_val = generate_high_dim_incident_dataset(key)
    x_test, y_test = generate_test_incidents(jax.random.key(2027))
    n_features = x_train.shape[1]

    init_params = {
        "w": jnp.zeros((n_features,), dtype=jnp.float32),
        "b": jnp.array(0.0, dtype=jnp.float32),
    }

    def train(weight_decay: float) -> dict[str, jax.Array]:
        return run_training_regime(
            init_params, x_train, y_train, x_val, y_val, weight_decay=weight_decay, n_steps=800
        )

    # Warmup (traces and compiles once; weight_decay is a traced argument)
    jax.block_until_ready(train(0.0))
    run_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(train(0.0))
        run_ms.append((time.perf_counter() - t0) * 1000.0)

    overfit = train(0.0)
    reg = train(0.35)

    def test_mse(params: dict[str, jax.Array]) -> float:
        return float(mse_eval(params, x_test, y_test))

    overfit_train = float(overfit["final_train_mse"])
    overfit_val = float(overfit["final_val_mse"])
    overfit_gap = overfit_val / max(overfit_train, 1e-6)

    # The gated model IS the best-validation checkpoint: report train MSE of that checkpoint
    reg_train = float(mse_eval(reg["best_params"], x_train, y_train))
    reg_val = float(reg["best_val_mse"])
    val_mse_reduction = overfit_val / reg_val

    # Honest numbers on the untouched test set, plus the ablation of each ingredient
    test = {
        "unregularized_final": test_mse(overfit["final_params"]),
        "weight_decay_only_final": test_mse(reg["final_params"]),
        "val_gate_only_best": test_mse(overfit["best_params"]),
        "weight_decay_plus_gate_best": test_mse(reg["best_params"]),
    }
    true_w, true_b = true_incident_weights(n_features)
    test_noise_floor = test_mse({"w": true_w, "b": true_b})

    return {
        "experiment": "05_overfitting_and_weight_decay",
        "backend": jax.default_backend(),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": n_features,
        "overfit_train_mse": round(overfit_train, 4),
        "overfit_val_mse": round(overfit_val, 4),
        "overfit_val_to_train_ratio": round(overfit_gap, 2),
        "overfit_weight_norm": round(float(overfit["final_weight_l2_norm"]), 2),
        "regularized_train_mse": round(reg_train, 4),
        "regularized_val_mse": round(reg_val, 4),
        "regularized_weight_norm": round(float(reg["best_weight_l2_norm"]), 2),
        "val_mse_reduction_ratio": round(val_mse_reduction, 2),
        "test_mse": {k: round(v, 4) for k, v in test.items()},
        "test_noise_floor_mse": round(test_noise_floor, 4),
        "test_mse_reduction_ratio": round(
            test["unregularized_final"] / test["weight_decay_plus_gate_best"], 2
        ),
        "median_800_step_scan_ms": round(statistics.median(run_ms), 4),
    }


def main() -> None:
    res = run_experiment()
    t = res["test_mse"]
    print("=" * 74)
    print("Experiment 05: Overfitting, Train/Val Split & L2 Weight Decay")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(
        f"Dataset Split               : {res['n_train']} train | {res['n_val']} val | "
        f"{res['n_test']} test"
    )
    print(f"Features                    : {res['n_features']} (4 causal, 24 noisy)")
    print("-" * 74)
    print(
        f"Unregularized (wd=0.0)      : train {res['overfit_train_mse']:.4f} | "
        f"val {res['overfit_val_mse']:.4f} | ||w|| {res['overfit_weight_norm']:.2f}"
    )
    print(
        f"WD 0.35 + Val Gate          : train {res['regularized_train_mse']:.4f} | "
        f"val {res['regularized_val_mse']:.4f} | ||w|| {res['regularized_weight_norm']:.2f}"
    )
    print(f"Val/Train Gap (wd=0.0)      : {res['overfit_val_to_train_ratio']:.2f}x")
    print(f"Val MSE Reduction           : {res['val_mse_reduction_ratio']:.2f}x (gate saw val)")
    print("-" * 74)
    print("Test MSE (never used for any decision):")
    print(f"  Unregularized, final step : {t['unregularized_final']:.4f}")
    print(f"  Weight decay only, final  : {t['weight_decay_only_final']:.4f}")
    print(f"  Val gate only (wd=0.0)    : {t['val_gate_only_best']:.4f}")
    print(f"  Weight decay + val gate   : {t['weight_decay_plus_gate_best']:.4f}")
    print(f"  Noise floor (true w, b)   : {res['test_noise_floor_mse']:.4f}")
    print(f"Test MSE Reduction          : {res['test_mse_reduction_ratio']:.2f}x")
    print(f"Median 800-Step @jit Scan   : {res['median_800_step_scan_ms']:.4f} ms")


if __name__ == "__main__":
    main()
