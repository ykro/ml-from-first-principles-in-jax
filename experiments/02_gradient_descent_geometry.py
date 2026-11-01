"""Experiment 02: How Gradient Descent Walks (SGD vs Momentum vs Adam in Pure PyTrees).

Demonstrates for Software Engineers:
1. Why vanilla Gradient Descent (`SGD`) struggles on ill-conditioned loss surfaces
   where one parameter dimension is 50x steeper than another (oscillating across
   the narrow canyon while creeping slowly along the valley floor).
2. How `Momentum` accumulates an exponential moving average of gradients (`v`) to
   dampen cross-canyon oscillations and accelerate along consistent directions.
3. How `Adam` combines first-moment (`m`) and second-moment (`v`) gradient estimates
   with bias correction so every parameter receives its own adaptive learning rate.
4. All 3 optimizers implemented in ~30 lines of pure JAX `jax.tree.map` without
   external libraries.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def ill_conditioned_loss(
    params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Anisotropic regression loss with condition number ~50 across feature scales."""
    preds = features @ params["w"] + params["b"]
    residual = preds - targets
    return 0.5 * jnp.mean(jnp.square(residual))


def sgd_step(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    lr: float = 0.015,
) -> dict[str, jax.Array]:
    """Vanilla Stochastic Gradient Descent update over any PyTree."""
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def momentum_step(
    params: dict[str, jax.Array],
    velocity: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    lr: float = 0.015,
    beta: float = 0.9,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Polyak Momentum update over any PyTree."""
    new_velocity = jax.tree.map(lambda v, g: beta * v + g, velocity, grads)
    new_params = jax.tree.map(lambda p, v: p - lr * v, params, new_velocity)
    return new_params, new_velocity


def adam_step(
    params: dict[str, jax.Array],
    m: dict[str, jax.Array],
    v: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    t: jax.Array,
    lr: float = 0.08,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """Adam optimizer step from first principles using jax.tree.map."""
    new_m = jax.tree.map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, m, grads)
    new_v = jax.tree.map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * jnp.square(g_i), v, grads)
    m_hat = jax.tree.map(lambda m_i: m_i / (1.0 - jnp.power(beta1, t)), new_m)
    v_hat = jax.tree.map(lambda v_i: v_i / (1.0 - jnp.power(beta2, t)), new_v)
    new_params = jax.tree.map(
        lambda p_i, mh_i, vh_i: p_i - lr * mh_i / (jnp.sqrt(vh_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, new_m, new_v


@partial(jax.jit, static_argnames=("n_steps",))
def run_compiled_optimizers(
    init_params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    n_steps: int = 180,
) -> dict[str, jax.Array]:
    """Run SGD, Momentum, and Adam side-by-side inside a single XLA compiled graph."""
    loss_and_grad = jax.value_and_grad(ill_conditioned_loss)

    # States
    p_sgd = init_params
    p_mom = init_params
    vel_mom = jax.tree.map(jnp.zeros_like, init_params)
    p_adam = init_params
    m_adam = jax.tree.map(jnp.zeros_like, init_params)
    v_adam = jax.tree.map(jnp.zeros_like, init_params)

    def body_fn(carry: tuple[Any, ...], step_idx: jax.Array) -> tuple[tuple[Any, ...], jax.Array]:
        ps, pm, vm, pa, ma, va = carry
        l_s, g_s = loss_and_grad(ps, features, targets)
        l_m, g_m = loss_and_grad(pm, features, targets)
        l_a, g_a = loss_and_grad(pa, features, targets)

        ps_next = sgd_step(ps, g_s, lr=0.025)
        pm_next, vm_next = momentum_step(pm, vm, g_m, lr=0.025, beta=0.9)
        pa_next, ma_next, va_next = adam_step(pa, ma, va, g_a, step_idx, lr=0.35)
        return (ps_next, pm_next, vm_next, pa_next, ma_next, va_next), jnp.stack([l_s, l_m, l_a])

    init_carry = (p_sgd, p_mom, vel_mom, p_adam, m_adam, v_adam)
    steps = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    final_carry, loss_history = jax.lax.scan(body_fn, init_carry, steps)
    return {
        "loss_history": loss_history,
        "sgd_final": ill_conditioned_loss(final_carry[0], features, targets),
        "momentum_final": ill_conditioned_loss(final_carry[1], features, targets),
        "adam_final": ill_conditioned_loss(final_carry[3], features, targets),
    }


def run_experiment(n_samples: int = 1024, n_steps: int = 180) -> dict[str, Any]:
    """Benchmark SGD vs Momentum vs Adam on an ill-conditioned problem."""
    key = jax.random.key(7)
    raw_x = jax.random.normal(key, shape=(n_samples, 8), dtype=jnp.float32)
    # Feature scales span 1.0 to 0.05 (creating an elongated ravine in weight space)
    scales = jnp.geomspace(8.0, 0.15, num=8, dtype=jnp.float32)
    features = raw_x * scales

    true_w = jnp.array([2.5, -3.0, 4.0, -5.5, 8.0, -11.0, 14.0, -18.0], dtype=jnp.float32)
    true_b = jnp.array(3.2, dtype=jnp.float32)
    targets = features @ true_w + true_b

    init_params = {
        "w": jnp.zeros((8,), dtype=jnp.float32),
        "b": jnp.array(0.0, dtype=jnp.float32),
    }
    initial_loss = float(ill_conditioned_loss(init_params, features, targets))

    # Warmup compiled scan
    out = run_compiled_optimizers(init_params, features, targets, n_steps=n_steps)
    out["adam_final"].block_until_ready()

    # Measure compiled XLA execution
    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = run_compiled_optimizers(init_params, features, targets, n_steps=n_steps)
        out["adam_final"].block_until_ready()
    compiled_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Measure eager step loop for comparison (single run of 30 steps scaled to 180)
    loss_and_grad = jax.value_and_grad(ill_conditioned_loss)
    t0 = time.perf_counter()
    p_e = init_params
    m_e = jax.tree.map(jnp.zeros_like, init_params)
    v_e = jax.tree.map(jnp.zeros_like, init_params)
    for idx in range(1, 31):
        _, g_e = loss_and_grad(p_e, features, targets)
        p_e, m_e, v_e = adam_step(p_e, m_e, v_e, g_e, jnp.array(float(idx)), lr=0.12)
    p_e["w"].block_until_ready()
    eager_adam_180_ms = (time.perf_counter() - t0) * 1000.0 * (n_steps / 30.0)

    history = out["loss_history"]
    threshold = 0.05

    def steps_to_reach(col: int) -> int:
        mask = history[:, col] < threshold
        reached = bool(jnp.any(mask))
        return int(jnp.argmax(mask) + 1) if reached else n_steps

    sgd_steps = steps_to_reach(0)
    mom_steps = steps_to_reach(1)
    adam_steps = steps_to_reach(2)

    sgd_final = float(out["sgd_final"])
    mom_final = float(out["momentum_final"])
    adam_final = float(out["adam_final"])

    return {
        "experiment": "02_gradient_descent_geometry",
        "backend": jax.default_backend(),
        "n_steps": n_steps,
        "initial_loss": round(initial_loss, 4),
        "sgd_final_loss": round(sgd_final, 6),
        "momentum_final_loss": round(mom_final, 6),
        "adam_final_loss": round(adam_final, 6),
        "sgd_steps_to_0_05": sgd_steps,
        "momentum_steps_to_0_05": mom_steps,
        "adam_steps_to_0_05": adam_steps,
        "eager_loop_ms": round(eager_adam_180_ms, 4),
        "compiled_scan_ms": round(compiled_ms, 4),
        "speedup": round(eager_adam_180_ms / compiled_ms, 2),
    }


def main() -> None:
    res = run_experiment()
    print("=" * 74)
    print("Experiment 02: Gradient Descent Geometry (SGD vs Momentum vs Adam)")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(f"Initial Ill-Conditioned Loss: {res['initial_loss']:.4f}")
    print("-" * 74)
    print(
        f"Vanilla SGD (180 steps)     : final loss = {res['sgd_final_loss']:.6f} "
        f"| steps to <0.05 = {res['sgd_steps_to_0_05']}"
    )
    print(
        f"Momentum    (180 steps)     : final loss = {res['momentum_final_loss']:.6f} "
        f"| steps to <0.05 = {res['momentum_steps_to_0_05']}"
    )
    print(
        f"Adam        (180 steps)     : final loss = {res['adam_final_loss']:.6f} "
        f"| steps to <0.05 = {res['adam_steps_to_0_05']}"
    )
    print("-" * 74)
    print(f"Eager Python Optimizer Loop : {res['eager_loop_ms']:.4f} ms")
    print(f"@jit + lax.scan Optimizer   : {res['compiled_scan_ms']:.4f} ms")
    print(f"Compiled Speedup            : {res['speedup']:.2f}x")


if __name__ == "__main__":
    main()
