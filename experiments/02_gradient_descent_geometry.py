"""Experiment 02: How Gradient Descent Walks (SGD vs Momentum vs Adam in Pure PyTrees).

Demonstrates for Software Engineers:
1. Why vanilla Gradient Descent (`SGD`) struggles on ill-conditioned loss surfaces
   (feature scales from 8.0 down to 0.15, so the curvature of the steepest direction is
   thousands of times larger than the flattest one): it oscillates across the narrow
   canyon while creeping slowly along the valley floor.
2. How `Momentum` accumulates an exponential moving average of gradients (`v`) to
   dampen cross-canyon oscillations and accelerate along consistent directions.
3. How `Adam` combines first-moment (`m`) and second-moment (`v`) gradient estimates
   with bias correction so every parameter receives its own adaptive learning rate.
4. All 3 optimizers implemented in ~30 lines of pure JAX `jax.tree.map` without
   external libraries.

How to read the numbers: the main run uses hand-picked learning rates; the sweep re-runs
each optimizer at its best learning rate from the same grid (the fair comparison). The
timing runs the SAME work (3 optimizers x 180 steps) three ways: eager op-by-op (naive
baseline), a Python loop over a @jax.jit step (fair baseline) and one @jax.jit + lax.scan.
Times are medians after a warmup and vary run to run.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def ill_conditioned_loss(
    params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Anisotropic regression loss: feature scales from 8.0 down to 0.15 (a ravine)."""
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


SGD_LR, MOMENTUM_LR, ADAM_LR = 0.025, 0.025, 0.35


def init_state(init_params: dict[str, jax.Array]) -> tuple[Any, ...]:
    """(p_sgd, p_mom, vel_mom, p_adam, m_adam, v_adam), all optimizers start at init_params."""
    zeros = jax.tree.map(jnp.zeros_like, init_params)
    return (init_params, init_params, zeros, init_params, zeros, zeros)


def three_optimizer_step(
    carry: tuple[Any, ...],
    step_idx: jax.Array,
    features: jax.Array,
    targets: jax.Array,
) -> tuple[tuple[Any, ...], jax.Array]:
    """One step of SGD, Momentum and Adam side by side (the unit of work every timing runs)."""
    loss_and_grad = jax.value_and_grad(ill_conditioned_loss)
    ps, pm, vm, pa, ma, va = carry
    l_s, g_s = loss_and_grad(ps, features, targets)
    l_m, g_m = loss_and_grad(pm, features, targets)
    l_a, g_a = loss_and_grad(pa, features, targets)

    ps_next = sgd_step(ps, g_s, lr=SGD_LR)
    pm_next, vm_next = momentum_step(pm, vm, g_m, lr=MOMENTUM_LR, beta=0.9)
    pa_next, ma_next, va_next = adam_step(pa, ma, va, g_a, step_idx, lr=ADAM_LR)
    return (ps_next, pm_next, vm_next, pa_next, ma_next, va_next), jnp.stack([l_s, l_m, l_a])


@partial(jax.jit, static_argnames=("n_steps",))
def run_compiled_optimizers(
    init_params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    n_steps: int = 180,
) -> dict[str, jax.Array]:
    """Run SGD, Momentum, and Adam side-by-side inside a single XLA compiled graph."""
    def body_fn(carry: tuple[Any, ...], step_idx: jax.Array) -> tuple[tuple[Any, ...], jax.Array]:
        return three_optimizer_step(carry, step_idx, features, targets)

    steps = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    final_carry, loss_history = jax.lax.scan(body_fn, init_state(init_params), steps)
    return {
        "loss_history": loss_history,
        "sgd_final": ill_conditioned_loss(final_carry[0], features, targets),
        "momentum_final": ill_conditioned_loss(final_carry[1], features, targets),
        "adam_final": ill_conditioned_loss(final_carry[3], features, targets),
    }


def run_experiment(
    n_samples: int = 1024,
    n_steps: int = 180,
    timing: bool = True,
) -> dict[str, Any]:
    """Benchmark SGD vs Momentum vs Adam on an ill-conditioned problem."""
    features, targets = make_ravine_dataset(n_samples)
    init_params = {
        "w": jnp.zeros((8,), dtype=jnp.float32),
        "b": jnp.array(0.0, dtype=jnp.float32),
    }
    initial_loss = float(ill_conditioned_loss(init_params, features, targets))

    # Curvature of the loss: Hessian of 0.5*mean(r^2) is X^T X / n (with a bias column)
    x_aug = jnp.concatenate([features, jnp.ones((n_samples, 1), dtype=jnp.float32)], axis=1)
    eigs = jnp.linalg.eigvalsh(x_aug.T @ x_aug / n_samples)
    condition_number = float(eigs[-1] / eigs[0])

    out = run_compiled_optimizers(init_params, features, targets, n_steps=n_steps)
    history = out["loss_history"]

    res: dict[str, Any] = {
        "experiment": "02_gradient_descent_geometry",
        "backend": jax.default_backend(),
        "n_steps": n_steps,
        "initial_loss": round(initial_loss, 4),
        "hessian_condition_number": round(condition_number, 1),
        "learning_rates": {"sgd": SGD_LR, "momentum": MOMENTUM_LR, "adam": ADAM_LR},
        "sgd_final_loss": round(float(out["sgd_final"]), 6),
        "momentum_final_loss": round(float(out["momentum_final"]), 6),
        "adam_final_loss": round(float(out["adam_final"]), 6),
        # None = never crossed the threshold within n_steps
        "sgd_steps_to_0_05": steps_to_reach(history[:, 0], LOSS_THRESHOLD),
        "momentum_steps_to_0_05": steps_to_reach(history[:, 1], LOSS_THRESHOLD),
        "adam_steps_to_0_05": steps_to_reach(history[:, 2], LOSS_THRESHOLD),
        "lr_sweep": learning_rate_sweep(init_params, features, targets, n_steps),
    }
    if timing:
        res.update(benchmark_three_ways(init_params, features, targets, n_steps))
    return res


LOSS_THRESHOLD = 0.05
# 31 log-spaced learning rates from 0.001 to 1.0, plus the three hand-picked ones
LR_GRID = sorted(
    {round(float(x), 4) for x in jnp.geomspace(0.001, 1.0, 31)} | {SGD_LR, MOMENTUM_LR, ADAM_LR}
)


def make_ravine_dataset(n_samples: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.key(7)
    raw_x = jax.random.normal(key, shape=(n_samples, 8), dtype=jnp.float32)
    # Feature scales span 8.0 down to 0.15 (an elongated ravine in weight space)
    scales = jnp.geomspace(8.0, 0.15, num=8, dtype=jnp.float32)
    features = raw_x * scales
    true_w = jnp.array([2.5, -3.0, 4.0, -5.5, 8.0, -11.0, 14.0, -18.0], dtype=jnp.float32)
    true_b = jnp.array(3.2, dtype=jnp.float32)
    return features, features @ true_w + true_b


def steps_to_reach(losses: jax.Array, threshold: float) -> int | None:
    """First step (1-based) whose loss is below threshold, or None if never reached."""
    mask = losses < threshold
    return int(jnp.argmax(mask) + 1) if bool(jnp.any(mask)) else None


def learning_rate_sweep(
    init_params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    n_steps: int,
) -> dict[str, dict[str, Any]]:
    """Fair comparison: each optimizer alone, at its best learning rate from the same grid."""
    loss_and_grad = jax.value_and_grad(ill_conditioned_loss)
    zeros = jax.tree.map(jnp.zeros_like, init_params)
    steps = jnp.arange(1, n_steps + 1, dtype=jnp.float32)

    def make_run(kind: str) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
        def run(lr: jax.Array) -> tuple[jax.Array, jax.Array]:
            def body(carry: tuple[Any, ...], t: jax.Array) -> tuple[tuple[Any, ...], jax.Array]:
                p, a, b = carry
                loss, g = loss_and_grad(p, features, targets)
                if kind == "sgd":
                    p = sgd_step(p, g, lr=lr)
                elif kind == "momentum":
                    p, a = momentum_step(p, a, g, lr=lr, beta=0.9)
                else:
                    p, a, b = adam_step(p, a, b, g, t, lr=lr)
                return (p, a, b), loss

            (p, _, _), losses = jax.lax.scan(body, (init_params, zeros, zeros), steps)
            return ill_conditioned_loss(p, features, targets), losses

        return run

    grid = jnp.array(LR_GRID, dtype=jnp.float32)
    best: dict[str, dict[str, Any]] = {}
    for kind in ("sgd", "momentum", "adam"):
        finals, histories = jax.jit(jax.vmap(make_run(kind)))(grid)
        finals = jnp.where(jnp.isfinite(finals), finals, jnp.inf)  # diverged runs lose
        i = int(jnp.argmin(finals))
        best[kind] = {
            "best_lr": LR_GRID[i],
            "final_loss": round(float(finals[i]), 6),
            "steps_to_0_05": steps_to_reach(histories[i], LOSS_THRESHOLD),
        }
    return best


def _median_ms(fn: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def benchmark_three_ways(
    init_params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    n_steps: int,
) -> dict[str, Any]:
    """Same work (3 optimizers x n_steps) three ways; median ms per full run after a warmup."""
    step_ids = [jnp.array(float(i), dtype=jnp.float32) for i in range(1, n_steps + 1)]
    jit_step = jax.jit(three_optimizer_step)

    def eager_loop() -> Any:
        carry = init_state(init_params)
        for t in step_ids:
            carry, _ = three_optimizer_step(carry, t, features, targets)
        return carry

    def jit_step_loop() -> Any:
        carry = init_state(init_params)
        for t in step_ids:
            carry, _ = jit_step(carry, t, features, targets)
        return carry

    def scan_once() -> Any:
        return run_compiled_optimizers(init_params, features, targets, n_steps=n_steps)

    # Warmups: eager needs every op seen once (2 steps); the jitted versions compile here
    carry = init_state(init_params)
    for t in step_ids[:2]:
        carry, _ = three_optimizer_step(carry, t, features, targets)
    jax.block_until_ready(carry)
    jax.block_until_ready(jit_step_loop())
    jax.block_until_ready(scan_once())

    repeats = {"eager": 3, "jit_step_loop": 15, "scan": 30}
    eager_ms = _median_ms(eager_loop, repeats["eager"])
    jit_loop_ms = _median_ms(jit_step_loop, repeats["jit_step_loop"])
    scan_ms = _median_ms(scan_once, repeats["scan"])
    return {
        "eager_loop_ms": round(eager_ms, 4),
        "jit_step_loop_ms": round(jit_loop_ms, 4),
        "compiled_scan_ms": round(scan_ms, 4),
        "speedup_vs_eager": round(eager_ms / scan_ms, 2),
        "speedup_vs_jit_step_loop": round(jit_loop_ms / scan_ms, 2),
        "timing_repeats": repeats,
    }


def _steps(value: int | None, n_steps: int) -> str:
    return str(value) if value is not None else f"none in {n_steps}"


def main() -> None:
    res = run_experiment()
    n = res["n_steps"]
    lrs = res["learning_rates"]
    print("=" * 74)
    print("Experiment 02: Gradient Descent Geometry (SGD vs Momentum vs Adam)")
    print("=" * 74)
    print(f"Backend Device              : {res['backend']}")
    print(f"Initial Ill-Conditioned Loss: {res['initial_loss']:.4f}")
    print(f"Hessian Condition Number    : {res['hessian_condition_number']:.1f}")
    print("-" * 74)
    for name, key in (("SGD", "sgd"), ("Momentum", "momentum"), ("Adam", "adam")):
        print(
            f"{name:<8} lr={lrs[key]:<6}       : loss = {res[key + '_final_loss']:.6f} "
            f"| steps to <0.05 = {_steps(res[key + '_steps_to_0_05'], n)}"
        )
    print("-" * 74)
    print(f"Fair check: each optimizer at its best lr from the same {len(LR_GRID)}-value grid")
    for name, key in (("SGD", "sgd"), ("Momentum", "momentum"), ("Adam", "adam")):
        b = res["lr_sweep"][key]
        print(
            f"{name:<8} best lr={b['best_lr']:<7}  : loss = {b['final_loss']:.6f} "
            f"| steps to <0.05 = {_steps(b['steps_to_0_05'], n)}"
        )
    print("-" * 74)
    print(f"Same work ({n} steps x 3 optimizers), median ms per run after warmup:")
    print(f"Eager op-by-op loop (naive) : {res['eager_loop_ms']:.4f} ms")
    print(f"Python loop of @jit step    : {res['jit_step_loop_ms']:.4f} ms")
    print(f"@jit + lax.scan (1 call)    : {res['compiled_scan_ms']:.4f} ms")
    print(f"Scan vs eager loop          : {res['speedup_vs_eager']:.2f}x")
    print(f"Scan vs loop of @jit step   : {res['speedup_vs_jit_step_loop']:.2f}x (fair)")


if __name__ == "__main__":
    main()
