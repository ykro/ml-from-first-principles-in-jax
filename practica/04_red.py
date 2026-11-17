import jax
import jax.numpy as jnp

k = jax.random.split(jax.random.key(0), 5)
y = jax.random.randint(k[0], (900,), 0, 3)  # anillo 0, 1 o 2
radio = 1.0 + y + 0.15 * jax.random.normal(k[1], (900,))
angulo = jax.random.uniform(k[2], (900,), maxval=2 * jnp.pi)
x = jnp.stack([radio * jnp.cos(angulo), radio * jnp.sin(angulo)], axis=1)


def red(p, x):
    h = jax.nn.gelu(x @ p["W1"] + p["b1"])  # 64 neuronas ocultas
    return h @ p["W2"] + p["b2"]  # 3 logits, uno por anillo


def error(p):  # cross-entropy
    logp = jax.nn.log_softmax(red(p, x))
    return -jnp.mean(logp[jnp.arange(900), y])


p = {"W1": jax.random.normal(k[3], (2, 64)), "b1": jnp.zeros(64),
     "W2": jax.random.normal(k[4], (64, 3)) / 8, "b2": jnp.zeros(3)}
grad = jax.jit(jax.grad(error))
for _ in range(1000):
    p = jax.tree.map(lambda v, d: v - 0.5 * d, p, grad(p))
print(f"aciertos: {jnp.mean(red(p, x).argmax(axis=1) == y):.1%}")
