import jax
import jax.numpy as jnp


def error(w):  # un cañón: muy empinado en w[0], casi plano en w[1]
    return 50 * w[0] ** 2 + 0.5 * w[1] ** 2


def sgd(w, s, g):
    return w - 0.019 * g, s


def momentum(w, v, g):
    v = 0.9 * v + g
    return w - 0.019 * v, v


def adam(w, s, g):
    m, v, t = s
    m, v, t = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g**2, t + 1
    paso = (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return w - 0.1 * paso, (m, v, t)


cero = jnp.zeros(2)
for opt, s in [(sgd, None), (momentum, cero), (adam, (cero, cero, 0))]:
    w = jnp.array([1.0, 1.0])
    for _ in range(100):
        w, s = opt(w, s, jax.grad(error)(w))
    print(f"{opt.__name__:9s} error {error(w):.5f}  w = {w}")
