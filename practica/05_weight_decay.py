import jax
import jax.numpy as jnp

# 28 métricas por incidente; solo las 4 primeras importan
w_real = jnp.zeros(28).at[:4].set(jnp.array([1.2, -0.9, 0.8, -0.7]))
k = jax.random.split(jax.random.key(0), 4)
x = jax.random.normal(k[0], (32, 28))  # 32 incidentes para entrenar
y = x @ w_real + jax.random.normal(k[1], (32,))
x_test = jax.random.normal(k[2], (1000, 28))  # 1000 que el modelo nunca ve
y_test = x_test @ w_real + jax.random.normal(k[3], (1000,))


def mse(w, x, y):
    return jnp.mean((x @ w - y) ** 2)


def entrenar(wd):
    def perdida(w):
        return mse(w, x, y) + 0.5 * wd * jnp.sum(w**2)

    paso = jax.jit(lambda w: w - 0.05 * jax.grad(perdida)(w))
    w = jnp.zeros(28)
    for _ in range(2000):
        w = paso(w)
    e, e_test = mse(w, x, y), mse(w, x_test, y_test)
    print(f"wd={wd}  train {e:.2f}  test {e_test:.2f}  norma {jnp.linalg.norm(w):.2f}")


entrenar(0.0)
entrenar(1.0)
