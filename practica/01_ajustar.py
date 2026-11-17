import jax
import jax.numpy as jnp

k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.uniform(k1, (2000, 2))  # uso de CPU y de RAM, de 0 a 1
ruido = jax.random.normal(k2, (2000,))
y = 120 + x @ jnp.array([30.0, -4.0]) + ruido  # latencia en ms


def error(p):
    pred = x @ p["w"] + p["b"]
    return jnp.mean((pred - y) ** 2)


@jax.jit
def paso(p):
    e, g = jax.value_and_grad(error)(p)
    return jax.tree.map(lambda v, d: v - 0.5 * d, p, g), e


p = {"w": jnp.zeros(2), "b": jnp.array(0.0)}
for i in range(251):
    p, e = paso(p)
    if i % 50 == 0:
        print(f"paso {i:3d}  error {e:9.2f}")
print("w =", p["w"], " b =", p["b"])
