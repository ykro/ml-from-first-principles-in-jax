import jax
import jax.numpy as jnp


def softmax(logits):
    e = jnp.exp(logits)
    return e / e.sum()


logits = jnp.array([10.0, 20.0, 120.0])  # normal, DDoS, fuga de memoria
print("exp:    ", jnp.exp(logits))
print("nuestra:", softmax(logits))
print("de JAX: ", jax.nn.softmax(logits))
