import jax
import jax.numpy as jnp


def log_softmax(logits):
    z = logits - logits.max()  # el mayor queda en 0 y exp(0) es 1
    return z - jnp.log(jnp.exp(z).sum())


logits = jnp.array([10.0, 20.0, 120.0])
print("nuestra:", log_softmax(logits))
print("de JAX: ", jax.nn.log_softmax(logits))
print("probabilidades:", jnp.exp(log_softmax(logits)))
