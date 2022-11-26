import jax.numpy as jnp


# TODO: define Loss classes, can cause performance penalty
def mse_loss(*, logits: jnp.ndarray, labels: jnp.ndarray):
    return ((labels - logits) ** 2).mean()
