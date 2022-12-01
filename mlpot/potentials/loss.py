import jax.numpy as jnp


# TODO: define Loss classes, can cause performance penalty
def mse_loss(*, logits: jnp.ndarray, targets: jnp.ndarray):
    return ((targets - logits) ** 2).mean()
