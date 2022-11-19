import jax.numpy as jnp

Tensor = jnp.ndarray


# TODO: define Loss classes, can cause performance penalty
def mse_loss(*, logits: Tensor, labels: Tensor):
    return ((labels - logits) ** 2).mean()
