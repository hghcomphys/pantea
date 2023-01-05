from jaxip.types import Array


# TODO: define Loss classes, can cause performance penalty
def mse_loss(*, logits: Array, targets: Array) -> Array:
    return ((targets - logits) ** 2).mean()
