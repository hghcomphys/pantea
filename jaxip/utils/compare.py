from typing import Any, Dict, List, Union

import jax.numpy as jnp

from jaxip.atoms.structure import Structure
from jaxip.types import Array


def compare(
    structure1: Structure,
    structure2: Structure,
    errors: Union[str, List] = "RMSEpa",
    return_difference: bool = False,
) -> Dict[str, Array]:
    """
    Compare the `force` and `total energy` values
    between two input structures and returning the desired errors metrics.

    :param structure1: first structure
    :param structure2: second structure
    :param error: a list of error metrics including `RMSE`, `RMSEpa`, `MSE`, and `MSEpa`. Defaults to [`RMSEpa`]
    :param return_difference: whether return energy and force array differences or not, defaults to False
    :return: a dictionary of error metrics.
    """
    assert all(
        structure1.atom_types == structure2.atom_types
    ), "Expected similar structures with the same atom types."

    result: Dict[str, Any] = dict()
    frc_diff: Array = structure1.forces - structure2.forces
    eng_diff: Array = structure1.total_energy - structure2.total_energy
    errors = [errors] if isinstance(errors, str) else errors
    print(f"Comparing two structures, error metrics: {', '.join(errors)}")
    errors = [x.lower() for x in errors]

    if "rmse" in errors:
        result["force_RMSE"] = jnp.sqrt(jnp.mean(frc_diff**2))
        result["energy_RMSE"] = jnp.sqrt(jnp.mean(eng_diff**2))
    if "rmsepa" in errors:
        result["force_RMSEpa"] = jnp.sqrt(jnp.mean(frc_diff**2))
        result["energy_RMSEpa"] = jnp.sqrt(jnp.mean(eng_diff**2)) / structure1.natoms
    if "mse" in errors:
        result["force_MSE"] = jnp.mean(frc_diff**2)
        result["energy_MSE"] = jnp.mean(eng_diff**2)
    if "msepa" in errors:
        result["force_MSEpa"] = jnp.mean(frc_diff**2)
        result["energy_MSEpa"] = jnp.mean(eng_diff**2) / structure1.natoms
    if return_difference:
        result["frc_diff"] = frc_diff
        result["eng_diff"] = eng_diff

    return result
