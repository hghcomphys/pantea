from __future__ import annotations

from typing import Any, Dict, Tuple

from jax import tree_util


class _BaseJaxPytreeDataClass:
    """
    Here we specify exactly which components of a `dataclass` should be treated as
    static and which should be treated as dynamic (array) values.

    The `hash` method then it would only based on the static attributes and
    used for JIT recompiling.

    It must be noted that we have to define first dynamic values (as tuple)
    and then static values (as dict) in child class constructor (same for dataclasses).

    Registering the class to jax pytree node is required.

    ..warning::
        Must be used only for dataclasses, and class attributes must have typings.

    Correctly JIT-compiling a class method
    See https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
    """

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        children: Tuple[Any, ...] = tuple(
            getattr(self, attr) for attr in self._get_jit_dynamic_attributes()
        )
        aux_data: Dict[str, Any] = {
            attr: getattr(self, attr) for attr in self._get_jit_static_attributes()
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple[Any, ...]
    ) -> _BaseJaxPytreeDataClass:
        return cls(*children, **aux_data)  # type: ignore

    @classmethod
    def _get_jit_dynamic_attributes(cls) -> Tuple[str, ...]:
        """Get JAX JIT compilation dynamic attribute names (i.e. jax.ndarray)."""
        return tuple(
            attr for attr, dtype in cls.__annotations__.items() if "Array" in str(dtype)
        )

    @classmethod
    def _get_jit_static_attributes(cls) -> Tuple[str, ...]:
        """Get JAX JIT compilation static attribute names."""
        return tuple(
            attr
            for attr in cls.__annotations__.keys()
            if attr not in cls._get_jit_dynamic_attributes()
        )

    def __hash__(self) -> int:
        """
        Define hash method based on the `JAX JIT compilation` static attributes.
        This is useful when wants to capture changes on the static argument of a jit-compiled method.
        """
        aux_data: Dict[str, Any] = self._tree_flatten()[1]
        return hash(tuple(frozenset(sorted(aux_data.items()))))


def register_jax_pytree_node(cls) -> None:
    """Register the input class as internal JAX pytree node."""
    tree_util.register_pytree_node(
        cls, cls._tree_flatten, cls._tree_unflatten  # type: ignore
    )
