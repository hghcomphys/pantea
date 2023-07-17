from __future__ import annotations

from typing import Any, Dict, Tuple

from jax import tree_util

from jaxip.logger import logger


class BaseJaxPytreeDataClass:
    """
    Here we specify exactly which components of a `dataclass` should be treated as
    static and which should be treated as dynamic (array) attributes.

    The `hash` method then it would only based on the static attributes and
    used for JIT re-compilation if they change.

    .. warning::

        It must be noted that we have to define first dynamic values (as tuple)
        and then static values (as dict) in child class constructor (same for dataclasses).

        Registering the class to jax pytree node is also required.

        Must be used only for dataclasses.

    See https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
    """

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        children: Tuple[Any, ...] = tuple(
            getattr(self, attr) for attr in self._get_jit_dynamic_attributes()
        )
        aux_data: Dict[str, Any] = {
            attr: getattr(self, attr)
            for attr in self._get_jit_static_attributes()
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple[Any, ...]
    ) -> BaseJaxPytreeDataClass:
        return cls(*children, **aux_data)  # type: ignore

    def __hash__(self) -> int:
        """
        Define hash based on the `JAX JIT compilation` static attributes.
        This is used for detecting changes on the static arguments
        of the jit-compiled method.
        """
        aux_data: Dict[str, Any] = self._tree_flatten()[1]
        return hash(tuple(frozenset(sorted(aux_data.items()))))

    @classmethod
    def _get_jit_dynamic_attributes(cls) -> Tuple[str, ...]:
        """Get JAX JIT compilation dynamic attribute names (i.e. jax.ndarray)."""
        return tuple(
            attr
            for attr, dtype in cls.__annotations__.items()
            if "Array" in str(dtype)
        )

    @classmethod
    def _get_jit_static_attributes(cls) -> Tuple[str, ...]:
        """Get JAX JIT compilation static attribute names."""
        dynamic_attributes = cls._get_jit_dynamic_attributes()
        return tuple(
            attr
            for attr in cls.__annotations__.keys()
            if attr not in dynamic_attributes
        )

    @classmethod
    def _assert_jit_attributes(
        cls,
        available: Tuple[str, ...],
        expected: Tuple[str, ...],
        tag: str = "",
    ) -> None:
        """Assert to ensure jit (static or dynamics) attributes are correctly identified."""
        if sorted(available) != sorted(expected):  # type: ignore
            logger.error(
                f"JIT {tag} attributes: expected {expected} but got {available}",
                exception=AssertionError,
            )

    @classmethod
    def _assert_jit_static_attributes(
        cls, expected: Tuple[str, ...] = tuple()
    ) -> None:
        cls._assert_jit_attributes(
            cls._get_jit_static_attributes(), expected, tag="static"
        )

    @classmethod
    def _assert_jit_dynamic_attributes(
        cls, expected: Tuple[str, ...] = tuple()
    ) -> None:
        cls._assert_jit_attributes(
            cls._get_jit_dynamic_attributes(), expected, tag="dynamic"
        )


def register_jax_pytree_node(cls) -> None:
    """Register the input class as internal JAX pytree node."""
    tree_util.register_pytree_node(
        cls, cls._tree_flatten, cls._tree_unflatten  # type: ignore
    )
