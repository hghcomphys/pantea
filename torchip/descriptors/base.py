from ..base import BaseTorchipClass


class Descriptor(BaseTorchipClass):
    """
    A base descriptor class.
    All descriptors must derive from this class.
    """

    def __init__(self, element: str) -> None:
        self.element = element

    @property
    def r_cutoff(self) -> float:
        """
        Return the cutoff radius for the derived descriptor.
        This will be used to extract the maximum cutoff radius required for updating the neighbor list.
        """
        raise NotImplementedError
