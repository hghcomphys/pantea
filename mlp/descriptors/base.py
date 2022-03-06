
class Descriptor:
  """
  A base descriptor class.
  All descriptors must derive from this class.
  """
  @property
  def r_cutoff(self) -> float:
    """
    Return the cutoff radius for the derived descriptor.
    This will be used to extract the maximum cutoff radius required for updating the neighbor list.
    """
    raise NotImplementedError
