from typing import Tuple, List


def tokenize(line: str, comment: str = None) -> Tuple[str, List[str]]:
  """
  An utility method to read the input line as a keyword and list of tokens.
  """
  _null = (None, None)
  # Skip comments
  if comment is not None:
    if line.startswith(comment):
      return _null
    else:
      line = line[:line.find(comment)]
  # Find keyword and values
  tokens = line.rstrip("/n").split()
  if len(tokens) > 1:
    return (tokens[0].lower(), tokens[1:])
  elif len(tokens) > 0:
    return (tokens[0].lower(), None)
  else:
    return _null