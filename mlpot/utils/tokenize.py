from typing import List, Optional, Tuple


def tokenize(
    line: str,
    comment: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    """An utility function to read the input line as a keyword and list of tokens."""
    # Skip comments
    if comment is not None:
        if line.startswith(comment):
            return (None, list())
        else:
            line = line[: line.find(comment)]
    # Find keyword and values
    tokens: List[str] = line.rstrip("/n").split()
    if len(tokens) > 1:
        return (tokens[0].lower(), tokens[1:])
    elif len(tokens) > 0:
        return (tokens[0].lower(), list())
    else:
        return (None, list())
