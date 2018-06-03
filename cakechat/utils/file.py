"""
Iterate over and process lines in file.
"""

import io
import os


def _do_nothing(line):
    """
    Default callback for the iterator, does nothing and returns line.
    """
    return line


class LinesIterator:
    """
    Iterate over lines in a text file,
    optionally processing each line with `callback`

    Attributes
    ----------
    path: path of the file
    callback: callable used for formatting

    Usage
    -----
    ```
    with LinesIterator(path, lambda x: json.loads(x)) as iterator:
        for line in iterator:
            pass
    ```
    """

    def __init__(self, path, callback=_do_nothing):
        # type: str, Callable[[str], Any]
        self._len = None
        self._fp = None
        self.path = path
        self.callback = callback

    def __len__(self):
        if self._len:
            return self._len
        with open(self.path) as file:
            length = 0
            for _ in file:
                length += 1
            self._len = length
            return length

    def __enter__(self):
        self._fp = io.open(self.path, encoding="UTF-8")
        return self

    def __exit__(self, *_):
        self._fp.close()

    def __iter__(self):
        if not self._fp:
            raise AttributeError("File wasn't opened yet.")

        for line in self._fp:
            yield self.callback(line)


def ensure_dir(path):
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    path : str
        Path to the directory.
    """
    if path and not os.path.exists(path):
        os.makedirs(path)


def is_non_empty_file(path):
    """
    Check if a file exists and is not empty.

    Parameters
    ----------
    path : str
        Path to the file.
    """
    return os.path.isfile(path) and os.stat(path).st_size != 0
