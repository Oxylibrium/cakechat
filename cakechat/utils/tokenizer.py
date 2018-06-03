"""
Provide the Tokenizer class to tokenize lines.
"""

import re
from typing import Dict, List, Iterable, Optional

import numpy as np

SPECIAL_TOKENS = {
    "PAD_TOKEN": u"_pad_",
    "UNKNOWN_TOKEN": u"_unk_",
    "START_TOKEN": u"_start_",
    "EOS_TOKEN": u"_end_",
}

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")

# TODO: Document this.
class Tokenizer:

    def __init__(
        self, token_to_index: Dict[str, str], line_len: int, context_len: int
    ):
        self.token_to_index = token_to_index
        self.line_len = line_len
        self.context_len = context_len

    @staticmethod
    def preprocess_line(line: str) -> List[str]:
        """
        Converts line to lowercase and splits it into a list of token strings.

        Parameters
        ----------
        line:
            Line to split.
        """
        return TOKEN_REGEX.findall(line.lower())

    def tokenize_line(
        self, line: str, add_terminators: Optional[bool] = False
    ) -> np.ndarray:
        """
        Transforms lines of text to matrix of indices of tokens.

        Parameters
        ----------
        line: str
            Line to tokenize.
        add_terminators: bool
            If true, pad starting and ending with START_TOKEN and EOS_TOKEN.
        """
        line = self.preprocess_line(line)

        tokens = np.full(
            self.line_len,
            self.token_to_index[SPECIAL_TOKENS["PAD_TOKEN"]],
            dtype=np.int32,
        )

        if add_terminators:
            line = (
                [SPECIAL_TOKENS["START_TOKEN"]]
                + line
                + [SPECIAL_TOKENS["EOS_TOKEN"]]
            )

        for token_idx, token in enumerate(line[: self.line_len]):
            tokens[token_idx] = (
                self.token_to_index[token]
                if token in self.token_to_index
                else self.token_to_index[SPECIAL_TOKENS["UNKNOWN_TOKEN"]]
            )

        return tokens

    def tokenize_context(
        self, context: List[str], add_terminators: Optional[bool] = False
    ) -> np.ndarray:
        """
        Transforms a context to a list of matrices of indices of tokens.

        Parameters
        ----------
        context:
            Context to transform to ids.
        """
        if len(context) < self.context_len:
            context = [""] * (self.context_len - len(context)) + context
        elif len(context) > self.context_len:
            context = context[-self.context_len :]

        data = np.zeros((self.context_len, self.line_len))

        for i, line in enumerate(context):
            data[i] = self.tokenize_line(line, add_terminators)

        return data

    def tokenize_lines(
        self,
        iterable: Iterable[str],
        length: Optional[int] = None,
        add_terminators: Optional[bool] = False,
    ) -> Iterable[np.ndarray]:
        """
        Tokenize a list or iterable of lines.
        Roughly similar to np.asarray(list(map(tokenize_line, iterable))),
        but preallocates the array.

        Parameters
        ----------
        iterable:
            Iterable or list of lines.
        length:
            Length to use, defaults to len(iterable).
        """
        if not length:
            length = len(iterable)

        lines = np.zeros((length, self.line_len), dtype=np.int32)

        number = 0
        for number, line in enumerate(iterable):
            lines[number] = self.tokenize_line(line, add_terminators)

        if number < len:
            raise ValueError

        return lines

    def tokenize_contexts(
        self,
        iterable: Iterable[List[str]],
        length: Optional[int] = None,
        add_terminators: Optional[bool] = False,
    ):
        """
        Tokenize a list or iterable of contexts.
        Roughly similar to np.asarray(list(map(tokenize_context, iterable))),
        but preallocates the array.

        Parameters
        ----------
        iterable:
            Iterable or list of contexts.
        length:
            Length to use, defaults to len(iterable).
        """
        if not length:
            length = len(iterable)

        contexts = np.zeros((length, self.context_len, self.line_len), dtype=np.int32)

        number = 0
        for number, context in enumerate(iterable):
            contexts[number] = self.tokenize_context(context, add_terminators)

        if number < len:
            raise ValueError

        return contexts
