"""
Provide the Tokenizer class to tokenize lines.
"""

import re
import numpy as np

SPECIAL_TOKENS = {'PAD_TOKEN': u'_pad_',
                  'UNKNOWN_TOKEN': u'_unk_',
                  'START_TOKEN': u'_start_',
                  'EOS_TOKEN': u'_end_'}

TOKEN_REGEX = re.compile(r'\w+|[^\w\s]')


class Tokenizer:
    def __init__(self, token_to_index, line_len, context_len):
        self.token_to_index = token_to_index
        self.line_len = line_len
        self.context_len = context_len
        self._regexp = TOKEN_REGEX

    def tokenize_line(self, line, add_terminators=False):
        """
        Transforms lines of text to matrix of indices of tokens.
        :param line: Line to transform to ids
        """
        line = self._regexp.findall(line.lower())

        tokens = np.full(
            self.line_len,
            self.token_to_index[SPECIAL_TOKENS['PAD_TOKEN']],
            dtype=np.int32)

        if add_terminators:
            line = [SPECIAL_TOKENS["START_TOKEN"]] \
                + line + [SPECIAL_TOKENS["EOS_TOKEN"]]

        for token_idx, token in enumerate(line[:self.line_len]):
            tokens[token_idx] = self.token_to_index[token] \
                if token in self.token_to_index else \
                self.token_to_index[SPECIAL_TOKENS['UNKNOWN_TOKEN']]

        return tokens

    def tokenize_context(self, context):
        """
        Transforms a context to a list of matrices of indices of tokens.
        :param context: Context to transform to ids
        """

        if len(context) < self.context_len:
            context = [''] * (self.context_len - len(context)) + context
        elif len(context) > self.context_len:
            context = context[-self.context_len:]

        data = map(self.tokenize_line, context)
        return np.asarray(list(data))
