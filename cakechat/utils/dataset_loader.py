import json
import os
import warning
from itertools import islice

import numpy as np

from cakechat.config import (CONTEXT_FREE_VAL_CORPUS_NAME,
                             CONTEXT_SENSITIVE_VAL_CORPUS_NAME,
                             DEFAULT_CONDITION, INPUT_CONTEXT_SIZE,
                             INPUT_SEQUENCE_LENGTH, QUESTIONS_CORPUS_NAME,
                             RANDOM_SEED, TEST_DATA_DIR, TRAIN_CORPUS_NAME,
                             TRAIN_SUBSET_SIZE)
from cakechat.dialog_model.model_utils import (Dataset, lines_to_context,
                                               transform_conditions_to_nn_input,
                                               transform_contexts_to_token_ids,
                                               transform_lines_to_nn_input)
from cakechat.utils.files_utils import is_non_empty_file, load_file
from cakechat.utils.logger import get_logger
from cakechat.utils.text_processing import (FileTextLinesIterator,
                                            ProcessedLinesIterator,
                                            get_alternated_dialogs_lines,
                                            get_dialog_lines_and_conditions,
                                            get_processed_corpus_path,
                                            get_tokens_sequence)
from cakechat.utils.tokenizer import Tokenizer

_logger = get_logger(__name__)


def get_tokenized_test_lines(corpus_name, token_to_index):
    corpus_path = os.path.join(TEST_DATA_DIR, '%s.txt' % corpus_name)
    if not is_non_empty_file(corpus_path):
        raise ValueError(
            'Test corpus file doesn\'t exist: {}'.format(corpus_path))
    test_lines = load_file(corpus_path)
    tokenizer = Tokenizer(
        token_to_index,
        INPUT_SEQUENCE_LENGTH,
        INPUT_CONTEXT_SIZE)

    result = list(map(tokenizer.tokenize_line, test_lines))
    return result


def _load_dataset_without_responses(corpus_name, token_to_index):
    tokenized_lines = get_tokenized_test_lines(corpus_name, token_to_index)
    # TODO: Single context
    context_tokens_ids = transform_contexts_to_token_ids(
        lines_to_context(tokenized_lines),
        token_to_index,
        INPUT_SEQUENCE_LENGTH,
        INPUT_CONTEXT_SIZE,
        max_contexts_num=len(tokenized_lines))
    return Dataset(x=context_tokens_ids, y=None, condition_ids=None)


def load_questions_set(token_to_index):
    return _load_dataset_without_responses(
        QUESTIONS_CORPUS_NAME, token_to_index)


def load_context_free_val(corpus_name, tokenize_line):
    corpus_path = get_processed_corpus_path(corpus_name)
    with open(corpus_path) as file:
        while True:
            x = tokenize_line(file.readline().strip())
            y = tokenize_line(file.readline().strip())
            if x and y:
                yield (x, y)
            else:
                break


# NOTE: old name load_context_sensitive_val
def load_contextual_set(filename, tokenizer, condition_to_idx, subset_size=None):
    corpus_path = get_processed_corpus_path(filename)
    corpus = FileTextLinesIterator(corpus_path)

    if subset_size:
        corpus = islice(corpus, subset_size)

    for ctx in corpus:
        ctx = json.loads(ctx)
        texts = [item['text'] for item in ctx]
        conditions = [item['condition'] for item in ctx]
        for n in range(1, len(texts)):
            x_ids = tokenizer.tokenize_context(texts[:n])
            y_ids = tokenizer.tokenize_line(texts[n], add_terminators=True)
            condition = condition_to_idx.get(conditions[n], 0)
            yield x_ids, y_ids, condition


def load_conditioned_train_set(
        token_to_index,
        condition_to_index,
        train_subset_size=TRAIN_SUBSET_SIZE):
    processed_corpus_path = get_processed_corpus_path(TRAIN_CORPUS_NAME)

    warning(DeprecationWarning("Replace with load_contextual_set"))

    def load_processed_dialogs_from_json(lines, text_field_name, condition_field_name):
        for line_json in lines:
            line_json = json.loads(line_json)
            yield [{
                text_field_name: entry['text'],
                condition_field_name: entry['condition']
            } for entry in line_json]

    dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(processed_corpus_path),
        text_field_name='text', condition_field_name='condition')

    if train_subset_size:
        dialogs = islice(dialogs, train_subset_size)

    train_lines, train_conditions = get_dialog_lines_and_conditions(
        get_alternated_dialogs_lines(dialogs),
        text_field_name='text', condition_field_name='condition')

    tokenized_alternated_train_lines = ProcessedLinesIterator(
        train_lines, processing_callbacks=[get_tokens_sequence])

    # prepare train set
    x_train, y_train, n_dialogs = transform_lines_to_nn_input(
        tokenized_alternated_train_lines, token_to_index)

    condition_ids_train = transform_conditions_to_nn_input(
        train_conditions, condition_to_index, n_dialogs)

    return Dataset(x=x_train, y=y_train, condition_ids=condition_ids_train)


def generate_subset(dataset, subset_size, random_seed=RANDOM_SEED):
    # Fix random seed here so that we get the same subsets every time the
    # function is called
    np.random.seed(random_seed)
    if subset_size > dataset.x.shape[0]:
        raise ValueError(
            'Error while generating subset of the validation data: \
            dataset size is less then subset size.')

    sample_idx = np.random.choice(
        dataset.x.shape[0],
        size=subset_size,
        replace=False)

    return Dataset(
        x=dataset.x[sample_idx],
        y=dataset.y[sample_idx] if dataset.y is not None else None,
        condition_ids=dataset.condition_ids[sample_idx]
        if dataset.condition_ids is not None else None)


def load_datasets(token_to_index, condition_to_index):
    train = load_conditioned_train_set(token_to_index, condition_to_index)
    validation = load_context_free_val(token_to_index)
    questions = load_questions_set(token_to_index)

    validation_set_size = validation.x.shape[0]

    train_subset = generate_subset(train, validation_set_size)

    # prepare conditioned subset
    defined_condition_mask = \
        train.condition_ids != condition_to_index[DEFAULT_CONDITION]

    defined_condition_dataset = Dataset(
        x=train.x[defined_condition_mask],
        y=train.y[defined_condition_mask],
        condition_ids=train.condition_ids[defined_condition_mask])

    defined_condition_dataset_len = defined_condition_dataset.x.shape[0]
    defined_condition_subset = generate_subset(
        defined_condition_dataset,
        min(validation_set_size, defined_condition_dataset_len))

    return train, questions, validation, train_subset, defined_condition_subset
