from cakechat.utils.text_processing.str_processor import get_tokens_sequence, replace_out_of_voc_tokens, \
    get_pretty_str_from_tokens_sequence
from cakechat.utils.text_processing.config import SPECIAL_TOKENS
from cakechat.utils.text_processing.utils import get_processed_corpus_path, get_index_to_token_path, \
    get_index_to_condition_path, load_index_to_item
from cakechat.utils.text_processing.dialog import get_flatten_dialogs, get_alternated_dialogs_lines, \
    get_dialog_lines_and_conditions
from cakechat.utils.text_processing.corpus_iterator import FileTextLinesIterator, ProcessedLinesIterator
