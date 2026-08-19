from eval.build import build_and_write_held_out_eval, build_held_out_eval_set
from eval.compare_chunking import compare_chunking_strategies
from eval.metrics import (
    aggregate_scores,
    aggregate_scores_by_language,
    evaluate_examples,
    evaluate_examples_by_language,
    hit_at_k,
    mrr,
    recall_at_k,
)
from eval.runner import run_eval
from eval.significance import bootstrap_mean_diff, compare_preset_scores, score_examples
from eval.split import (
    DEFAULT_SPLIT,
    EvalSplitConfig,
    build_shuffle_permutation,
    get_corpus_row_indices,
    get_dev_row_indices,
    get_eval_row_indices,
    get_shuffle_permutation,
    get_validation_rows,
    load_split_config,
    resolve_slice_row_indices,
    write_split_config,
)
from eval.validate import validate_eval_coverage, validate_eval_file

__all__ = [
    "DEFAULT_SPLIT",
    "EvalSplitConfig",
    "aggregate_scores",
    "aggregate_scores_by_language",
    "bootstrap_mean_diff",
    "build_and_write_held_out_eval",
    "build_held_out_eval_set",
    "build_shuffle_permutation",
    "compare_chunking_strategies",
    "compare_embedding_presets",
    "compare_preset_scores",
    "evaluate_examples",
    "evaluate_examples_by_language",
    "get_corpus_row_indices",
    "get_dev_row_indices",
    "get_eval_row_indices",
    "get_shuffle_permutation",
    "get_validation_rows",
    "hit_at_k",
    "load_split_config",
    "mrr",
    "recall_at_k",
    "resolve_slice_row_indices",
    "run_eval",
    "score_examples",
    "validate_eval_coverage",
    "validate_eval_file",
    "write_split_config",
]
