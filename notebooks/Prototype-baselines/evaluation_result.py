"""A script of evaluating the Hallucination Detection using Avg-Log-Probability and Avg-Monte-Carlo-Similarity
"""

# %% import packages
import typing as ty
from pathlib import Path

import pickle
import zlib
import json

import matplotlib.axes
import numpy as np
import pandas as pd

import logzero

import seaborn as sns
import matplotlib.pyplot as plot
from summary_abstractive import visualization_header

logger = logzero.logger

# %%

try:
    import sklearn
except ImportError:
    raise ImportError("Install first `scikit-learn`.")
# end try

# %% Path to resource inputs

PATH_DIR_AVG_LOG_PROB = "/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache/FaiseqTranslationModelHandlerVer2WordEmbeddings/beam/1.0"
PATH_DIR_AVG_MONTE_CARLO_SIM = ""

path_dir_avg_log_prob = Path(PATH_DIR_AVG_LOG_PROB)
assert path_dir_avg_log_prob.exists()


PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()
path_dataset_cnn = PATH_DATASET_CNN


THRESHOLD_AVG_LOG_PROB = 40
THRESHOLD_AVG_MONTE_CARLO_SIM = 40

threshold_avg_log_prob = THRESHOLD_AVG_LOG_PROB

# %% functions

# loading the dataset

class UniqueDocumentId(ty.NamedTuple):
    datasource: str  # ex. cnn_dailymail
    document_id_original: str  # ex. 0
    abstractiveness_constraint: str  # ex. none

    def to_str(self) -> str:
        if '/' in self.abstractiveness_constraint:
            abstractiveness_constraint = self.abstractiveness_constraint.replace("/", "-inverse-")
        else:
            abstractiveness_constraint = self.abstractiveness_constraint
        # else
        return f'{self.datasource}-{self.document_id_original}-{abstractiveness_constraint}'
# end class


def load_dataset(path_dataset: Path) -> ty.Dict[UniqueDocumentId, ty.Dict]:
    with path_dataset.open('r') as f:
        seq_dataset_obj = [json.loads(_line) for _line in f.readlines()]
    # end with


    # allocating the unique id to all records
    dict_unique_id2records = {
        UniqueDocumentId(datasource=_obj['dataset_name'], document_id_original=_obj['document_id'], abstractiveness_constraint=_obj['abstractiveness_constraint']): _obj 
        for _obj in seq_dataset_obj
    }
    logger.info(f'Count Unique document -> {len(dict_unique_id2records)}')

    return dict_unique_id2records


# loading the log-prob files

def load_cache_files(path_dir_avg_log_prob: Path) -> ty.Dict[str, ty.Dict]:
    """Loading the cached object. The cached object refers to `TranslationResultContainer`.
    The `log_probability_score` filed saves the average log probability.
    """

    logger.debug("Loading cached files having the log probability...")

    seq_list_files = list(path_dir_avg_log_prob.rglob("*pkl.zlib"))
    assert len(seq_list_files) > 0

    dict_document_id2cache = {}

    for _path_zlib in seq_list_files:
        with _path_zlib.open('rb') as f:
            _cache_obj = pickle.loads(zlib.decompress(f.read()))
        # end with
        _document_id = _path_zlib.name.replace('.pkl.zlib', '')

        dict_document_id2cache[_document_id] = _cache_obj
    # end for
    assert len(dict_document_id2cache) > 0

    logger.debug(f"Loading done. {len(dict_document_id2cache)} documents are ready.")

    return dict_document_id2cache


import numpy.typing as npt
from sklearn.metrics import confusion_matrix

class EvaluationResultContainer(ty.NamedTuple):
    n_total: int
    n_target_label: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    precision: float
    recall: float
    f1: float


def _func_get_evaluation(
        array_label_prediction: npt.NDArray[np.int8],
        array_label_ground_truth: npt.NDArray[np.int8]
        ) -> EvaluationResultContainer:
    """An abstract function to get the evaluation metric."""
    # confusion matrix
    # computing elements of a confusion matrix.
    matrix_confusion = confusion_matrix(array_label_ground_truth, array_label_prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix_confusion.ravel()

    # computing p, r, f1
    # evaluation metrics
    if tp == 0:
        precision = 0.0
        recall = 0.0
        f_score = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * (precision * recall) / (precision + recall)

        if np.isnan(precision):
            precision = 0.0
        else:
            precision = float(precision)
        if np.isnan(recall):
            recall = 0.0
        else:
            recall = float(recall)
        if np.isnan(f_score):
            f_score = 0.0
        else:
            f_score = float(f_score)
        # end if
    # end if

    return EvaluationResultContainer(
        n_total=len(array_label_ground_truth),
        n_target_label=int(np.sum(array_label_ground_truth)),
        true_positive=int(tp),
        false_positive=int(fp),
        true_negative=int(tn),
        false_negative=int(fn),
        precision=precision,
        recall=recall,
        f1=f_score)


def _generate_ground_truth_label(dict_document_id2record: ty.Dict[UniqueDocumentId, ty.Dict], label_mode: str = '<3') -> ty.Dict[str, int]:
    """The dataset has no labels of hallucnation or not. I set a convertion rule of making the voting score into a binary label."""
    dict_document_id2gold_label = {}
    for _key_obj, _record in dict_document_id2record.items():
        assert 'annotator_votes' in _record
        _voting_annotations: ty.List[int] = _record['annotator_votes']
        
        if label_mode == '<3':
            _gold_label = 1 if sum(_voting_annotations) < 3 else 0
        else:
            raise ValueError()
        # end if
        dict_document_id2gold_label[_key_obj.to_str()] = _gold_label
    # end for

    _simple_stats: float = sum(dict_document_id2gold_label.values()) / len(dict_document_id2gold_label)
    logger.info(f"Generated gold labels with mode={label_mode}. Hallucination record {_simple_stats * 100}%")

    return dict_document_id2gold_label


import matplotlib

def _make_precision_recall_curve(seq_evaluation_records_container: ty.List[EvaluationResultContainer]):
    """Generating the precision-recall curve for MMD Flagger Ver1.
    
    This method plots the precision-recall curve. 
    """

    seq_precision = []
    seq_recall = []
    for _eval_record in seq_evaluation_records_container:
        _precision = _eval_record.precision
        _recall = _eval_record.recall
        seq_precision.append(_precision)
        seq_recall.append(_recall)
    # end for

    # Note: I can not compute the Area-Under-Curve because the precision is not
    # # computing the Area-Under-Curve.
    # precision = np.array(seq_precision)
    # recall = np.array(seq_recall)
    # # score_auc = auc(precision, recall)
    # score_auc = 0.0

    # I want to visualise X: recall, Y: precision.
    f, ax = plot.subplots()
    sns.lineplot(x=seq_recall, y=seq_precision, ax=ax)
    ax.set_title(f"Precision-Recall Curve.")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')

    return f, ax



# %% main, setting the global variables.

dict_document_id2record = load_dataset(path_dataset_cnn)

# %%

# TODO: I need to set the example-id selection. The MMD-Flagger used the subset.

path_dir_output = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/results/log_prob")
path_dir_output.mkdir(parents=True, exist_ok=True)

path_dir_analysis_files = path_dir_output / 'analysis_files'
path_dir_analysis_files.mkdir(parents=True, exist_ok=True)

# main procedure for avg log prob

# loading the cache files
dict_document_id2dict_obj = load_cache_files(path_dir_avg_log_prob)
# making a dict {document-id: avg-log-score}
dict_document_id2avg_log_prob: ty.Dict[str, float] = {_doc_id: _obj['log_probability_score'] for _doc_id, _obj in dict_document_id2dict_obj.items()}
# setting the threshold.
_threshold = np.percentile(np.array(list(dict_document_id2avg_log_prob.values())), q=threshold_avg_log_prob)

# getting the ground-truth label
dict_document_id2ground_truth_label = _generate_ground_truth_label(dict_document_id2record, label_mode="<3")

# getting the prediction labels
dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold else 0 for _key_obj, _score in dict_document_id2avg_log_prob.items()}


def get_evaluation_arrays(dict_document_idprediction_label: ty.Dict[str, int],
                          dict_document_id2ground_truth_label: ty.Dict[str, int]) -> EvaluationResultContainer:
    """Getting the evaluation scores. 
    
    Args:
        dict_document_idprediction_label: document-id 2 prediction binary label, 
        dict_document_id2ground_truth_label: document-id 2 ground-truth binary label,         
    """

    # giving the binary lavel
    _array_prediction = []
    _array_gold = []
    for _document_id in dict_document_idprediction_label:
        _array_prediction.append(dict_document_idprediction_label[_document_id])
        _array_gold.append(dict_document_id2ground_truth_label[_document_id])
    # end for

    eval_container = _func_get_evaluation(np.array(_array_prediction), np.array(_array_gold))

    return eval_container
# end def

eval_container_fixed_threshold = get_evaluation_arrays(dict_document_idprediction_label, dict_document_id2ground_truth_label)


# ROC curve and AUC score

n_plot_eval = 1000
_threshold_percentiles = np.arange(0, 1.0, step=n_plot_eval)
seq_eval_container = []
for _percentile in _threshold_percentiles:
    _threshold = np.percentile(np.array(list(dict_document_id2avg_log_prob.values())), q=_percentile)
    _dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold else 0 for _key_obj, _score in dict_document_id2avg_log_prob.items()}
    _eval_container = get_evaluation_arrays(_dict_document_idprediction_label, dict_document_id2ground_truth_label)
    seq_eval_container.append(_eval_container)
# end for
_f, _ax = _make_precision_recall_curve(seq_eval_container)

# TODO png, to csv

# visualization and file outcome

# visualizing the histogram
_f, _ax = plot.subplots()
sns.histplot(np.array(list(dict_document_id2avg_log_prob.values())), ax=_ax)
# TODO, setting the vertical line at the threshold.
_path_png_score_histogram = path_dir_analysis_files / 'score_histogram.png'
_f.savefig(_path_png_score_histogram.as_posix(), dpi=300, bbox_inches='tight')

# 
_path_json_document_id2avg_log_prob = path_dir_analysis_files / 'document_id2avg_log.json'
with _path_json_document_id2avg_log_prob.open('w') as f:
    f.write(json.dumps(dict_document_id2avg_log_prob, indent=4))
# end with
