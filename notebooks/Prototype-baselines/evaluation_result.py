"""A script of evaluating the Hallucination Detection using Avg-Log-Probability and Avg-Monte-Carlo-Similarity
"""

# %% import packages
import typing as ty
import matplotlib

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

import random

gen_random = random.Random(42)
N_CALIBRATION = 200
N_CORRECT_GENERATION = 250


logger = logzero.logger

# %%

try:
    import sklearn
except ImportError:
    raise ImportError("Install first `scikit-learn`.")
# end try

# %% Path to resource inputs

PATH_DIR_AVG_LOG_PROB = "/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache/FaiseqTranslationModelHandlerVer2WordEmbeddings/beam/1.0"
PATH_DIR_AVG_MONTE_CARLO_SIM = "/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/results/mc_sim/result-2025_07_22/mc_sim.jsonl"

path_dir_avg_log_prob = Path(PATH_DIR_AVG_LOG_PROB)
assert path_dir_avg_log_prob.exists()

path_dir_avg_monte_carlo_sim = Path(PATH_DIR_AVG_MONTE_CARLO_SIM)
assert path_dir_avg_monte_carlo_sim.exists()


PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()
path_dataset_cnn = PATH_DATASET_CNN


PATH_GENERATION_FILE_CACHE = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache")
assert PATH_GENERATION_FILE_CACHE.exists()


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


def collect_beam_search_cache_files(path_dir_cache: Path) -> ty.List[str]:
    """Collecting the document-id that the cache file of the beam-search generation is ready."""
    assert path_dir_cache.exists()

    path_dir_beam = path_dir_cache / "FaiseqTranslationModelHandlerVer2WordEmbeddings" /  'beam'
    assert path_dir_beam.exists()

    seq_pt_files = list(path_dir_beam.rglob(r"**/*pt"))
    seq_zlib_files = list(path_dir_beam.rglob(r"**/*zlib"))

    # file names -> document id
    seq_files = seq_pt_files + seq_zlib_files
    seq_files_no = [file_name.name.replace(".pt", '').replace(".pkl.zlib", '') for file_name in seq_files]
    seq_files_no = list(set(seq_files_no))

    return seq_files_no


def collect_stochastic_cache_files(path_dir_cache: Path) -> ty.List[str]:
    """collecting file numbers that stochastic samples are ready."""
    assert path_dir_cache.exists()

    path_dir_stochastic = path_dir_cache / "FaiseqTranslationModelHandlerVer2WordEmbeddings" /  'stochastic'
    assert path_dir_stochastic.exists()

    seq_pt_files = list(path_dir_stochastic.rglob(r"**/*pt"))
    seq_zlib_files = list(path_dir_stochastic.rglob(r"**/*zlib"))

    # file names -> document id
    seq_files = seq_pt_files + seq_zlib_files
    seq_files_no = [file_name.name.replace(".pt", '').replace(".pkl.zlib", '') for file_name in seq_files]
    seq_files_no = list(set(seq_files_no))

    return seq_files_no



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



def _make_ratio_true_positive_and_false_positive_curve(seq_evaluation_records_container: ty.List[EvaluationResultContainer]):
    """Generating the precision-recall curve for MMD Flagger Ver1.
    
    This method plots the precision-recall curve. 
    """

    # seq_precision = []
    # seq_recall = []
    _seq_value_plot = []
    for _eval_record in seq_evaluation_records_container:
        tp = _eval_record.true_positive
        fp = _eval_record.false_positive
        fn = _eval_record.false_negative
        tn = _eval_record.true_negative

        _tp_rate = (tp) / (tp + fn)
        _fp_rate = (fp) / (fp + tn)

        _seq_value_plot.append(dict(tp_rate=_tp_rate, fp_rate=_fp_rate))
    # end for        

    f, ax = plot.subplots()

    _df_rate_plot = pd.DataFrame(_seq_value_plot)

    sns.lineplot(data=_df_rate_plot, x="fp_rate", y="tp_rate", linewidth=3, ax=ax, legend=False)

    # adding the chance rate line.
    y = x = np.arange(0.0, 1.05, 0.05)
    sns.lineplot(x=x, y=y, dashes=True, linewidth=3, ax=ax, linestyle='--', legend=False)

    seq_tpr = [0.0] + _df_rate_plot.tp_rate
    seq_fpr = [0.0] + _df_rate_plot.fp_rate
    # Compute AUC using trapezoidal rule
    _roc_auc_score = np.trapz(seq_tpr, seq_fpr)            

    ax.set_title(f"ROC Curve (AUC = {round(_roc_auc_score, 4)})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


    return f, ax



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


def get_document_id_correct_outcome_geneations(dict_document_id2record: ty.Dict[UniqueDocumentId, ty.Dict]) -> ty.List[str]:
    """Selecting sub-dataset; the condition is the same as the MMD-Flagger execution."""

    # candidate of calibration-dataset; the voting are all 3.
    seq_correct_records = []
    _obj: ty.Dict
    for _unique_id, _obj in dict_document_id2record.items():
        sum_annotator_votes = sum(_obj["annotator_votes"])

        if sum_annotator_votes == 3:
            seq_correct_records.append(_unique_id.to_str())
    # end for

    seq_beam_cache_file_no = collect_beam_search_cache_files(PATH_GENERATION_FILE_CACHE)
    logger.info(f"Found beam-search cache files: {len(seq_beam_cache_file_no)}")


    # taking intersection of `seq_calibration_records` and `seq_beam_cache_file_no` -> calibration record id.
    set_ids_correct_records = set(seq_correct_records).intersection(set(seq_beam_cache_file_no))
    assert len(set_ids_correct_records) > 0
    logger.info(f"Correct Generation Population: {len(set_ids_correct_records)}")


    seq_document_ids_stochastic_ready = collect_stochastic_cache_files(PATH_GENERATION_FILE_CACHE)
    # (population-correct-summarization \cap population-stochastic-sampling-ready) -> population-for-flagging
    seq_document_id_correct_summarization_stochastic_ready = gen_random.sample(list(set(set_ids_correct_records) & set(seq_document_ids_stochastic_ready)), k=min([len(set(set_ids_correct_records) & set(seq_document_ids_stochastic_ready)), N_CORRECT_GENERATION]))

    return seq_document_id_correct_summarization_stochastic_ready


# %% main


def main_procedure_eval_log_probability():
    # %% main, setting the global variables.

    dict_document_id2record = load_dataset(path_dataset_cnn)

    # %% selecting the data subset
    seq_document_id_correct_summarization_stochastic_ready = get_document_id_correct_outcome_geneations(dict_document_id2record)


    # %%

    # TODO: I need to set the example-id selection. The MMD-Flagger used the subset.

    path_dir_output = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/results/log_prob")
    path_dir_output.mkdir(parents=True, exist_ok=True)

    path_dir_analysis_files = path_dir_output / 'analysis_files'
    path_dir_analysis_files.mkdir(parents=True, exist_ok=True)

    # main procedure for avg log prob

    # getting the ground-truth label
    dict_document_id2ground_truth_label = _generate_ground_truth_label(dict_document_id2record, label_mode="<3")
    seq_document_id_hallucination_label = [_document_id for _document_id, _label in dict_document_id2ground_truth_label.items() if _label == 1]
    seq_document_id_subset = seq_document_id_hallucination_label + seq_document_id_correct_summarization_stochastic_ready

    # loading the cache files
    dict_document_id2dict_obj_all = load_cache_files(path_dir_avg_log_prob)

    # taking the data subset
    dict_document_id2dict_obj = {_document_id:_record_obj for _document_id, _record_obj in dict_document_id2dict_obj_all.items() if _document_id in seq_document_id_subset}

    # making a dict {document-id: avg-log-score}
    dict_document_id2avg_log_prob: ty.Dict[str, float] = {_doc_id: _obj['log_probability_score'] for _doc_id, _obj in dict_document_id2dict_obj.items()}

    logger.info(f"Evaluation Target Dataset Records N = {len(dict_document_id2avg_log_prob)}, N(correct)={len(seq_document_id_correct_summarization_stochastic_ready)}, N(hallucination)={len(seq_document_id_hallucination_label)}")

    # -------------------------------------
    # evaluating with the fixed threshold.
    # setting the threshold.
    _threshold_fixed = np.percentile(np.array(list(dict_document_id2avg_log_prob.values())), q=threshold_avg_log_prob)

    # getting the prediction labels
    dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold_fixed else 0 for _key_obj, _score in dict_document_id2avg_log_prob.items()}

    # -------------------------------------
    # saving the eval score at the fixed threshold
    _path_json_fixed_threshold = path_dir_analysis_files / 'eval_fixed_threshold.json'

    eval_container_fixed_threshold = get_evaluation_arrays(dict_document_idprediction_label, dict_document_id2ground_truth_label)

    with _path_json_fixed_threshold.open('w') as f:
        obj_dump = eval_container_fixed_threshold._asdict()
        obj_dump['threshold_percentile'] = threshold_avg_log_prob
        f.write(json.dumps(obj_dump, indent=4))
    # end with

    # visualization and file outcome

    # visualizing the histogram
    _f, _ax = plot.subplots()
    sns.histplot(np.array(list(dict_document_id2avg_log_prob.values())), ax=_ax)
    _ax.axvline(x=_threshold_fixed.item(), color='red', linestyle='-')
    _path_png_score_histogram = path_dir_analysis_files / 'score_histogram.png'
    _f.savefig(_path_png_score_histogram.as_posix(), dpi=300, bbox_inches='tight')

    # -------------------------------------------------
    # ROC curve and AUC score
    n_plot_eval = 1000

    _score_min = min(list(dict_document_id2avg_log_prob.values()))
    _score_max = max(list(dict_document_id2avg_log_prob.values()))

    # I create a sequence of threshold values.
    _seq_threshold = np.linspace(_score_min, _score_max, num=n_plot_eval)    
    seq_eval_container: ty.List[EvaluationResultContainer] = []
    seq_threshold = []
    for _threshold in _seq_threshold:
        # _threshold = np.percentile(np.array(list(dict_document_id2avg_log_prob.values())), q=_percentile)
        _dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold else 0 for _key_obj, _score in dict_document_id2avg_log_prob.items()}
        _eval_container = get_evaluation_arrays(_dict_document_idprediction_label, dict_document_id2ground_truth_label)
        
        if _eval_container.precision == 0.0:
            continue
        else:
            seq_eval_container.append(_eval_container)
            seq_threshold.append(_threshold)
    # end for
    _f_precision_recall, _ax_precision_recall = _make_ratio_true_positive_and_false_positive_curve(seq_eval_container)

    _path_precision_recall_curve = path_dir_analysis_files / 'true_positive_and_false_positive_curve.png'
    _f_precision_recall.savefig(_path_precision_recall_curve.as_posix(), dpi=300, bbox_inches='tight')

    ## saving the precision-recall curve
    _seq_record_eval_container = [_eval_container._asdict() for _eval_container in seq_eval_container]
    df_precision_recall_curve = pd.DataFrame(_seq_record_eval_container)
    df_precision_recall_curve['threshold'] = pd.Series(seq_threshold)
    _path_tsv_precision_recall_curve = path_dir_analysis_files / 'true_positive_and_false_positive_curve.tsv'
    df_precision_recall_curve.to_csv(_path_tsv_precision_recall_curve, sep='\t', index=False)

    # -------------------------------------------------
    # saving the document-id -> avg log probability score.
    _path_json_document_id2avg_log_prob = path_dir_analysis_files / 'document_id2avg_log.json'
    with _path_json_document_id2avg_log_prob.open('w') as f:
        f.write(json.dumps(dict_document_id2avg_log_prob, indent=4))
    # end with


def main_procedure_eval_monte_carlo_similarity():

    dict_document_id2avg_mc_sim: ty.Dict[str, float] = {}
    assert path_dir_avg_monte_carlo_sim.exists()
    with path_dir_avg_monte_carlo_sim.open() as f:
        __line_parsed = [json.loads(_line) for _line in f.readlines()]
        dict_document_id2avg_mc_sim = {_obj['doc_id']: _obj['avg_score'] for _obj in __line_parsed}
    # end with

    dict_document_id2record = load_dataset(path_dataset_cnn)

    # %% selecting the data subset
    seq_document_id_correct_summarization_stochastic_ready = get_document_id_correct_outcome_geneations(dict_document_id2record)

    path_dir_output = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/results/mc_sim")
    path_dir_output.mkdir(parents=True, exist_ok=True)

    path_dir_analysis_files = path_dir_output / 'analysis_files'
    path_dir_analysis_files.mkdir(parents=True, exist_ok=True)

    # getting the ground-truth label
    dict_document_id2ground_truth_label = _generate_ground_truth_label(dict_document_id2record, label_mode="<3")
    seq_document_id_hallucination_label = [_document_id for _document_id, _label in dict_document_id2ground_truth_label.items() if _label == 1]
    seq_document_id_subset = seq_document_id_hallucination_label + seq_document_id_correct_summarization_stochastic_ready

    # # loading the cache files
    # dict_document_id2dict_obj_all = load_cache_files(path_dir_avg_log_prob)

    # taking the data subset: {document-id: MC-SIM}
    dict_document_id2avg_mc_sim = {_document_id:_score_sim for _document_id, _score_sim in dict_document_id2avg_mc_sim.items() if _document_id in seq_document_id_subset}

    logger.info(f"Evaluation Target Dataset Records N = {len(dict_document_id2avg_mc_sim)}, N(correct)={len(seq_document_id_correct_summarization_stochastic_ready)}, N(hallucination)={len(seq_document_id_hallucination_label)}")

    # -------------------------------------
    # TODO: eval: fixed threshold.
    _threshold_fixed = np.percentile(np.array(list(dict_document_id2avg_mc_sim.values())), q=threshold_avg_log_prob)

    # getting the prediction labels
    dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold_fixed else 0 for _key_obj, _score in dict_document_id2avg_mc_sim.items()}

    # -------------------------------------
    # saving the eval score at the fixed threshold
    _path_json_fixed_threshold = path_dir_analysis_files / 'eval_fixed_threshold.json'

    eval_container_fixed_threshold = get_evaluation_arrays(dict_document_idprediction_label, dict_document_id2ground_truth_label)

    with _path_json_fixed_threshold.open('w') as f:
        obj_dump = eval_container_fixed_threshold._asdict()
        obj_dump['threshold_percentile'] = threshold_avg_log_prob
        f.write(json.dumps(obj_dump, indent=4))
    # end with

    # -------------------------------------

    # visualizing the histogram
    _f, _ax = plot.subplots()
    sns.histplot(np.array(list(dict_document_id2avg_mc_sim.values())), ax=_ax)
    _ax.axvline(x=_threshold_fixed.item(), color='red', linestyle='-')
    _ax.set_xlim(0.0, 1.0)
    _path_png_score_histogram = path_dir_analysis_files / 'score_histogram.png'
    _f.savefig(_path_png_score_histogram.as_posix(), dpi=300, bbox_inches='tight')

    # -------------------------------------------------
    # ROC curve and AUC score
    n_plot_eval = 1000

    _score_min = min(list(dict_document_id2avg_mc_sim.values()))
    _score_max = max(list(dict_document_id2avg_mc_sim.values()))

    # I create a sequence of threshold values.
    _seq_threshold = np.linspace(_score_min, _score_max, num=n_plot_eval)    
    seq_eval_container: ty.List[EvaluationResultContainer] = []
    seq_threshold = []
    for _threshold in _seq_threshold:
        _dict_document_idprediction_label = {_key_obj: 1 if _score < _threshold else 0 for _key_obj, _score in dict_document_id2avg_mc_sim.items()}
        _eval_container = get_evaluation_arrays(_dict_document_idprediction_label, dict_document_id2ground_truth_label)
        
        if _eval_container.precision == 0.0:
            continue
        else:
            seq_eval_container.append(_eval_container)
            seq_threshold.append(_threshold)
    # end for
    _f_precision_recall, _ax_precision_recall = _make_ratio_true_positive_and_false_positive_curve(seq_eval_container)

    _path_precision_recall_curve = path_dir_analysis_files / 'true_positive_and_false_positive_curve.png'
    _f_precision_recall.savefig(_path_precision_recall_curve.as_posix(), dpi=300, bbox_inches='tight')

    ## saving the precision-recall curve
    _seq_record_eval_container = [_eval_container._asdict() for _eval_container in seq_eval_container]
    df_precision_recall_curve = pd.DataFrame(_seq_record_eval_container)
    df_precision_recall_curve['threshold'] = pd.Series(seq_threshold)
    _path_tsv_precision_recall_curve = path_dir_analysis_files / 'true_positive_and_false_positive_curve.tsv'
    df_precision_recall_curve.to_csv(_path_tsv_precision_recall_curve, sep='\t', index=False)

    # -------------------------------------------------
    # saving the document-id -> avg log probability score.
    _path_json_document_id2avg_log_prob = path_dir_analysis_files / 'document_id2avg_log.json'
    with _path_json_document_id2avg_log_prob.open('w') as f:
        f.write(json.dumps(dict_document_id2avg_mc_sim, indent=4))
    # end with




if __name__ == '__main__':
    main_procedure_eval_log_probability()
    main_procedure_eval_monte_carlo_similarity()
