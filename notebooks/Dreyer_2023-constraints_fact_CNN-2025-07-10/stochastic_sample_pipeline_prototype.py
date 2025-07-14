# %%
from pathlib import Path
import typing as ty
import random

from summary_abstractive.module_model_handler.ver2 import (
    FaiseqTranslationModelHandlerVer2WordEmbeddings, 
    TranslationResultContainer,
    EvaluationTargetTranslationPair)
from summary_abstractive.exceptions import ParameterSettingException
import json
import numpy as np


# %%
PATH_MODEL_BART_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/models/bart.large.cnn")
assert PATH_MODEL_BART_CNN.exists()

# %%
PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()

# %%
PATH_CACHE_DIR_BASE = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache")
assert PATH_CACHE_DIR_BASE.exists()

# %%
summary_model_handler = FaiseqTranslationModelHandlerVer2WordEmbeddings(
    path_cache_dir=PATH_CACHE_DIR_BASE,
    path_dir_fairseq_model=PATH_MODEL_BART_CNN
)

# %%
with PATH_DATASET_CNN.open('r') as f:
    seq_dataset_obj = [json.loads(_line) for _line in f.readlines()]
# end with

# %%
import logging

from summary_abstractive import logger_module
from datetime import datetime

path_log_dir = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-2025-07-10/generations") / f'{datetime.now().isoformat()}.log'

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(path_log_dir)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_module.formatter)

std_handler = logging.StreamHandler()
std_handler.setLevel(logging.DEBUG)
std_handler.setFormatter(logger_module.formatter)

logger.addHandler(file_handler)
logger.addHandler(std_handler)

# re-setting the log level.
logging.getLogger('fairseq').setLevel(logging.WARNING)

# %%
# -------------------------------------------
# getting git commit id
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
logger.info(f'Current Git Commit: {sha}')

# %%
n_sampling = 25
tau_parameters = [float(f'{_tau:.1f}') for _tau in np.arange(0.1, 1.0, 0.1)]


# document-id definition: "dataset-type"-"document-id"-"abstract-command"
# for example: cnn_dailymail-0-none

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

# %%
# allocating the unique id to all records
dict_unique_id2records = {
    UniqueDocumentId(datasource=_obj['dataset_name'], document_id_original=_obj['document_id'], abstractiveness_constraint=_obj['abstractiveness_constraint']): _obj 
    for _obj in seq_dataset_obj
}
logger.info(f'Count Unique document -> {len(dict_unique_id2records)}')


# %%
# inspecting the annotation label.

# hallucination labels
seq_document_unique_id_hallucination = [_unique_id for _unique_id, _obj in dict_unique_id2records.items() if sum(_obj["annotator_votes"]) == 0]
logger.info(f"Hallucination record -> {len(seq_document_unique_id_hallucination)}")

# calibration records
seq_document_unique_id_calibration_whole = [_unique_id for _unique_id, _obj in dict_unique_id2records.items() if sum(_obj["annotator_votes"]) == 3]
logger.info(f"Calibration record -> {len(seq_document_unique_id_calibration_whole)}")


# %% downsampling the calibration record. It's too much

N_CALIBRATION_RECORD = 200
RANDOM_SEED = 42

random_gen = random.Random(RANDOM_SEED)
seq_document_unique_id_calibration = random_gen.sample(seq_document_unique_id_calibration_whole, k=N_CALIBRATION_RECORD)
logger.info(f"Downsampling the calibration records. {len(seq_document_unique_id_calibration)}")

# %%

# shuffling the keys
import random
gen_random = random.Random()
seq_key_shuffle = list(dict_unique_id2records.keys())
gen_random.shuffle(seq_key_shuffle)


for _unique_id in seq_key_shuffle:
    if (_unique_id not in seq_document_unique_id_calibration) and (_unique_id not in seq_document_unique_id_hallucination):
        continue
    # end if

    _obj = dict_unique_id2records[_unique_id]

    _document_unique_id: str = _unique_id.to_str()

    _document_full: str = _obj['document_full']
    _document_original: str = _obj['document_original']
    
    _penalty_command: str = _obj['abstractiveness_constraint']
    
    assert _document_full == _document_original

    _input_record = EvaluationTargetTranslationPair(sentence_id=_document_unique_id, source=_document_full, target="")

    logger.info('=' * 30)
    logger.info(f"document-id = {_document_unique_id}")
    for _tau in tau_parameters:
        try:
            summary_model_handler.translate_sample_multiple_times(
                input_text=_input_record,
                n_sampling=n_sampling,
                temperature=_tau,
                penalty_command=_penalty_command
            )
            logger.info(f"done tau={_tau}")
        except ParameterSettingException:
            logger.error(f"Error at tau={_tau} and `ParameterSettingException`")

