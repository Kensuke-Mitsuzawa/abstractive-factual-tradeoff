# %%
from pathlib import Path
import typing as ty
import random

from summary_abstractive.module_model_handler.ver2 import module_statics
from summary_abstractive.module_model_handler.ver2 import utils

from summary_abstractive.module_model_handler.ver2 import (
    TranslationResultContainer,
    EvaluationTargetTranslationPair,
    FaiseqTranslationModelHandlerVer2WordEmbeddings)
from summary_abstractive.module_model_handler.ver2.module_obtain_word_embedding import obtain_word_embedding

import json
import numpy as np


# %%
PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()

# %%
PATH_CACHE_DIR_BASE = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache")
assert PATH_CACHE_DIR_BASE.exists()


# %%
with PATH_DATASET_CNN.open('r') as f:
    seq_dataset_obj = [json.loads(_line) for _line in f.readlines()]
# end with


# %% loading the handler. note: I do not use the handler. But, the handler easily loads the model file.

PATH_MODEL_BART_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/models/bart.large.cnn")
assert PATH_MODEL_BART_CNN.exists()

summary_model_handler = FaiseqTranslationModelHandlerVer2WordEmbeddings(
    path_cache_dir=PATH_CACHE_DIR_BASE,
    path_dir_fairseq_model=PATH_MODEL_BART_CNN
)


# %%
import logging

from summary_abstractive import logger_module
from datetime import datetime

path_log_dir = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-2025-07-10/generations") / f'{__file__}_{datetime.now().isoformat()}.log'

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


for _unique_id, _obj in dict_unique_id2records.items():
    if (_unique_id not in seq_document_unique_id_calibration) and (_unique_id not in seq_document_unique_id_hallucination):
        continue
    # end if

    _document_unique_id: str = _unique_id.to_str()

    _document_full: str = _obj['document_full']
    _document_original: str = _obj['document_original']
    _summary_raw: str = _obj['summary_raw']
    
    _penalty_command: str = _obj['abstractiveness_constraint']
    
    assert _document_full == _document_original

    # _input_record = EvaluationTargetTranslationPair(sentence_id=_document_unique_id, source=_document_full, target="")

    logger.info('=' * 30)
    logger.info(f"document-id = {_document_unique_id}")

    # tokenizing
    _tensor_token_ids_document_source = summary_model_handler.bart_model.encode(_document_full)
    _tensor_token_ids_summary = summary_model_handler.bart_model.encode(_summary_raw)

    _tensor_beam_word_embedding = obtain_word_embedding(summary_model_handler.bart_model, _tensor_token_ids_summary)

    extractive_penalty_fct: str = utils.get_extractive_penalty_fct(_unique_id.abstractiveness_constraint)
    
    argument_translation_conditions= dict(
        beam=module_statics.BEAM,
        min_len = module_statics.MIN_LEN,
        max_len_a = module_statics.MAX_LEN_A,
        max_len_b = module_statics.MAX_LEN_B,
        length_penalty = module_statics.LENPEN,
        no_repeat_ngram_size = module_statics.NO_REPEAT_NGRAM_SIZE,
        extractive_penalty_fct=extractive_penalty_fct
    )
    _generation_obj_container = TranslationResultContainer(
        source_text=_document_full,
        translation_text=_summary_raw,
        source_language='source',
        target_language='target',
        source_tensor_tokens=_tensor_token_ids_document_source.cpu(),
        target_tensor_tokens=_tensor_token_ids_summary.cpu(),
        log_probability_score=None,
        dict_layer_embeddings={summary_model_handler._get_decoder_word_embedding_layer_name(): _tensor_beam_word_embedding.cpu()},
        argument_translation_conditions=argument_translation_conditions
    )

    summary_model_handler._save_cache(
        sentence_id=_document_unique_id, 
        tau_param=1.0,
        translation_obj=_generation_obj_container,
        n_sampling=None)
