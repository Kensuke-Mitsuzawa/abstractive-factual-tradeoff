import torch
import json
import typing as ty

from pathlib import Path

from summary_abstractive.module_model_handler.ver2 import (
    TranslationResultContainer,
    EvaluationTargetTranslationPair,
    FaiseqTranslationModelHandlerVer2WordEmbeddings)


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
# -------------------------------------------


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


# 1. Load the pre-trained BART model from PyTorch Hub
# Using force_reload=True to ensure you have the latest version
print("Loading BART model...")
bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn', force_reload=False)
bart.eval()  # Set the model to evaluation mode

# Use GPU if available
if torch.cuda.is_available():
    bart.cuda()
print("Model loaded successfully.")

# 2. Define your source and target sentences
source_sentence = "Scientists have been studying the effects of climate change on polar bears."
# A plausible target sentence to score
target_sentence = "Researchers are examining how climate change impacts polar bears."

# 3. Tokenize sentences using the model's encoder
# The .encode() method handles adding special tokens (BOS/EOS)
# .unsqueeze(0) adds the batch dimension
source_tokens = bart.encode(source_sentence).unsqueeze(0)
target_tokens = bart.encode(target_sentence).unsqueeze(0)

# Move tensors to the same device as the model
if bart.device.type == 'cuda':
    source_tokens = source_tokens.cuda()
    target_tokens = target_tokens.cuda()

# 4. Score the target sequence using generate()
# We pass the tokenized target to the `target_tokens` argument.
# This instructs the generator to score this sequence instead of decoding a new one.
with torch.no_grad():
    hypotheses = bart.generate(
        source_tokens,
        beam=1,  # Use beam=1 for efficiency as we are scoring, not searching
        target_tokens=target_tokens
    )

# 5. Extract and interpret the scores
# The result is a list (for the batch) of lists (for beam hypotheses).
# hypotheses[0][0] contains the scored reference sequence.
scored_result = hypotheses[0][0]

# The 'score' is the cumulative log probability of the entire sequence.
# It is the sum of the 'positional_scores'.
total_log_prob = scored_result['score']

# 'positional_scores' contains the log probability of each token.
# P(token_i | source, token_0, ..., token_{i-1})
per_token_log_probs = scored_result['positional_scores']

# 'tokens' are the token IDs of the scored sequence.
output_tokens = scored_result['tokens']


# --- Display the results ---
print("\n" + "="*50)
print(f"Source: '{source_sentence}'")
print(f"Target: '{target_sentence}'")
print(f"\nTotal Sequence Log Probability: {total_log_prob.item():.4f}")
print("This score is calculated as the sum of the individual token log probabilities:")
print(f"$$\\log P(Y|X) = \\sum_{i=1}^{N} \\log P(y_i | y_{<i}, X)$$")

print("\n--- Per-Token Scores ---")
print(f"{'Token':<20} | {'Log-Prob Score'}")
print("-" * 40)

# Decode each token and print it with its score
for i, token_id in enumerate(output_tokens):
    token_str = bart.decode(torch.tensor([token_id]))
    score = per_token_log_probs[i].item()
    print(f"{token_str.strip():<20} | {score:.4f}")

# Verify that the sum of positional scores equals the total score
recalculated_score = torch.sum(per_token_log_probs).item()
print("-" * 40)
print(f"{'Sum of positional scores:':<20} | {recalculated_score:.4f}")
print("="*50)