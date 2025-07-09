# %% [markdown]
# # Reproducing codebase as Python API
# 
# The notebook reproduces the trained-BART model as the Python API not as the CLI. 

# %%
import os
import sys
import argparse
import logging
import re
import typing as ty

from tqdm import tqdm
from warnings import warn
from torch.multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)
from functools import partial
import more_itertools as mit

import torch
import fairseq
from fairseq.models.bart import BARTHubInterface
from fairseq.models.bart import BARTModel

import nvgpu

from pathlib import Path

# %%
import logzero

from datetime import datetime
_datetime_exec = datetime.now()

path_log = Path('logs')
path_log.mkdir(parents=True, exist_ok=True)

logzero.logfile(f"logs/{_datetime_exec.isoformat()}.log")

logger = logzero.logger

# %%
def load_model(task: Path, model_path: Path) -> BARTHubInterface:
    """
    Args:
        task: a path to the directory of the model.
        model_path: a path to 'model.pt' file.
    """
    assert task.exists()
    assert model_path.exists()

    logger.info(f"Loading model {model_path}")
    model_dirname, model_fname = os.path.split(model_path.as_posix())
    bart = BARTModel.from_pretrained(
        model_dirname,
        checkpoint_file=model_fname,
        data_name_or_path=task.as_posix()
    )
    logger.info(f"Loading done.")
    return bart

PATH_MODEL_FILE = Path('/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/models/bart.large.cnn')

bart_model = load_model(PATH_MODEL_FILE, PATH_MODEL_FILE / 'model.pt')

# %%
type(bart_model)

# %%
logger.info(str(bart_model))

# %%
if torch.cuda.is_available():
    device_obj = torch.device('cuda:0')
else:
    device_obj = torch.device('cpu')
# end if

bart_model = bart_model.to(device_obj)

# %%
from torch.nn.modules.sparse import Embedding

# |V| -> 1024, where 1024 is the word embedding size
layer_word_embedding: Embedding = bart_model.model.decoder.embed_tokens

# test obtaining word embedding
tensor_word_embed: torch.Tensor = layer_word_embedding(torch.tensor([0, 10]).to(bart_model.device))
assert isinstance(tensor_word_embed, torch.Tensor)


# %%

def get_word_embedding(bart: BARTHubInterface, tensor_token_ids: torch.Tensor) -> torch.Tensor:
    pass


# %%
def bart_sample_stochastic(bart: BARTHubInterface,
                           batch: ty.List[str],
                           temperature: float,
                           extractive_penalty_fct: str,
                           random_seed_value: int,                           
                           beam: int = 4,
                           lenpen: float = 2.0,  # length penalty
                           min_len: int = 55,
                           max_len_a: int = 0,
                           max_len_b: int = 140,
                           no_repeat_ngram_size: int = 3):
    dict_parameters = dict(
        beam=beam,
        lenpen=lenpen,
        sampling=True,
        temperature=temperature,
        min_len=min_len, 
        max_len_a=max_len_a, 
        max_len_b=max_len_b,
        no_repeat_ngram_size=no_repeat_ngram_size,
        extractive_penalty_fct=extractive_penalty_fct)

    with (torch.random.fork_rng(), torch.no_grad()):
        torch.manual_seed(random_seed_value)
        torch.cuda.manual_seed_all(random_seed_value)  # if you are using multi-GPU.        
        res = bart.sample(batch, **dict_parameters)

    return res
# end def



# %%
# case CNN constraints dataset
import json

PATH_CONSTRAINS_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_CONSTRAINS_CNN.exists()

with PATH_CONSTRAINS_CNN.open() as f:
    seq_dataset = [json.loads(_line) for _line in f.readlines()]
# end with

logger.info(f'{len(seq_dataset)} records')

# double check: all xsum
for _record in seq_dataset:
    assert _record['dataset_name'] == 'cnn_dailymail'
# end for

# %%
def get_source_and_summary(record_obj: ty.Dict) -> ty.Tuple[str, str]:
    # return record_obj['document_original'], record_obj['summary_raw']
    return record_obj['document_full'], record_obj['summary_raw']
# end def

target_document_index = [1, 10, 100, 200]

# %%
def get_extractive_penalty_fct(penalty_command: str) -> str:
    dict_commnad2ep = dict(
        lambda4 = 'log_exp(2,4.804488)',  # lambda4
        lambda2 = 'log_exp(2,2.402244)',  # lambda2
        lambda1 = 'log_exp(2,1.201122)',  # lambda1
        none = 'none()',
        linear = 'linear()',
    )
    dict_commnad2ep['1/lambda2'] = 'log_exp(2,0.416277447)'  # 1/lambda2, log_exp(2, 1 / (1.20112 * 2))
    dict_commnad2ep['1/lambda1'] = 'log_exp(2,0.832556281)'  # 1/lambda1, log_exp(2, 1 / 1.20112)

    assert penalty_command in dict_commnad2ep

    return dict_commnad2ep[penalty_command]


def init_model(testresource: Path) -> BARTHubInterface:
    path_model = testresource / 'models/bart.large.cnn'
    assert path_model.exists(), f'no model found at {path_model}'

    bart_model = load_model(path_model, path_model / 'model.pt')

    return bart_model

def test_stochastic_sampling_sample_same(testresource: Path):
    bart_model = init_model(testresource)

    if torch.cuda.is_available():
        device_obj = torch.device('cuda:0')
    else:
        device_obj = torch.device('cpu')
    # end if

    bart_model = bart_model.to(device_obj)    

    document_input = 'He\'s a blue chip college basketball recruit. She\'s a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn\'t be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School\'s basketball team in Louisville, Kentucky, who\'s headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern\'s prom. So why is he taking Ellie instead? "She\'s great... she listens and she\'s easy to talk to" he said. Trey made the prom-posal (yes, that\'s what they are calling invites to prom these days) in the gym during Ellie\'s P.E. class. Trina Helson, a teacher at Eastern, alerted the school\'s newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn\'t surpristed by Trey\'s actions. "That\'s the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let\'s Party Like it\'s 1989," a reference to the latest album by Taylor Swift, Ellie\'s favorite singer. Trey also got the OK from Ellie\'s parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie\'s mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey\'s future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he\'s known for a long time, often works with other kids . Trey\'s mother, Shelly Moses, was also proud of her son. "It\'s exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he\'s a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can\'t stop thinking about prom. "Ellie can\'t wait to go dress shopping" her mother said. "Because I\'ve only told about a million people!" Ellie interjected.'
    extractive_penalty_fct = get_extractive_penalty_fct('none')

    random_seed = 42
    seq_summary_stochastic_1st = bart_sample_stochastic(
        bart=bart_model,
        temperature=1.0,
        batch=[document_input],
        random_seed_value=random_seed,
        extractive_penalty_fct=extractive_penalty_fct)
    
    seq_summary_stochastic_2nd = bart_sample_stochastic(
        bart=bart_model,
        temperature=1.0,
        batch=[document_input],
        random_seed_value=random_seed,
        extractive_penalty_fct=extractive_penalty_fct)
    
    assert seq_summary_stochastic_1st[0] == seq_summary_stochastic_2nd[0]


def test_stochastic_sampling_sample_variant(testresource: Path):
    bart_model = init_model(testresource)

    if torch.cuda.is_available():
        device_obj = torch.device('cuda:0')
    else:
        device_obj = torch.device('cpu')
    # end if

    bart_model = bart_model.to(device_obj)    

    document_input = 'He\'s a blue chip college basketball recruit. She\'s a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn\'t be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School\'s basketball team in Louisville, Kentucky, who\'s headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern\'s prom. So why is he taking Ellie instead? "She\'s great... she listens and she\'s easy to talk to" he said. Trey made the prom-posal (yes, that\'s what they are calling invites to prom these days) in the gym during Ellie\'s P.E. class. Trina Helson, a teacher at Eastern, alerted the school\'s newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn\'t surpristed by Trey\'s actions. "That\'s the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let\'s Party Like it\'s 1989," a reference to the latest album by Taylor Swift, Ellie\'s favorite singer. Trey also got the OK from Ellie\'s parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie\'s mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey\'s future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he\'s known for a long time, often works with other kids . Trey\'s mother, Shelly Moses, was also proud of her son. "It\'s exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he\'s a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can\'t stop thinking about prom. "Ellie can\'t wait to go dress shopping" her mother said. "Because I\'ve only told about a million people!" Ellie interjected.'
    extractive_penalty_fct = get_extractive_penalty_fct('none')

    random_seed = 42
    seq_summary_stochastic_1st = bart_sample_stochastic(
        bart=bart_model,
        temperature=1.0,
        batch=[document_input],
        random_seed_value=random_seed,
        extractive_penalty_fct=extractive_penalty_fct)
    
    random_seed = 1
    seq_summary_stochastic_2nd = bart_sample_stochastic(
        bart=bart_model,
        temperature=1.0,
        batch=[document_input],
        random_seed_value=random_seed,
        extractive_penalty_fct=extractive_penalty_fct)
    
    assert seq_summary_stochastic_1st[0] != seq_summary_stochastic_2nd[0]

# --------------------------

path_resources = Path('/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources')

# test_stochastic_sampling_sample_same(path_resources)
# test_stochastic_sampling_sample_variant(path_resources)


# %%

def test_compatible_methods_generate_and_sample(testresources: Path):
    # using the generate method
    # test compatibale output, `generate` method == `sample` method.
    bart_model = init_model(testresources)

    if torch.cuda.is_available():
        device_obj = torch.device('cuda:0')
    else:
        device_obj = torch.device('cpu')
    # end if

    bart_model = bart_model.to(device_obj)    



    document_input = 'He\'s a blue chip college basketball recruit. She\'s a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn\'t be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School\'s basketball team in Louisville, Kentucky, who\'s headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern\'s prom. So why is he taking Ellie instead? "She\'s great... she listens and she\'s easy to talk to" he said. Trey made the prom-posal (yes, that\'s what they are calling invites to prom these days) in the gym during Ellie\'s P.E. class. Trina Helson, a teacher at Eastern, alerted the school\'s newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn\'t surpristed by Trey\'s actions. "That\'s the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let\'s Party Like it\'s 1989," a reference to the latest album by Taylor Swift, Ellie\'s favorite singer. Trey also got the OK from Ellie\'s parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie\'s mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey\'s future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he\'s known for a long time, often works with other kids . Trey\'s mother, Shelly Moses, was also proud of her son. "It\'s exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he\'s a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can\'t stop thinking about prom. "Ellie can\'t wait to go dress shopping" her mother said. "Because I\'ve only told about a million people!" Ellie interjected.'

    beam: int = 4
    lenpen: float = 2.0  # length penalty
    min_len: int = 55
    max_len_a: int = 0
    max_len_b: int = 140
    no_repeat_ngram_size: int = 3
    extractive_penalty_fct = get_extractive_penalty_fct('none')

    dict_parameters = dict(
        beam=beam,
        lenpen=lenpen,
        sampling=False,
        min_len=min_len, 
        max_len_a=max_len_a, 
        max_len_b=max_len_b,
        no_repeat_ngram_size=no_repeat_ngram_size,
        extractive_penalty_fct=extractive_penalty_fct)

    tensor_input_ids = bart_model.encode(document_input)
    tensor_stack = torch.stack([tensor_input_ids]).to(bart_model.device)
    generated_ids = bart_model.generate(tensor_stack, **dict_parameters)
    text_summary_generate_method: str = bart_model.decode(generated_ids[0]['tokens'])

    text_summary_generate_sample: ty.List[str] = bart_model.sample([document_input], **dict_parameters)

    assert text_summary_generate_method == text_summary_generate_sample[0]

    # %%
