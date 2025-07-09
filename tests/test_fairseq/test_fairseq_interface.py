import pytest
import time
from pathlib import Path
import logzero
import os
import typing as ty

import torch
import fairseq
from fairseq.models.bart import BARTHubInterface
from fairseq.models.bart import BARTModel

from summary_abstractive.utils import get_extractive_penalty_fct


logger = logzero.logger

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



@pytest.fixture(scope="module")
def long_running_setup(resource_path_root: Path):
    """This fixture will run only once for all tests in this module."""
    logger.info("\n--- ⏳ STARTING long-running procedure (once) ---")

    path_dir_bart_model_cnn = resource_path_root / 'models/bart.large.cnn'
    assert path_dir_bart_model_cnn.exists()
    
    bart_model = load_model(path_dir_bart_model_cnn, path_dir_bart_model_cnn / 'model.pt')

    if torch.cuda.is_available():
        device_obj = torch.device('cuda:0')
    else:
        device_obj = torch.device('cpu')
    # end if

    bart_model = bart_model.to(device_obj)


    result = {"bart_model": bart_model, "device": device_obj}
    logger.info("--- ✅ Long-running procedure FINISHED ---")

    yield result # Your tests will receive this 'result' object

    # Teardown code runs after all tests in the module are done
    logger.info("\n--- 🧹 Tearing down the resource (once) ---")
    del result


def test_stochastic_sampling_sample_same(long_running_setup: ty.Dict):
    bart_model = long_running_setup["bart_model"]
    device_obj = long_running_setup["device"]

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


def test_stochastic_sampling_sample_variant(long_running_setup: ty.Dict):
    bart_model = long_running_setup["bart_model"]
    device_obj = long_running_setup["device"]

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


def test_compatible_methods_generate_and_sample(long_running_setup: ty.Dict):
    # using the generate method
    # test compatibale output, `generate` method == `sample` method.
    bart_model = long_running_setup["bart_model"]
    device_obj = long_running_setup["device"]


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

    # by interface of `genarate`
    tensor_input_ids = bart_model.encode(document_input)
    tensor_stack = torch.stack([tensor_input_ids]).to(bart_model.device)
    generated_ids = bart_model.generate(tensor_stack, **dict_parameters)
    text_summary_generate_method: str = bart_model.decode(generated_ids[0]['tokens'])

    # by interface of `sample`
    text_summary_generate_sample: ty.List[str] = bart_model.sample([document_input], **dict_parameters)

    assert text_summary_generate_method == text_summary_generate_sample[0]