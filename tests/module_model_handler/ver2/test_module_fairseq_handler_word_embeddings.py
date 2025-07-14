from pathlib import Path
import tempfile
import shutil
import numpy as np
import logzero
import pytest
import torch
import typing as ty

from summary_abstractive.module_model_handler.ver2.module_fairseq_handler_word_embeddings import (
    FaiseqTranslationModelHandlerVer2WordEmbeddings,
    EvaluationTargetTranslationPair)


logger = logzero.logger


def init_handler(resource_path_root: Path) -> FaiseqTranslationModelHandlerVer2WordEmbeddings:
    path_dir_bart_model_cnn = resource_path_root / 'bart.large.cnn'
    assert path_dir_bart_model_cnn.exists()

    path_temp_dir = Path(tempfile.mkdtemp())
    path_temp_dir.mkdir(parents=True, exist_ok=True)

    handler = FaiseqTranslationModelHandlerVer2WordEmbeddings(
        path_cache_dir=path_temp_dir,
        path_dir_fairseq_model=path_dir_bart_model_cnn,
    )
    return handler



def long_running_setup(resource_path_root: Path) -> ty.Dict:
    """This fixture will run only once for all tests in this module."""
    logger.info("\n--- ⏳ STARTING long-running procedure (once) ---")

    path_dir_bart_model_cnn = resource_path_root / 'models'
    assert path_dir_bart_model_cnn.exists()
    
    model_handler = init_handler(path_dir_bart_model_cnn)

    if torch.cuda.is_available():
        device_obj = torch.device('cuda:0')
    else:
        device_obj = torch.device('cpu')
    # end if

    result = {"handler": model_handler, "device": device_obj}
    logger.info("--- ✅ Long-running procedure FINISHED ---")

    # yield result # Your tests will receive this 'result' object

    # Teardown code runs after all tests in the module are done
    # logger.info("\n--- 🧹 Tearing down the resource (once) ---")
    # del result

    return result


def test_FaiseqTranslationModelHandlerVer2WordEmbeddings_pipeline(resource_path_root: Path):

    document_input = EvaluationTargetTranslationPair(
        source='He\'s a blue chip college basketball recruit. She\'s a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn\'t be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School\'s basketball team in Louisville, Kentucky, who\'s headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern\'s prom. So why is he taking Ellie instead? "She\'s great... she listens and she\'s easy to talk to" he said. Trey made the prom-posal (yes, that\'s what they are calling invites to prom these days) in the gym during Ellie\'s P.E. class. Trina Helson, a teacher at Eastern, alerted the school\'s newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn\'t surpristed by Trey\'s actions. "That\'s the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let\'s Party Like it\'s 1989," a reference to the latest album by Taylor Swift, Ellie\'s favorite singer. Trey also got the OK from Ellie\'s parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie\'s mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey\'s future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he\'s known for a long time, often works with other kids . Trey\'s mother, Shelly Moses, was also proud of her son. "It\'s exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he\'s a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can\'t stop thinking about prom. "Ellie can\'t wait to go dress shopping" her mother said. "Because I\'ve only told about a million people!" Ellie interjected.',
        target='',
        sentence_id='test'
    )
    penalty_command = 'none'

    setup_obj = long_running_setup(resource_path_root)
    handler: FaiseqTranslationModelHandlerVer2WordEmbeddings = setup_obj['handler']
    seq_res = handler.translate_sample_multiple_times(
        input_text=document_input,
        n_sampling=5,
        temperature=0.5,
        penalty_command=penalty_command
    )
    assert len(seq_res) == 5

    max_len_tensor = max([len(_obj.target_tensor_tokens.numpy()) for _obj in seq_res])
    # check the variery (confirming random seeds)
    # L2 distance must be > 0.0 if token-id tensor has the variations.
    stack_l2_tensor_out = []
    for _i_ind, _obj_i in enumerate(seq_res):
        for _j_ind, _obj_j in enumerate(seq_res):
            if _i_ind == _j_ind:
                continue
            # end if
            _array_pad_i = np.pad(_obj_i.target_tensor_tokens.numpy(), (0, max_len_tensor - len(_obj_i.target_tensor_tokens)))
            _array_pad_j = np.pad(_obj_j.target_tensor_tokens.numpy(), (0, max_len_tensor - len(_obj_j.target_tensor_tokens)))
            _l2_ij = sum((_array_pad_i - _array_pad_j) ** 2)
            stack_l2_tensor_out.append(_l2_ij)
        # end for
    # end for

    assert sum(stack_l2_tensor_out) > 0

    shutil.rmtree(handler.path_cache_dir)



def test_FaiseqTranslationModelHandlerVer2WordEmbeddings_sampling_multi_input(resource_path_root: Path):
    from summary_abstractive.module_model_handler.ver2 import module_statics


    document_input = EvaluationTargetTranslationPair(
        source='He\'s a blue chip college basketball recruit. She\'s a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn\'t be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School\'s basketball team in Louisville, Kentucky, who\'s headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern\'s prom. So why is he taking Ellie instead? "She\'s great... she listens and she\'s easy to talk to" he said. Trey made the prom-posal (yes, that\'s what they are calling invites to prom these days) in the gym during Ellie\'s P.E. class. Trina Helson, a teacher at Eastern, alerted the school\'s newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn\'t surpristed by Trey\'s actions. "That\'s the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let\'s Party Like it\'s 1989," a reference to the latest album by Taylor Swift, Ellie\'s favorite singer. Trey also got the OK from Ellie\'s parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie\'s mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey\'s future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he\'s known for a long time, often works with other kids . Trey\'s mother, Shelly Moses, was also proud of her son. "It\'s exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he\'s a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can\'t stop thinking about prom. "Ellie can\'t wait to go dress shopping" her mother said. "Because I\'ve only told about a million people!" Ellie interjected.',
        target='',
        sentence_id='test'
    )
    penalty_command = 'none'

    setup_obj = long_running_setup(resource_path_root)
    handler: FaiseqTranslationModelHandlerVer2WordEmbeddings = setup_obj['handler']

    tensor_source_tokens = handler.bart_model.encode(document_input.source)
    tensor_source_tokens = tensor_source_tokens.to(torch.int64)

    seq_res = handler._sampling_multi_input(
        n_sampling=5,
        temperature=0.5,
        penalty_command=penalty_command,
        min_len=module_statics.MIN_LEN,
        max_len_a=module_statics.MAX_LEN_A,
        max_len_b=module_statics.MAX_LEN_B,
        source_text=document_input.source,
        tensor_source_tokens=tensor_source_tokens,
        length_penalty=module_statics.LENPEN,
        no_repeat_ngram_size=module_statics.NO_REPEAT_NGRAM_SIZE
    )
    assert len(seq_res) == 5

    max_len_tensor = max([len(_obj.target_tensor_tokens.numpy()) for _obj in seq_res])
    # check the variery (confirming random seeds)
    # L2 distance must be > 0.0 if token-id tensor has the variations.
    stack_l2_tensor_out = []
    for _i_ind, _obj_i in enumerate(seq_res):
        for _j_ind, _obj_j in enumerate(seq_res):
            if _i_ind == _j_ind:
                continue
            # end if
            _array_pad_i = np.pad(_obj_i.target_tensor_tokens.numpy(), (0, max_len_tensor - len(_obj_i.target_tensor_tokens)))
            _array_pad_j = np.pad(_obj_j.target_tensor_tokens.numpy(), (0, max_len_tensor - len(_obj_j.target_tensor_tokens)))
            _l2_ij = sum((_array_pad_i - _array_pad_j) ** 2)
            stack_l2_tensor_out.append(_l2_ij)
        # end for
    # end for

    assert sum(stack_l2_tensor_out) > 0

    shutil.rmtree(handler.path_cache_dir)
