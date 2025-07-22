"""A prototype script of computing Monte-Carlo Similarity.
This script is specially designed for fairseq 0.9.0.
The later version fairseq has different approach in the Dropout."""

# %%
import torch
from pathlib import Path
import typing as ty
import random

from summary_abstractive.module_model_handler.ver2 import module_statics
from summary_abstractive.module_model_handler.ver2 import utils

from summary_abstractive.module_model_handler.ver2 import (
    TranslationResultContainer,
    EvaluationTargetTranslationPair,
    FaiseqTranslationModelHandlerVer2WordEmbeddings,
    module_statics)
from summary_abstractive.module_model_handler.ver2.module_obtain_word_embedding import obtain_word_embedding

import abc

from datetime import datetime

import json
import numpy as np

from fairseq.models.bart import BARTHubInterface
from fairseq.models.bart import BARTModel

from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.sequence_generator import SequenceGenerator


import logzero


# %% Setting the logger directory


PATH_MC_SIM = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/logs/mc_sim")
PATH_MC_SIM.mkdir(parents=True, exist_ok=True)

_log_file_name = datetime.now().isoformat()

logzero.logfile(PATH_MC_SIM / f'{_log_file_name}.log')

logger = logzero.logger

# %%

try:
    import nltk
    from nltk.translate import meteor_score
    from nltk.tokenize import word_tokenize
except ImportError:
    raise Exception("You have to install nltk package first.")
# end try


# %% class definition

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


class MonteCarloSimilarityResult(ty.NamedTuple):
    document_id: str
    original_document_id: str
    abstractiveness_constraint: str

    document_source: str
    summary_reference: str

    generated_summary: ty.List[ty.Dict]

    metric: str
    similarity_score: ty.List[float]
    avg_similarity_score: float




class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self,
                 reference: str,
                 seq_hypothesis: ty.List[str]) -> ty.List[float]:
        pass


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        # setup nltk tokenizer
        nltk.download('punkt_tab')
        nltk.download('wordnet')

    def __call__(self,
                 reference: str,
                 seq_hypothesis: ty.List[str]) -> ty.List[float]:
        """Compute METEOR over multiple MC Dropout samples."""
        scores = []
        for _hyp in seq_hypothesis:
            assert isinstance(_hyp, str), f"_hyp must be a string"
            _seq_tokens_hyp = word_tokenize(_hyp)
            assert isinstance(_seq_tokens_hyp, list), f"_seq_tokens must be a list"

            _seq_tokens_ref = word_tokenize(reference)
            assert isinstance(_seq_tokens_ref, list), f"_seq_tokens must be a list"

            # _score = meteor_score.stem_match([_seq_tokens_ref], _seq_tokens_hyp)
            _score = meteor_score.meteor_score([_seq_tokens_ref], _seq_tokens_hyp)
            scores.append(_score)
        # end for

        return scores



# %% functions


def set_config_dropout_mode(fairseq_interface: BARTHubInterface, dropout: ty.Union[str, float] = 'default') -> ty.List[ty.Dict]:

    # setting the mode into the training mode.
    fairseq_interface.train()
    fairseq_interface.model.train()

    seq_config_obj = []
    for name, module in fairseq_interface.model.named_modules():
        # print(name, module.training, type(module))

        if isinstance(module, TransformerDecoderLayer):
            assert hasattr(module, 'dropout')
            module.training = True
            if dropout == 'default':
                pass
            else:
                assert isinstance(dropout, float)
                module.dropout = dropout
            # end if

            _rate_dropout = module.dropout  # you can set any dropout parameter.
            logger.debug(f"Activated dropout modules. Layer: {name} and Class {type(module)}, Training-mode: {module.training}, Dropout-rate={_rate_dropout}")

            _dict_config = dict(
                layer=name,
                class_name=str(type(module)),
                dropout_rate=_rate_dropout
            )

            seq_config_obj.append(_dict_config)
        # end if
    # end for
    return seq_config_obj



def get_parameter_config(penalty_command: str, retain_dropout: bool = True):

    extractive_penalty_fct: str = utils.get_extractive_penalty_fct(penalty_command)
    
    dict_parameters = dict(
        # beam=module_statics.BEAM,
        beam_size=module_statics.BEAM,
        # lenpen=module_statics.LENPEN,
        len_penalty=module_statics.LENPEN,
        # sampling=False,
        min_len=module_statics.MIN_LEN, 
        max_len_a=module_statics.MAX_LEN_A, 
        max_len_b=module_statics.MAX_LEN_B,
        no_repeat_ngram_size=module_statics.NO_REPEAT_NGRAM_SIZE,
        extractive_penalty_fct=extractive_penalty_fct,
        # temperature=1.0,
        retain_dropout=retain_dropout
    )

    return dict_parameters



# %%



def get_dropout_samples(source_text: str, penalty_command: str, retain_dropout: bool = True, num_samples: int = 10) -> ty.List[ty.Dict]:

    # %% setting the sequence generator object.

    fairseq_interface.model.train()  # keep dropout active
    fairseq_interface.model.eval = lambda *args, **kwargs: None  # disable eval()

    dict_parameters = get_parameter_config(retain_dropout=retain_dropout, penalty_command=penalty_command)

    generator = SequenceGenerator(
        [fairseq_interface.model],
        fairseq_interface.task.target_dictionary,
        **dict_parameters
    )

    # %% main loop.

    generated_outputs = []
    for _ in range(num_samples):
        with torch.no_grad():
            _encoded_tensor = fairseq_interface.encode(source_text)

            src_tokens = _encoded_tensor.unsqueeze(0)  # [1, seq_len]
            src_lengths = torch.LongTensor([src_tokens.size(1)])        

            sample = {
                "net_input": {
                    "src_tokens": src_tokens.to(fairseq_interface.device),       # assuming you're on GPU
                    "src_lengths": src_lengths.to(fairseq_interface.device),
                }
            }

            _out_obj = generator(sample)
        # end with
        _text_generated = fairseq_interface.decode(_out_obj[0][0]['tokens'])

        _obj_generated = dict(
            source_text=source_text,
            generation_parameters=dict_parameters,
            tensor_generated=_out_obj[0][0]['tokens'].detach().cpu(),
            text_generated=_text_generated
        )

        generated_outputs.append(_obj_generated)
        # print(_text_generated)
    # end for
    return generated_outputs

# %% Test settting

def _run_tests():

    num_samples = 10

    source_text = """The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that \"so far no videos were used in the crash investigation.\" He added, \"A person who has such a video needs to immediately give it to the investigators.\" Robin's comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported. \"Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing.\" \"It is a very disturbing scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were \"completely wrong\" and \"unwarranted.\" Cell phones have been collected at the site, he said, but that they \"hadn't been exploited yet.\" Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical \"no.\" Reichelt told \"Erin Burnett: Outfront\" that he had watched the video and stood by the report, saying Bild and Paris Match are \"very confident\" that the clip is real. He noted that investigators only revealed they'd recovered cell phones from the crash site after Bild and Paris Match published their reports. \"That is something we did not know before. ... Overall we can say many things of the investigation weren't revealed by the investigation at the beginning,\" he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week 
    in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a \"previous episode of severe depression,\" the airline said Tuesday. Email correspondence between Lubitz and the school discovered in 
    an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz'
    s battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a \"swift and
    seamless clarification\" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been 
    working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash sit
    e, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois H
    ollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims'
    personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The de
    tails about Lubitz's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz's possible motive for downing the jet. A
    Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and \"held all the licenses required.\" Earlier, a spokesman for the prosecutor's office in Dusseldorf, 
    Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidenc
    e suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot's license, a European government official 
    briefed on the investigation told CNN on Tuesday. While flying was \"a big part of his life,\" the source said, it's only one theory being considered. Another source, a law enforcement official briefed on the investigation, a
    lso told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had seen
    an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous men
    tal health struggles, there's more to the story, said Brian Russell, a forensic psychologist. \"Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren't going to keep doi
    ng their job and they're upset about that and so they're suicidal,\" he said. \"But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who 
    had nothing to do with the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, whil
    e Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.","annotator_votes_combined":1,"summary_raw":"Lufthansa confir
    ms co-pilot Andreas Lubitz battled depression years before crash. A French prosecutor says he is not aware of any video footage from on board the plane. German daily Bild and French Paris Match claim to have a cell phone vide
    o of the crash. \" It is a very disturbing scene,\" says Bild's editor-in-chief."""


    # set config
    seq_config_obj = set_config_dropout_mode(fairseq_interface=fairseq_interface)

    # Test Section.
    # 
    #  generating the samples without dropout
    seq_generated_without_dropout = get_dropout_samples(source_text=source_text, penalty_command='none', retain_dropout=False, num_samples=num_samples)
    assert len(set([_obj['text_generated'] for _obj in seq_generated_without_dropout])) == 1


    #  generating the samples with dropout
    seq_generated_without_dropout = get_dropout_samples(source_text=source_text, penalty_command='none', retain_dropout=True, num_samples=num_samples)
    assert len(set([_obj['text_generated'] for _obj in seq_generated_without_dropout])) > 1
# end def


# %% loading dataset
PATH_DATASET_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/datasets/constraints_fact_v1.0/cnn_dailymail/collect.json")
assert PATH_DATASET_CNN.exists()

with PATH_DATASET_CNN.open('r') as f:
    seq_dataset_obj = [json.loads(_line) for _line in f.readlines()]
# end with


# %% loading the handler. note: I do not use the handler. But, the handler easily loads the model file.

PATH_MODEL_BART_CNN = Path("/workdir/kmitsuzawa/Project/neurips-2025/ConstraintsFact-Dreyer-2023/abstractive-factual-tradeoff/tests/testresources/models/bart.large.cnn")
assert PATH_MODEL_BART_CNN.exists()

PATH_CACHE_DIR_BASE = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/summary_cache")
assert PATH_CACHE_DIR_BASE.exists()

summary_model_handler = FaiseqTranslationModelHandlerVer2WordEmbeddings(
    path_cache_dir=PATH_CACHE_DIR_BASE,
    path_dir_fairseq_model=PATH_MODEL_BART_CNN
)


PATH_DIR_RESULT_OUTPUT = Path("/workdir/kmitsuzawa/DATA/mitsuzaw/project_UCA/MT_MMD/flagging_dreyer_2023/Dreyer_2023-constraints_fact_CNN-Baseline-2025-07-21/results/mc_sim/result-2025_07_22")
PATH_DIR_RESULT_OUTPUT.mkdir(parents=True, exist_ok=True)

# %%

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('')
# end if

# %% 

metric_obj = MeteorMetric()

fairseq_interface = summary_model_handler.bart_model
fairseq_interface = fairseq_interface.to(device)

# set config and dropout. Setting dropout=0.3, following Guerreiro, 2023 "Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation"
seq_config_obj = set_config_dropout_mode(fairseq_interface=fairseq_interface,
                                         dropout=0.3)


# allocating the unique id to all records
dict_unique_id2records = {
    UniqueDocumentId(datasource=_obj['dataset_name'], document_id_original=_obj['document_id'], abstractiveness_constraint=_obj['abstractiveness_constraint']): _obj 
    for _obj in seq_dataset_obj
}
logger.info(f'Count Unique document -> {len(dict_unique_id2records)}')


# Test function
# _run_tests()

path_dir_generated_torch_files = PATH_DIR_RESULT_OUTPUT / "generated_torch"
path_dir_generated_torch_files.mkdir(parents=True, exist_ok=True)


path_quick_result = PATH_DIR_RESULT_OUTPUT / "mc_sim.jsonl"
result_f_obj = path_quick_result.open('w')

# %% dropout configuration file
path_configuration_settings = PATH_DIR_RESULT_OUTPUT / "dropout_configuration.json"
with  path_configuration_settings.open('w') as f:
    f.write(json.dumps(seq_config_obj, indent=4))
# end with


# %% main loop

for _unique_id, _record in dict_unique_id2records.items():
    _path_output_result = path_dir_generated_torch_files / f"{_unique_id.to_str()}.pt"
    
    if _path_output_result.exists():
        _obj_generated_result = torch.load(_path_output_result)
        
        _doc_id = _obj_generated_result['document_id']
        _avg_score = _obj_generated_result['avg_similarity_score']

        result_f_obj.write(json.dumps(dict(doc_id=_doc_id, avg_score=_avg_score)) + '\n')
        result_f_obj.flush()
    else:
        logger.info(f"Processing document-id {_unique_id}...")

        _document_unique_id: str = _unique_id.to_str()

        _document_id_original: str = _record["document_id"]

        _document_full: str = _record['document_full']
        _document_original: str = _record['document_original']
        _summary_raw: str = _record['summary_raw']
        
        _penalty_command: str = _record['abstractiveness_constraint']
        
        assert _document_full == _document_original

        _monte_carlo_generated_obj = get_dropout_samples(
            source_text=_document_original,
            penalty_command=_penalty_command,
            retain_dropout=True,
            num_samples=10
        )

        # computing the Meteor similarity
        _seq_generated_text: ty.List[str] = [_obj['source_text'] for _obj in _monte_carlo_generated_obj]
        _seq_metrics: ty.List[float] = metric_obj(reference=_summary_raw, seq_hypothesis=_seq_generated_text)
        _seq_metrics = [float(_score) for _score in _seq_metrics]
        _avg_meteor_metrics = float(np.mean(_seq_metrics))


        _similarity_result = MonteCarloSimilarityResult(
            document_id=_document_unique_id,
            original_document_id=_document_id_original,
            abstractiveness_constraint=_penalty_command,
            document_source=_document_original,
            summary_reference=_summary_raw,
            generated_summary=_monte_carlo_generated_obj,
            metric=str(metric_obj.__class__),
            similarity_score=_seq_metrics,
            avg_similarity_score=_avg_meteor_metrics
        )

        logger.info(f"Done processing document-id {_unique_id}. Avg-Sim -> {_similarity_result.avg_similarity_score}")
        torch.save(_similarity_result._asdict(), _path_output_result)

        _doc_id = _similarity_result.document_id
        _avg_score = _similarity_result.avg_similarity_score

        result_f_obj.write(json.dumps(dict(doc_id=_doc_id, avg_score=_avg_score)) + '\n')
        result_f_obj.flush()
# end for

result_f_obj.close()
