import typing as ty
import torch
import logging
import tempfile
import random
import copy
from pathlib import Path

import zlib
import pickle

import GPUtil

from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.bart import BARTHubInterface
from fairseq.models.bart import BARTModel

from ..ver2.module_base import EvaluationTargetTranslationPair
# from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...exceptions import ParameterSettingException
from ..ver2.module_base import TranslationResultContainer

from . import module_statics
from . import utils
from . import module_obtain_word_embedding
from .module_base import (
    BaseTranslationModelHandlerVer2,
    TranslationResultContainer,
    EvaluationTargetTranslationPair)


module_logger = logging.getLogger(__name__)

class FaiseqTranslationModelHandlerVer2WordEmbeddings(BaseTranslationModelHandlerVer2):
    def __init__(self,
                 path_dir_fairseq_model: Path,
                 is_select_gpu_flexible: bool = True,
                 path_cache_dir: ty.Optional[Path] = None, 
                 is_use_cache: bool = True,
                 random_seed: int = 42,
                 is_zlib_compress: bool = True,
                 is_save_convert_float16: bool = False,
                 torch_device: ty.Optional[torch.device] = None
                 ):
        """A class for handling the Fairseq translation model.
        
        Args:
            random_seed: A random seed for the FairSeq Call. 
                If -1, the random seed is set randomly.
        """
        super().__init__(
            path_cache_dir=path_cache_dir,
            is_use_cache=is_use_cache,
            is_zlib_compress=is_zlib_compress,
            is_save_convert_float16=is_save_convert_float16)            

        # self.model_encoder_decoder_mt = model_encoder_decoder_mt
        self.is_select_gpu_flexible: bool = is_select_gpu_flexible
        self.random_seed = random_seed

        assert path_dir_fairseq_model.exists()

        self.path_dir_fairseq_model = path_dir_fairseq_model
        
        self.bart_model = self._load_model(self.path_dir_fairseq_model)

        if self.is_select_gpu_flexible:
            if torch.cuda.is_available():
                _device_id = self._get_less_busy_cuda_device()
                self.torch_device = torch.device(_device_id)
            else:
                self.torch_device = torch.device('cpu')
        else:
            self.torch_device = torch_device
        # end if

        self.bart_model = self.bart_model.to(self.torch_device)


    @staticmethod
    def _load_model(path_dir_fairseq_model: Path) -> BARTHubInterface:
        assert path_dir_fairseq_model.exists()
        path_model_pt = path_dir_fairseq_model / 'model.pt'
        assert path_model_pt.exists()

        return utils.load_model(path_dir_fairseq_model, path_model_pt)

    # ------------------------------
    # utils from ver2 (backport)

    def get_all_possible_layers(self) -> ty.Tuple[ty.List[str], ty.List[str]]:
        return [], [self._get_decoder_word_embedding_layer_name()]

    # def _get_decoder_word_embedding_layer_name(self) -> str:
    #     return "decoder.word_embedding"
    
    # def _get_cache_file_name(self, sentene_id: str, is_zlib_compress: bool) -> str:
    #     if is_zlib_compress:
    #         return f'{sentene_id}.pkl.zlib'
    #     else:
    #         return f'{sentene_id}.pt'

    # def _generate_cache_file_path(self, 
    #                               sentene_id: str,
    #                               tau_parameter: float,
    #                               n_sampling: ty.Optional[int],
    #                               is_zlib_compress: bool = True
    #                               ) -> Path:
    #     _file_name = self._get_cache_file_name(sentene_id, is_zlib_compress=is_zlib_compress)

    #     if n_sampling is None:
    #         return self.path_cache_dir / self.__class__.__name__.__str__() / 'beam' / str(tau_parameter) / _file_name
    #     else:
    #         assert n_sampling is not None
    #         return self.path_cache_dir / self.__class__.__name__.__str__() / 'stochastic' / str(tau_parameter) / str(n_sampling) / _file_name
    #     # end if

    # def _save_cache(self, 
    #                 sentence_id: str,
    #                 tau_param: float,
    #                 translation_obj: ty.Union[TranslationResultContainer, ty.List[TranslationResultContainer]],
    #                 n_sampling: ty.Optional[int]):
    #     if self.is_save_convert_float16:
    #         # converting float32 object into float16.
    #         if isinstance(translation_obj, TranslationResultContainer):
    #             translation_obj = translation_obj.convert_embedding_float16()
    #         else:
    #             translation_obj = [o.convert_embedding_float16() for o in translation_obj]
    #         # end if
    #     # end if

    #     if isinstance(translation_obj, TranslationResultContainer):
    #         _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=None, is_zlib_compress=self.is_zlib_compress)
    #         _obj = translation_obj._asdict()
    #     elif isinstance(translation_obj, list):
    #         _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=self.is_zlib_compress)
    #         _obj = [o._asdict() for o in translation_obj]
    #     else:
    #         raise TypeError()
    #     # end if
            
    #     _path_file.parent.mkdir(parents=True, exist_ok=True)

    #     if self.is_zlib_compress:
    #         pickled_data = pickle.dumps(_obj)
    #         compressed_data_zlib = zlib.compress(pickled_data)
    #         with open(_path_file, "wb") as f:
    #             f.write(compressed_data_zlib)
    #         # end with
    #     else:
    #         torch.save(_obj, _path_file)
    #     # end if

    # def _load_cache(self, 
    #                 sentence_id: str,
    #                 tau_param: float,
    #                 n_sampling: ty.Optional[int]                    
    #                 ) -> ty.Optional[ty.Union[TranslationResultContainer, ty.List[TranslationResultContainer]]]:
    #     _path_file_zlib = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=True)
    #     _path_file_pt = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=False)

    #     try:
    #         if _path_file_zlib.exists():
    #             with _path_file_zlib.open('rb') as f:
    #                 obj_saved = pickle.loads(zlib.decompress(f.read()))
    #         elif _path_file_pt.exists():
    #             obj_saved = torch.load(_path_file_pt)
    #         else:
    #             return None
    #         # end with
    #     except (zlib.error, IOError) as e:
    #         # the cache file is broken.
    #         return None
    #     # end if
        
    #     if isinstance(obj_saved, list):
    #         obj_cache = [TranslationResultContainer(**o) for o in obj_saved]
    #     else:
    #         obj_cache = TranslationResultContainer(**obj_saved)
    #     # end if
    #     return obj_cache
    
    # def _is_exist_cache(self, 
    #                     sentence_id: str,
    #                     tau_param: float,
    #                     n_sampling: ty.Optional[int]) -> ty.Optional[Path]:
    #     _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=True)
    #     if _path_file.exists():
    #         return _path_file
    #     # end if
    #     _path_file = self._generate_cache_file_path(sentence_id, tau_parameter=tau_param, n_sampling=n_sampling, is_zlib_compress=False)
    #     if _path_file.exists():
    #         return _path_file
    #     # end if
    #     #
    #     return None        

    # def _is_exist_cache_or_fetch(self, 
    #                              sentence_id: str,
    #                              tau_param: float,
    #                              n_sampling: ty.Optional[int]
    #                              ) -> ty.Union[bool, TranslationResultContainer, ty.List[TranslationResultContainer]]:
    #     _path_file = self._is_exist_cache(sentence_id, tau_param=tau_param, n_sampling=n_sampling)
    #     if _path_file is None:
    #         return False
    #     # end if
        
    #     if _path_file.exists():
    #         _obj = self._load_cache(sentence_id, tau_param=tau_param, n_sampling=n_sampling)
    #         if _obj is None:
    #             return False
    #         else:
    #             return _obj
    #     else:
    #         return False

    # ------------------------------
    # Sampling

    def _sampling_multi_input(self,
                              source_text: str,
                              tensor_source_tokens: torch.Tensor,
                              penalty_command: str,
                              temperature: float,
                              n_sampling: int,
                              min_len: int,
                              max_len_a: float,
                              max_len_b: int,
                              length_penalty: float,
                              no_repeat_ngram_size: int,
                              batch_size: int = 5,
                              ) -> ty.List[TranslationResultContainer]:
        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."
        seq_random_seed_values = list(range(0, 9999))
        _gen_random.shuffle(seq_random_seed_values)

        i_iteration = 0
        seq_stack_sampling = []
        while len(seq_stack_sampling) < n_sampling:
            extractive_penalty_fct = utils.get_extractive_penalty_fct(penalty_command)

            if len(seq_stack_sampling) < (n_sampling - batch_size):
                seq_input_tensor = [tensor_source_tokens] * batch_size
            else:
                seq_input_tensor = [tensor_source_tokens] * (n_sampling - len(seq_stack_sampling))
            # end if

            dict_parameters = dict(
                beam=1,
                lenpen=length_penalty,
                sampling=True,
                temperature=temperature,
                min_len=min_len, 
                max_len_a=max_len_a, 
                max_len_b=max_len_b,
                no_repeat_ngram_size=no_repeat_ngram_size,
                extractive_penalty_fct=extractive_penalty_fct)

            try:
                _tensor_token_id_cuda = torch.stack(seq_input_tensor).to(self.bart_model.device)
            except RuntimeError:
                error_message = f'Attempting transfer the input-id tensor to cuda, failed to do. dtype(tensor_source_tokens) = {tensor_source_tokens.dtype}, tensor-shape={tensor_source_tokens.shape}, tensor={tensor_source_tokens}'
                raise ParameterSettingException(error_message)
            else:
                pass
            # end try block.

            with torch.random.fork_rng():
                torch.manual_seed(seq_random_seed_values[i_iteration])
                torch.cuda.manual_seed_all(seq_random_seed_values[i_iteration])  # if you are using multi-GPU.
                
                generated_obj = self.bart_model.generate(_tensor_token_id_cuda, **dict_parameters)
            # end with

            _dict_parameters_cached = copy.deepcopy(dict_parameters)
            _dict_parameters_cached['random_seed'] = seq_random_seed_values[i_iteration]
            seq_stack_sampling.append([_dict_parameters_cached, generated_obj])
            i_iteration += 1
        # end with

        # post-processing
        seq_output_obj = []
        for _index_in_batch, _t_exec_result in enumerate(seq_stack_sampling):
            _d_argument = _t_exec_result[0]
            _obj_translations = _t_exec_result[1]            

            _n_tokens = len(_obj_translations[0]['tokens'])  # type: ignore
            _tensor_translation = _obj_translations[0]['tokens']

            # getting the word embedding            
            _dict_layer_embeddings = {
                self._get_decoder_word_embedding_layer_name(): module_obtain_word_embedding.obtain_word_embedding(self.bart_model, _tensor_translation).cpu()
            }            
    
            _log_score = _obj_translations[0]['score'].item()
            text_translation: str = self.bart_model.decode(_obj_translations[0]['tokens'])  # type: ignore

            return_obj = TranslationResultContainer(
                source_text=source_text,
                translation_text=text_translation,
                source_tensor_tokens=tensor_source_tokens.cpu(),
                target_tensor_tokens=_obj_translations[0]['tokens'].cpu(),
                source_language='source',
                target_language='target',
                log_probability_score=_log_score,
                dict_layer_embeddings=_dict_layer_embeddings,
                argument_translation_conditions=_d_argument
            )
            seq_output_obj.append(return_obj)
        # end for

        return seq_output_obj

    def _sampling_single_input(self,
                               source_text: str,
                               tensor_source_tokens: torch.Tensor,
                               penalty_command: str,
                               temperature: float,
                               n_sampling: int,
                               min_len: int,
                               max_len_a: float,
                               max_len_b: int,
                               length_penalty: float,
                               no_repeat_ngram_size: int,
                               n_max_attempts: int = 10
                               ) -> ty.List[TranslationResultContainer]:
        # Note: Possible `generate` options are defined at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810 
        # `generate` method executes `inference_step` method of `TranslationTask` class.
        # `**kwargs` arguments are passed to `build_generator` method of `TranslationTask` class first,
        # and then, the generator object is passed to the `inference_step` method.
        # See `build_generator` API at here: https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.build_generator
        # The args object is `fairseq.dataclass.configs.GenerationConfig`.
        # The `GenerationConfig` definition is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810
        
        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."
        seq_random_seed_values = list(range(0, 9999))
        _gen_random.shuffle(seq_random_seed_values)

        output_stack = []
        i_error_attempt = 0
        i_sampling = 0

        extractive_penalty_fct = utils.get_extractive_penalty_fct(penalty_command)

        dict_parameters = dict(
            beam=1,
            lenpen=length_penalty,
            sampling=True,
            temperature=temperature,
            min_len=min_len, 
            max_len_a=max_len_a, 
            max_len_b=max_len_b,
            no_repeat_ngram_size=no_repeat_ngram_size,
            extractive_penalty_fct=extractive_penalty_fct)

        while len(output_stack) < n_sampling:
            try:
                _tensor_token_id_cuda = torch.stack([tensor_source_tokens]).to(self.bart_model.device)
            except RuntimeError:
                error_message = f'Attempting transfer the input-id tensor to cuda, failed to do. dtype(tensor_source_tokens) = {tensor_source_tokens.dtype}, tensor-shape={tensor_source_tokens.shape}, tensor={tensor_source_tokens}'
                raise ParameterSettingException(error_message)
            else:
                pass
            # end try block.

            try:
                with torch.random.fork_rng():
                    torch.manual_seed(seq_random_seed_values[i_sampling])
                    torch.cuda.manual_seed_all(seq_random_seed_values[i_sampling])  # if you are using multi-GPU.
                    
                    generated_obj = self.bart_model.generate(_tensor_token_id_cuda, **dict_parameters)
                # end with
            except (AssertionError, RuntimeError) as e:
                if i_error_attempt >= n_max_attempts:
                    error_message = (
                            f"Exceeded the maximum number of attempts: {n_max_attempts}",
                            f"Exception: {e}",
                            f"With the temperature paramater = {temperature}",
                            f"Source Text: {_tensor_token_id_cuda}"
                    )
                    module_logger.error(error_message)
                    raise ParameterSettingException(error_message)
                else:
                    i_error_attempt += 1
                    i_sampling += 1
                    continue
                # end if
            else:
                _dict_parameters_cached = copy.deepcopy(dict_parameters)
                _dict_parameters_cached['random_seed'] = seq_random_seed_values[i_sampling]
                output_stack.append([_dict_parameters_cached, generated_obj])
                i_sampling += 1
            # end try
        # end while


        # post-processing
        seq_output_obj = []
        for _index_in_batch, _t_exec_result in enumerate(output_stack):
            _d_argument = _t_exec_result[0]
            _obj_translations = _t_exec_result[1]            

            _n_tokens = len(_obj_translations[0]['tokens'])  # type: ignore
            _tensor_translation = _obj_translations[0]['tokens']

            # getting the word embedding            
            _dict_layer_embeddings = {
                self._get_decoder_word_embedding_layer_name(): module_obtain_word_embedding.obtain_word_embedding(self.bart_model, _tensor_translation).cpu()
            }            
    
            _log_score = _obj_translations[0]['score'].item()
            text_translation: str = self.bart_model.decode(_obj_translations[0]['tokens'])  # type: ignore

            return_obj = TranslationResultContainer(
                source_text=source_text,
                translation_text=text_translation,
                source_tensor_tokens=tensor_source_tokens.cpu(),
                target_tensor_tokens=_obj_translations[0]['tokens'].cpu(),
                source_language='source',
                target_language='target',
                log_probability_score=_log_score,
                dict_layer_embeddings=_dict_layer_embeddings,
                argument_translation_conditions=_d_argument
            )
            seq_output_obj.append(return_obj)
        # end for

        return seq_output_obj
    # end def

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]
    
    def _call_fairseq_interface_stochastic(self, 
                                source_text: str,
                                tensor_source_tokens: torch.Tensor,
                                temperature: float,
                                n_sampling: int,
                                min_len: int,
                                max_len_a: float,
                                max_len_b: int,
                                penalty_command: str,
                                length_penalty: float,
                                no_repeat_ngram_size: int,
                                is_sampling_in_iteration: bool = False,
                                is_auto_recovery_sampling: bool = True,
                                n_max_attempts: int = 100,
                                batch_size: int = 10,
                                target_layers_extraction: ty.Optional[ty.List[str]] = None
                                ) -> ty.List[TranslationResultContainer]:
        """Simply, I call the fairseq interface to generate translations.
        This interface has dedicated procedures for calling the fairseq translation model since the interface often causes assertion errors when the temperature is a small value.
        See the description of `is_auto_recovery_sampling` for the details.

        Args:
            is_auto_recovery_sampling: If True, the function tries to recover the sampling process when the assertion error occurs.
                It switches the sampling method to the iteration-based sampling automatically when this method encounters the assertion error.
            n_max_attempts: The maximum number of attempts to recover the sampling process.
                When the attemtps exceed this value, the function raises an exception.
        """
        tensor_source_tokens = tensor_source_tokens.to(torch.int64)

        with torch.no_grad():
            # The block below is still in work
            if is_sampling_in_iteration:
                output_stack = self._sampling_single_input(
                    source_text=source_text,
                    tensor_source_tokens=tensor_source_tokens,
                    penalty_command=penalty_command,
                    temperature=temperature,
                    n_max_attempts=n_max_attempts,
                    n_sampling=n_sampling,
                    min_len=min_len,
                    max_len_a=max_len_a,
                    max_len_b=max_len_b,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size)
            else:
                try:
                    output_stack = self._sampling_multi_input(
                        source_text=source_text,
                        tensor_source_tokens=tensor_source_tokens,
                        penalty_command=penalty_command,
                        temperature=temperature,
                        n_sampling=n_sampling,
                        min_len=min_len,
                        max_len_a=max_len_a,
                        max_len_b=max_len_b,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        batch_size=batch_size)
                except (AssertionError, RuntimeError) as e:
                    if is_auto_recovery_sampling:
                        module_logger.warning(f"Assertion error occurred: {e}")
                        output_stack = self._sampling_single_input(
                            source_text=source_text,
                            tensor_source_tokens=tensor_source_tokens,
                            penalty_command=penalty_command,
                            temperature=temperature,
                            n_max_attempts=n_max_attempts,
                            n_sampling=n_sampling,
                            min_len=min_len,
                            max_len_a=max_len_a,
                            max_len_b=max_len_b,
                            length_penalty=length_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size)
                    else:
                        raise e
                    # end if
                # end try-except
            # end if
        # end with

        return output_stack

    
    # --------------------------------------------------------------
    # public methods

    def translate_beam_search(self,
                              input_text: EvaluationTargetTranslationPair,
                              penalty_command: str,
                              temperature: float = 1.0,
                              min_len: int = module_statics.MIN_LEN,
                              max_len_a: float = module_statics.MAX_LEN_A,
                              max_len_b: int = module_statics.MAX_LEN_B,
                              length_penalty: float = module_statics.LENPEN,
                              no_repeat_ngram_size: int = module_statics.NO_REPEAT_NGRAM_SIZE
                              ) -> TranslationResultContainer:
        """
        Args:

        Exception:
            `ParameterSettingException`
        """
        if self.is_use_cache:
            false_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=None)
            
            if isinstance(false_or_cache, TranslationResultContainer):
                return false_or_cache
            # end if
        # end if
        
        source_text = input_text.source

        extractive_penalty_fct = utils.get_extractive_penalty_fct(penalty_command)

        dict_parameters = dict(
            beam=1,
            lenpen=length_penalty,
            sampling=False,
            min_len=min_len, 
            max_len_a=max_len_a, 
            max_len_b=max_len_b,
            no_repeat_ngram_size=no_repeat_ngram_size,
            extractive_penalty_fct=extractive_penalty_fct,
            temperature=temperature)


        tensor_source_tokens = self.bart_model.encode(source_text)
        
        try:
            generated_obj = self.bart_model.generate(torch.stack([tensor_source_tokens]).to(self.bart_model.device), **dict_parameters)
        except (AssertionError, RuntimeError) as e:
            error_message = (
                    f"Exception at the beam-search = {temperature}",
                    f"Source Text: {tensor_source_tokens}"
            )
            raise ParameterSettingException(error_message)
        # end try

        _tensor_translation = generated_obj[0]['tokens']

        # getting the word embedding            
        _dict_layer_embeddings = {
            self._get_decoder_word_embedding_layer_name(): module_obtain_word_embedding.obtain_word_embedding(self.bart_model, _tensor_translation).cpu()
        }            

        _log_score = generated_obj[0]['score'].item()
        text_translation: str = self.bart_model.decode(_obj_translations[0]['tokens'])  # type: ignore

        return_obj = TranslationResultContainer(
            source_text=source_text,
            translation_text=text_translation,
            source_tensor_tokens=tensor_source_tokens.cpu(),
            target_tensor_tokens=_tensor_translation.cpu(),
            source_language='source',
            target_language='target',
            log_probability_score=_log_score,
            dict_layer_embeddings=_dict_layer_embeddings,
            argument_translation_conditions=dict_parameters
        )

        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                translation_obj=return_obj,
                tau_param=temperature,
                n_sampling=None)
        # end if

        return return_obj        
    

    def translate_sample_multiple_times(self,
                                        input_text: EvaluationTargetTranslationPair,
                                        n_sampling: int,
                                        temperature: float,
                                        penalty_command: str,
                                        min_len: int = module_statics.MIN_LEN,
                                        max_len_a: float = module_statics.MAX_LEN_A,
                                        max_len_b: int = module_statics.MAX_LEN_B,
                                        length_penalty: float = module_statics.LENPEN,
                                        no_repeat_ngram_size: int = module_statics.NO_REPEAT_NGRAM_SIZE,
                                        n_max_attempts: int = 10,
                                        batch_size: int = 50,
                                        target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                        is_sampling_in_iteration: bool = False,
                                        is_auto_recovery_sampling: bool = True,
                                        ) -> ty.List[TranslationResultContainer]:
        if self.is_use_cache:
            exists_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=n_sampling)
            if isinstance(exists_or_cache, list):
                module_logger.debug(f"Cache found for id={input_text.sentence_id} and tau={temperature} and n-sample={n_sampling}")
                return exists_or_cache
            # end if        

        source_text = input_text.source

        encoded_obj = self.bart_model.encode(source_text)
        encoded_obj = encoded_obj.to(torch.int)

        seq_result = self._call_fairseq_interface_stochastic(
            source_text=source_text,
            tensor_source_tokens=encoded_obj,
            temperature=float(temperature),  # note: error occurs when the variable by numpy's. So, I forcelly put back to the Python's type.
            n_sampling=n_sampling,
            min_len=min_len,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            penalty_command=penalty_command,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            n_max_attempts=n_max_attempts,
            target_layers_extraction=target_layers_extraction,
            batch_size=batch_size,
            is_sampling_in_iteration=is_sampling_in_iteration,
            is_auto_recovery_sampling=is_auto_recovery_sampling)

        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                tau_param=temperature,
                n_sampling=n_sampling,
                translation_obj=seq_result)
        # end if

        return seq_result
