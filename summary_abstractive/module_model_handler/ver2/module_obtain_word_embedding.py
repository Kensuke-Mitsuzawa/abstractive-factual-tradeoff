from torch.nn.modules.sparse import Embedding
from fairseq.models.bart import BARTHubInterface

import torch

def obtain_word_embedding(bart_model: BARTHubInterface,
                          tensor_token_id: torch.Tensor) -> torch.Tensor:

    # |V| -> 1024, where 1024 is the word embedding size
    layer_word_embedding: Embedding = bart_model.model.decoder.embed_tokens

    # test obtaining word embedding
    tensor_word_embed: torch.Tensor = layer_word_embedding(tensor_token_id.to(bart_model.device))
    assert isinstance(tensor_word_embed, torch.Tensor)

    return tensor_word_embed
