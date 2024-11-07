import logging

import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TransformerManager:
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self._load_model(model_name)
        self._setup_embeddings()

    def _load_model(self, model_name):
        self.model = AutoModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            output_attentions=True,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model.eval()

    def _setup_embeddings(self):
        self.position_embedding_start_idx = 2
        self.head_size = self.config.hidden_size // self.config.num_attention_heads
        self.vocab_base_embeddings = self.model.embeddings.word_embeddings.weight.data
        self.positional_embeddings = self.model.embeddings.position_embeddings.weight
        self.layer_norm = self.model.embeddings.LayerNorm
        self.normalized_vocab_embeddings = self.layer_norm(self.vocab_base_embeddings)
        self.position_adjusted_vocab = None
        self.normalized_position_adjusted_vocab = None

    def get_layer_context_weights(self, layer):
        return self.model.encoder.layer[layer].attention.output.dense.weight

    def get_raw_vocab_embeddings(self, selected_token_idx=None, apply_positional_embeddings=True ):
        if apply_positional_embeddings:
            self.adjust_vocab_to_token_position(selected_token_idx)
            return F.normalize(self.position_adjusted_vocab, p=2, dim=-1)
        else:
            return F.normalize(self.vocab_base_embeddings, p=2, dim=-1)

    def get_normalized_vocab_embeddings(self, apply_positional_embeddings=True):
        if apply_positional_embeddings:
            return self.normalized_position_adjusted_vocab
        else:
            return self.normalized_vocab_embeddings

    def adjust_vocab_to_token_position(self, selected_token_idx):
        self.position_adjusted_vocab = (
            self.vocab_base_embeddings
            + self.positional_embeddings[
                selected_token_idx + self.position_embedding_start_idx
            ]
        )
        self.normalized_position_adjusted_vocab = self.layer_norm(
            self.position_adjusted_vocab
        )

        logging.debug(selected_token_idx)
        logging.debug(f"{self.vocab_base_embeddings[selected_token_idx][:5]=}")
        logging.debug(
            f"{self.positional_embeddings[selected_token_idx + self.position_embedding_start_idx][:5]=}"
        )
        logging.debug(f"{self.position_adjusted_vocab[selected_token_idx][:5]=}")

    def get_layer_value_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.value.weight

    def get_layer_attention_weights(self, outputs, selected_layer):
        attentions = outputs.attentions[selected_layer]
        return attentions
