import time
from functools import wraps

import pandas as pd
import torch.nn.functional as F

from numpy_index_flat_ip import NumpyIndexFlatIP


def log_name_and_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Method '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


class AttentionAnalyzer:
    def __init__(self, transformer, k=15):
        self.transformer = transformer
        self.num_layers = self.transformer.config.num_hidden_layers
        self.k = k
        self._setup_indexes()

    def _setup_indexes(self, f=768):
        self.indexes = {}
        f = self.transformer.config.hidden_size  # 768
        index = NumpyIndexFlatIP(f)  # inner product of normalized vectors = cosine similarity
        vocab_embeddings = (
            self.transformer.get_raw_vocab_embeddings(
                selected_token_idx=0, apply_positional_embeddings=False
            )
            .detach()
            .numpy()
        )
        index.add(vocab_embeddings)  # type: ignore
        self.indexes["raw"] = index

        index = NumpyIndexFlatIP(f)  # inner product of normalized vectors = cosine similarity
        vocab_embeddings = (
            self.transformer.get_raw_vocab_embeddings(
                selected_token_idx=0, apply_positional_embeddings=True
            )
            .detach()
            .numpy()
        )
        index.add(vocab_embeddings)  # type: ignore
        self.indexes["positional"] = index

    def get_tokens_for_text(self, text):
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        tokens = self.transformer.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_list = [[token] for token in tokens]
        return token_list

    def compute_post_attention_values(self, selected_layer, outputs):
        """
        Compute the post-attention encodings for all tokens at a specified layer.
        Outputs a tensor of shape <seq_len, num_heads, embedding_size>.
        """
        selected_layer = int(selected_layer)
        # get weights for selected_layer
        selected_layer_value_weights = self.transformer.get_layer_value_weights(selected_layer)

        # actual input embeddings (not all-same-position-encoded)
        # <seq_len, hidden_size>, the input for selected_layer
        layer_sequence_inputs = outputs.hidden_states[selected_layer].squeeze(0)

        # multiply input encodings by value weights to get value encodings
        # so values for all tokens, all heads
        # <seq_len, hidden_size/num_heads>
        value_encodings = layer_sequence_inputs @ selected_layer_value_weights

        # get all attention weights for all heads, single layer, single token
        # <num_heads, seq_len, seq_len>
        layer_attention_weights = self.transformer.get_layer_attention_weights(
            outputs, selected_layer
        ).squeeze(0)

        # split value encoding into per-head encodings
        # <seq_len, num_heads, hidden_size/num_heads>
        encodings_per_head = value_encodings.view(
            value_encodings.size(0),
            self.transformer.config.num_attention_heads,
            self.transformer.head_size,
        )

        # permute encodings to <num_heads, seq_len, hidden_size/num_heads>
        encodings_per_head_permuted = encodings_per_head.permute(1, 0, 2)

        # multiply attention weights by encodings
        post_attn_layer_encodings = layer_attention_weights.bmm(encodings_per_head_permuted)

        # end up with <seq_len, num_heads, hidden_size/num_heads>
        post_attn_layer_encodings = post_attn_layer_encodings.permute(1, 0, 2)

        return post_attn_layer_encodings

    def get_closest_vocabulary_tokens(
        self,
        similarities,  # FAISS inner product results (D)
        indices,  # FAISS indices array (I)
        selected_token_id,
        k=15,
        include_selected=True,
    ):
        # Find rank in full results first
        full_rank = (indices[0] == selected_token_id).nonzero()[0].item()

        # Then take top k for display
        top_k_similarities = similarities[0][:k]
        top_k_indices = indices[0][:k]

        # Convert to token strings and format with ranks
        token_strings = self.transformer.tokenizer.convert_ids_to_tokens(top_k_indices)
        result_strings = [
            ("▶" if idx == selected_token_id else "") + f"{i + 1}. {token} ({sim:.3f})"
            for i, (token, sim, idx) in enumerate(
                zip(token_strings, top_k_similarities, top_k_indices)
            )
        ]

        # If selected token not in top k and we want to include it
        if include_selected and selected_token_id not in top_k_indices:
            token = self.transformer.tokenizer.convert_ids_to_tokens([selected_token_id])[0]
            full_similarity = similarities[0][full_rank]
            result_strings.append(f"▶{full_rank + 1}. {token} ({full_similarity:.3f})")

        return result_strings

    def get_similarities_df(self, token_metrics, selected_token):
        if selected_token is None:
            selected_token = 1

        token_str = token_metrics["token_strings"][selected_token]
        token_id = token_metrics["token_ids"][selected_token]

        # Format similarities for each layer
        similarities = {}
        for layer in range(self.transformer.config.num_hidden_layers):
            D, I = token_metrics["faiss_results"][selected_token][layer]
            similarities[f"layer {layer}"] = self.get_closest_vocabulary_tokens(
                D, I, token_id, k=self.k
            )

        # Pad all columns to same length
        max_length = max(len(layer_results) for layer_results in similarities.values())
        for layer in similarities:
            current_length = len(similarities[layer])
            if current_length < max_length:
                similarities[layer].extend([""] * (max_length - current_length))
        similarity_df = pd.DataFrame(
            similarities,
            columns=[
                f"layer {layer}" for layer in range(self.transformer.config.num_hidden_layers)
            ],
        )
        return token_str, similarity_df

    def get_all_token_metrics(self, text, use_positional=False):
        """
        Returns:
        token_strings:
            A list of string representations of the input tokens.
            For example, token_strings[3] is the string corresponding to the fourth token.

        token_ids:
            A tensor of token ids indexed by token position in the sequence.
            This tensor was used as the input to the model.
            For example, token_ids[3] is the integer ID of the fourth token.

        faiss_results:
            A list of FAISS search results for each token in the input sequence.
            For each token position `token_seq`, `faiss_results[token_seq]` is a list
            of length `num_layers`, where each element is a tuple:

            faiss_results[token_seq][layer_idx] == (similarities, indices)

            where:
            - `similarities` is a NumPy array of shape (1, vocab_size) containing
            the cosine similarities between the normalized post-residual-add encoding
            and all vocabulary token embeddings.
            - `indices` is a NumPy array of shape (1, vocab_size) with corresponding token IDs in
            similarity order.
        """
        # Setup
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        outputs = self.transformer.model(**inputs)
        num_layers = self.transformer.config.num_hidden_layers
        index = self.indexes["raw" if not use_positional else "positional"]

        # Track metrics for all tokens
        token_strings = []
        faiss_results = []
        layer_post_attention_values = {}

        # Calculate metrics for each token position
        for pos in range(len(input_ids)):
            # Get token info
            tok_id = input_ids[pos]
            tok_str = self.transformer.tokenizer.convert_ids_to_tokens(tok_id.unsqueeze(0))[0]
            token_strings.append(tok_str)

            token_faiss_results = []

            # Calculate metrics at each layer
            for layer_idx in range(num_layers):
                # Get layer output for this token
                if layer_idx not in layer_post_attention_values:
                    layer_post_attention_values[layer_idx] = self.compute_post_attention_values(
                        layer_idx, outputs
                    )
                post_attention = layer_post_attention_values[layer_idx]

                concatenated_heads = post_attention[pos].reshape(-1)
                layer_context_weights = self.transformer.get_layer_context_weights(layer_idx)
                context_output = concatenated_heads @ layer_context_weights
                layer_input = outputs.hidden_states[layer_idx][0, pos]
                layer_output = context_output + layer_input
                layer_output = F.normalize(layer_output.unsqueeze(0), p=2, dim=-1).squeeze(0)
                layer_output = layer_output.detach()

                # Use FAISS to get similarities and rankings
                similarities, indexes = index.search(
                    layer_output.unsqueeze(0), self.transformer.config.vocab_size
                )
                token_faiss_results.append((similarities, indexes))  # Store raw results
            faiss_results.append(token_faiss_results)
        return {
            "token_strings": token_strings,
            "token_ids": input_ids,
            "faiss_results": faiss_results,
        }

    @staticmethod
    def get_rankings_for_token(pos, faiss_results, token_ids):
        """
        For a given token position, compute a list of ranks across all layers.
        Each rank is the index at which the original token ID appears in the FAISS result.
        """
        tok_id = token_ids[pos].item()
        rankings = []
        for _, indexes in faiss_results[pos]:
            rank = (indexes[0] == tok_id).nonzero()[0].item()
            rankings.append(rank)
        return rankings

    @staticmethod
    def get_similarities_for_token(pos, faiss_results, token_ids):
        """
        For a given token position, compute a list of cosine similarities to the
        original token embedding across all layers.

        Each similarity is taken from the FAISS result for that token and layer.
        """

        tok_id = token_ids[pos].item()
        similarities = []
        for D, I in faiss_results[pos]:
            rank = (I[0] == tok_id).nonzero()[0].item()
            similarity = D[0][rank]
            similarities.append(similarity)
        return similarities
