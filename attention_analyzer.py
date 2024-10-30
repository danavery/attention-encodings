import logging
import pandas as pd
import torch
import torch.nn.functional as F


class AttentionAnalyzer:
    def __init__(self, transformer, k=15):
        self.transformer = transformer
        self.k = k

    def get_tokens_for_text(self, text):
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        tokens = self.transformer.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_list = [[token] for token in tokens]
        return token_list

    def get_distances(
        self, post_attention_encoding, comparison_encodings, distance_type="Euclidean"
    ):
        """
        Compute the distances between a post-attention encoding and a set of comparison encodings.
        """
        if distance_type == "Euclidean":
            return F.pairwise_distance(post_attention_encoding, comparison_encodings)
        elif distance_type == "Cosine":
            return 1 - F.cosine_similarity(
                post_attention_encoding, comparison_encodings, dim=-1
            )
        else:
            raise ValueError(f"Invalid distance type: {distance_type}")

    def get_all_layer_0_value_encodings(self, apply_positional_embeddings=True):
        layer_0_value_weights = self.transformer.get_layer_value_weights(0)

        # get normalized vocab embeddings and project to value space
        return self.project_to_value_by_head(
            self.transformer.get_normalized_vocab_embeddings(
                apply_positional_embeddings=apply_positional_embeddings
            ),
            layer_0_value_weights,
        )

    def project_to_value_by_head(self, embeddings, value_weights):
        x = embeddings @ value_weights
        # reshape into individual head value encodings
        x_heads = x.view(
            x.size(0),
            self.transformer.config.num_attention_heads,
            self.transformer.head_size,
        )
        return x_heads

    def compute_post_attention_values(self, selected_layer, outputs):
        """
        Compute the post-attention encodings for all tokens at a specified layer.
        Outputs a tensor of shape <seq_len, num_heads, embedding_size>.
        """
        selected_layer = int(selected_layer)
        # get weights for selected_layer
        selected_layer_value_weights = self.transformer.get_layer_value_weights(
            selected_layer
        )

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
        post_attn_layer_encodings = layer_attention_weights.bmm(
            encodings_per_head_permuted
        )

        # end up with <seq_len, num_heads, hidden_size/num_heads>
        post_attn_layer_encodings = post_attn_layer_encodings.permute(1, 0, 2)

        return post_attn_layer_encodings

    def get_closest(
        self,
        k,
        transformer,
        input_ids,
        post_attention_encoding,
        comparison_encodings,
        distance_type="Euclidean",
    ):
        encoding_distances = self.get_distances(
            post_attention_encoding, comparison_encodings, distance_type
        )
        distances, encodings = torch.topk(encoding_distances, k, largest=False)

        if input_ids is not None:  # the indexes are into input_ids instead of encodings
            encodings_str = transformer.tokenizer.convert_ids_to_tokens(
                input_ids[encodings].tolist()
            )
        else:
            encodings_str = transformer.tokenizer.convert_ids_to_tokens(
                encodings.tolist()
            )
        distances = ["{0:.3f}".format(distance.item()) for distance in distances]

        return list(zip(encodings_str, distances))

    def closest_to_all_values(
        self,
        text="Time flies like an arrow.",
        selected_token=1,
        selected_layer=0,
        distance_type="Euclidean",
        apply_positional_embeddings=True,
    ):
        """
        Computes and returns the closest token encodings to a specified token's
        post-attention encoding across all attention heads at a specified layer.

        The closeness between encodings is measured in the embedding space using
        either Euclidean or Cosine distance.
        """
        # if you click on something that's not a token first, Gradio supplies None as selected_token
        # which is suboptimal
        if selected_token is None:
            selected_token = 1
        logging.debug(f"{selected_token=}")
        self.transformer.adjust_vocab_to_token_position(selected_token)

        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)  # <seq_len>, don't need batch dim

        token_display_string = f"{self.transformer.tokenizer.convert_ids_to_tokens(input_ids[selected_token].item())} ({input_ids[selected_token]})"

        # get all value encodings for all tokens at layer 0
        layer_0_value_encodings = self.get_all_layer_0_value_encodings(
            apply_positional_embeddings=apply_positional_embeddings
        )

        # run the model to get the attention weights and post-attention encodings
        outputs = self.transformer.model(**inputs)

        # get the post-attention encodings for all tokens at the selected layer
        post_attention_encodings = self.compute_post_attention_values(
            selected_layer, outputs
        )

        closest_df, closest_outputs_df = self.get_result_dataframes(
            selected_token,
            distance_type,
            input_ids,
            layer_0_value_encodings,
            post_attention_encodings,
        )
        return (token_display_string, closest_df, closest_outputs_df)

    def get_result_dataframes(
        self,
        selected_token,
        distance_type,
        input_ids,
        layer_0_value_encodings,
        post_attention_encodings,
    ):
        """
        Creates dataframes comparing token encodings across attention heads:
        1. Closest vocabulary tokens to post-attention encoding
        2. Closest sequence tokens to post-attention encoding
        """
        num_heads = self.transformer.config.num_attention_heads
        closest_df = []
        closest_outputs_df = []

        for head in range(num_heads):
            # Compare to level 0 vocabulary value encodings
            vocab_comparisons = self.get_closest(
                self.k,
                self.transformer,
                None,
                post_attention_encodings[selected_token, head, :],
                layer_0_value_encodings[:, head, :],
                distance_type=distance_type,
            )
            closest_df.append(
                pd.DataFrame(
                    [f"{token} ({dist})" for token, dist in vocab_comparisons],
                    columns=[f"head {head}"],
                )
            )

            # Compare to other sequence token encodings at
            sequence_comparisons = self.get_closest(
                len(input_ids),
                self.transformer,
                input_ids,
                post_attention_encodings[selected_token, head, :],
                post_attention_encodings[:, head, :],
                distance_type=distance_type,
            )
            closest_outputs_df.append(
                pd.DataFrame(
                    [f"{token} ({dist})" for token, dist in sequence_comparisons],
                    columns=[f"head {head}"],
                )
            )

        return (pd.concat(closest_df, axis=1), pd.concat(closest_outputs_df, axis=1))