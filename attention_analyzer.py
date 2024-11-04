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
        self, post_attention_encoding, comparison_encodings, distance_type="Cosine"
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

    def get_layer_value_encodings(self, layer=0, apply_positional_embeddings=True):
        layer_value_weights = self.transformer.get_layer_value_weights(layer)

        # get normalized vocab embeddings and project to value space
        return self.project_to_value_by_head(
            self.transformer.get_normalized_vocab_embeddings(
                apply_positional_embeddings=apply_positional_embeddings
            ),
            layer_value_weights,
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

    def _find_closest_encodings(
        self,
        reference_encoding,  # the post-attention encoding we're comparing from
        comparison_encodings,  # either layer_0 or post-attention encodings
        distance_type="Cosine",
    ):
        """
        Returns sorted distances and indices for closest encodings
        """
        distances = self.get_distances(
            reference_encoding, comparison_encodings, distance_type
        )
        all_distances, all_indices = torch.sort(distances, descending=False)
        return all_distances, all_indices

    def get_closest_vocabulary_tokens(
        self,
        reference_encoding,
        base_encodings,
        selected_token_id,
        other_token_ids=[],
        show_other_tokens=False,
        k=15,
        distance_type="Cosine",
    ):
        """
        Find closest vocabulary tokens to the reference encoding.
        Returns list of strings formatted with rank numbers and distances.

        Args:
            reference_encoding: Post-attention encoding for selected token or residual output
            base_encodings: Vocabulary encodings to compare to (either layer 0 value encodings or raw vocab embeddings)
            selected_token_id: Token ID of the selected token to find its rank
            other_token_ids: Token IDs of other tokens to always include in the results
            show_other_tokens: Whether to show other tokens in the results at all. Currently false because it hasn't proven useful yet.
            k: Number of closest tokens to return (default 15)
            distance_type: Distance metric to use (default "Cosine")

        Returns:
            List of strings formatted as "rank. token (distance)"
            If selected token not in top k, appends it with its actual rank
        """
        # Get all distances and sorted indices
        all_distances, all_indices = self._find_closest_encodings(
            reference_encoding, base_encodings, distance_type
        )

        # Find rank of selected token
        selected_token_rank = (all_indices == selected_token_id).nonzero()[0].item() + 1

        # Take top k
        distances = all_distances[:k]
        indices = all_indices[:k]

        # Convert to token strings and format with ranks
        token_strings = self.transformer.tokenizer.convert_ids_to_tokens(
            all_indices.tolist()
        )
        result_strings = [
            ("▶" if indices[i] == selected_token_id else "")
            + f"{i+1}. {token} ({dist:.3f})"
            for i, (token, dist) in enumerate(zip(token_strings, distances))
        ]
        # Add original token if not in top k
        for index in range(k, len(all_indices)):
            if all_indices[index] == selected_token_id:
                result_strings.append(
                    f"▶{index+1}. {token_strings[index]} ({all_distances[index]:.3f})"
                )
            elif all_indices[index] in other_token_ids:
                result_strings.append(
                    f"•{index+1}. {token_strings[index]} ({all_distances[index]:.3f})"
                )
        return result_strings

    def get_closest_sequence_tokens(
        self,
        reference_encoding,
        post_attention_encodings,
        input_ids,  # needed to convert positions to token IDs
        selected_position,  # position in sequence we're comparing from
        distance_type="Cosine",
    ):
        """
         Find closest sequence tokens to the reference encoding.
         Returns list of strings formatted with rank numbers and distances.

         Args:
        reference_encoding: Post-attention encoding for selected token
        post_attention_encodings: Encodings for all tokens in sequence
        input_ids: Tensor mapping positions to token IDs
        selected_position: Position in sequence of selected token
        distance_type: Distance metric to use (default "Cosine")

         Returns:
             List of strings formatted as "rank. token (distance)" for all sequence tokens
        """
        # Get all distances and sorted indices (these are sequence positions)
        all_distances, position_indices = self._find_closest_encodings(
            reference_encoding, post_attention_encodings, distance_type
        )

        # Convert positions to token IDs then to strings
        token_ids = input_ids[position_indices]
        token_strings = self.transformer.tokenizer.convert_ids_to_tokens(
            token_ids.tolist()
        )

        # Format all results (we want all sequence tokens)
        result_strings = [
            f"{i+1}. {token} ({dist:.3f})"
            for i, (token, dist) in enumerate(zip(token_strings, all_distances))
        ]

        return result_strings

    def closest_to_all_values(
        self,
        text="Time flies like an arrow.",
        selected_token=1,
        selected_layer=0,
        distance_type="Cosine",
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
        layer_0_value_encodings = self.get_layer_value_encodings(
            layer=0, apply_positional_embeddings=apply_positional_embeddings
        )

        # run the model to get the attention weights and post-attention encodings
        outputs = self.transformer.model(**inputs)

        # get the post-attention encodings for all tokens at the selected layer
        post_attention_encodings = self.compute_post_attention_values(
            selected_layer, outputs
        )

        closest_layer_0_df, closest_outputs_df = self.get_result_dataframes(
            selected_token,
            distance_type,
            input_ids,
            layer_0_value_encodings,
            post_attention_encodings,
        )
        return (
            token_display_string,
            closest_layer_0_df,
            closest_outputs_df,
        )

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
        1. Closest level 0 vocabulary value encodings to selected token's post-attention encoding
        2. Closest other sequence tokens to selected token's post-attention encoding
        """
        num_heads = self.transformer.config.num_attention_heads
        vocab_distances_df = []
        sequence_distances_df = []

        for head in range(num_heads):
            # Compare to level 0 vocabulary value encodings
            vocab_comparisons = self.get_closest_vocabulary_tokens(
                post_attention_encodings[selected_token, head, :],
                layer_0_value_encodings[:, head, :],
                input_ids[selected_token],
                k=self.k,
                distance_type=distance_type,
            )
            vocab_distances_df.append(
                pd.DataFrame(
                    vocab_comparisons,
                    columns=[f"head {head}"],
                )
            )

            # Compare to other sequence token encodings in the selected layer's post-attention outputs
            sequence_comparisons = self.get_closest_sequence_tokens(
                post_attention_encodings[selected_token, head, :],
                post_attention_encodings[:, head, :],
                input_ids,
                selected_token,
                distance_type=distance_type,
            )
            sequence_distances_df.append(
                pd.DataFrame(
                    sequence_comparisons,
                    columns=[f"head {head}"],
                )
            )

        return (
            pd.concat(vocab_distances_df, axis=1),
            pd.concat(sequence_distances_df, axis=1),
        )

    def get_token_journey(self, text, token_position, layer, head):
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        outputs = self.transformer.model(**inputs)
        if token_position is None:
            token_position = 1
        token_id = input_ids[token_position]
        token_str = self.transformer.tokenizer.convert_ids_to_tokens(
            token_id.unsqueeze(0)
        )[0]

        # Start with initial embedding and project it to head size
        initial_embedding = outputs.hidden_states[0].squeeze(0)[token_position].detach()
        # Reshape to match head size
        head_size = self.transformer.head_size
        initial_embedding = initial_embedding.view(
            -1, self.transformer.config.num_attention_heads, head_size
        )[0, head]
        embeddings = [initial_embedding]  # Start list with initial embedding

        # Then get post-attention encodings for each layer
        num_layers = self.transformer.config.num_hidden_layers
        for layer_idx in range(num_layers):
            post_attention = self.compute_post_attention_values(layer_idx, outputs)
            embeddings.append(post_attention[token_position, head].detach())

        # Get all tokens at current layer
        current_layer_post_attention = self.compute_post_attention_values(
            layer, outputs
        )
        all_tokens_current_layer = current_layer_post_attention[:, head].detach()

        return {
            "embeddings": embeddings,
            "all_current": all_tokens_current_layer,
            "token_info": {
                "id": token_id.item(),
                "string": token_str,
                "position": token_position,
            },
        }

    def get_residual_distances(
        self, text, token_position, distance_type, use_positional
    ):
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        outputs = self.transformer.model(**inputs)
        if token_position is None:
            token_position = 1
        token_id = input_ids[token_position]
        token_str = self.transformer.tokenizer.convert_ids_to_tokens(
            token_id.unsqueeze(0)
        )[0]

        # we have the outputs of the model, and the string for the token
        # at the selected position. Now we need to get the output of the residual
        # addition of the current layer's input embedding and the concatenated
        # post-attention output so we can compare the output of the residual
        # connection to the initial token embedding

        residual_output_distances = {}
        for layer in range(self.transformer.config.num_hidden_layers):
            # get the post-attention encodings for all tokens at the selected layer
            post_attention_encodings = self.compute_post_attention_values(
                layer, outputs
            )
            # get the post-attention encodings for the selected position
            concatenated_heads = post_attention_encodings[token_position].reshape(-1)

            # get the context weights for the selected layer
            layer_context_weights = self.transformer.get_layer_context_weights(layer)

            # multiply the concatenated heads by the context weights
            context_output = concatenated_heads @ layer_context_weights

            # get the input embedding for the selected position
            layer_input = outputs.hidden_states[layer][0, token_position]

            # add the context output and the input embedding to get the residual output
            layer_output = context_output + layer_input

            # get the embeddings for the original vocabulary tokens
            # (optionally including positional embeddings)
            comparison_encodings = self.transformer.get_raw_vocab_embeddings(
                selected_token_idx=token_position,
                apply_positional_embeddings=use_positional,
            )

            # get the closest vocabulary encodings to the residual output
            closest_tokens = self.get_closest_vocabulary_tokens(
                layer_output,
                comparison_encodings,
                token_id,
                other_token_ids=input_ids,
                k=self.k,
                distance_type=distance_type,
            )
            residual_output_distances[f"layer {layer}"] = closest_tokens

        return residual_output_distances

    def get_residual_distances_df(
        self, text, selected_token, distance_type, use_positional
    ):
        distances = self.get_residual_distances(
            text, selected_token, distance_type, use_positional
        )
        # Find max length across all layers
        max_length = max(len(layer_results) for layer_results in distances.values())

        # Pad each layer's results to max_length
        for layer in distances:
            current_length = len(distances[layer])
            if current_length < max_length:
                distances[layer].extend([""] * (max_length - current_length))

        return pd.DataFrame(
            distances,
            columns=[
                f"layer {layer}"
                for layer in range(self.transformer.config.num_hidden_layers)
            ],
        )
