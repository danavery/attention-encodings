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
    # keep

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

    #keep
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

    # keep
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

        for index in range(k, len(all_indices)):
            if all_indices[index] == selected_token_id:
                result_strings.append(
                    f"▶{index+1}. {token_strings[index]} ({all_distances[index]:.3f})"
                )
            elif show_other_tokens and all_indices[index] in other_token_ids:
                result_strings.append(
                    f"•{index+1}. {token_strings[index]} ({all_distances[index]:.3f})"
                )
        return result_strings

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

        # get the embeddings for the original vocabulary tokens
        # (optionally including positional embeddings)
        comparison_encodings = self.transformer.get_raw_vocab_embeddings(
            selected_token_idx=token_position,
            apply_positional_embeddings=use_positional,
        )

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

        return token_str, residual_output_distances

    def get_residual_distances_df(
        self, text, selected_token, distance_type, use_positional
    ):
        token_str, distances = self.get_residual_distances(
            text, selected_token, distance_type, use_positional
        )
        # Find max length across all layers
        max_length = max(len(layer_results) for layer_results in distances.values())

        # Pad each layer's results to max_length
        for layer in distances:
            current_length = len(distances[layer])
            if current_length < max_length:
                distances[layer].extend([""] * (max_length - current_length))

        return token_str, pd.DataFrame(
            distances,
            columns=[
                f"layer {layer}"
                for layer in range(self.transformer.config.num_hidden_layers)
            ],
        )

    def get_token_residual_journey(
        self, text, token_position, layer, distance_type="Cosine"
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

        # Start with initial embedding and project it to head size
        initial_embedding = outputs.hidden_states[0].squeeze(0)[token_position].detach()
        embeddings = [initial_embedding]  # Start list with initial embedding

        # Then get post-attention encodings for each layer
        num_layers = self.transformer.config.num_hidden_layers
        for layer_idx in range(num_layers):
            # Get post-attention encodings and concatenate heads for the selected position
            post_attention = self.compute_post_attention_values(layer_idx, outputs)
            concatenated_heads = post_attention[token_position].reshape(-1)

            # Apply context weights to the concatenated heads
            layer_context_weights = self.transformer.get_layer_context_weights(
                layer_idx
            )
            context_output = concatenated_heads @ layer_context_weights

            # Add the context output and the input embedding to get the residual output
            layer_input = outputs.hidden_states[layer_idx][0, token_position]
            layer_output = context_output + layer_input

            embeddings.append(layer_output.detach())

        # Get all tokens at current layer
        post_attention_encodings = self.compute_post_attention_values(layer, outputs)
        concatenated_heads = post_attention_encodings.reshape(
            post_attention_encodings.size(0), -1
        )  # reshape for all tokens
        layer_context_weights = self.transformer.get_layer_context_weights(layer)
        context_output = concatenated_heads @ layer_context_weights
        layer_input = outputs.hidden_states[layer][0]  # all tokens
        all_tokens_current_layer = (context_output + layer_input).detach()

        # Print distances from initial to each layer output
        # print(f"\nDistances from initial embedding for token {token_str}:")
        # for i, emb in enumerate(embeddings[1:], 1):  # Skip first since it's initial
        #     dist = 1 - F.cosine_similarity(embeddings[0], emb, dim=0)
        #     print(f"Layer {i-1}: {dist.item():.3f}")

        return {
            "embeddings": embeddings,
            "all_current": all_tokens_current_layer,
            "token_info": {
                "id": token_id.item(),
                "string": token_str,
                "position": token_position,
            },
        }

    def get_all_residual_token_distances(self, text, selected_token, distance_type="Cosine"):
        # Setup
        if selected_token is None:
            selected_token = 1  # Default to first real token (after <s>)

        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        outputs = self.transformer.model(**inputs)
        num_layers = self.transformer.config.num_hidden_layers

        # Track distances for all tokens
        distances_by_token = {}
        token_strings = {}

        # Calculate distances for each token position
        for pos in range(len(input_ids)):
            # Get token info
            tok_id = input_ids[pos]
            tok_str = self.transformer.tokenizer.convert_ids_to_tokens(tok_id.unsqueeze(0))[0]
            token_strings[pos] = tok_str

            # Get initial embedding for this token
            initial = outputs.hidden_states[0].squeeze(0)[pos].detach()
            token_distances = []

            # Calculate distance at each layer
            for layer_idx in range(num_layers):
                # Get layer output for this token
                post_attention = self.compute_post_attention_values(layer_idx, outputs)
                concatenated_heads = post_attention[pos].reshape(-1)
                layer_context_weights = self.transformer.get_layer_context_weights(layer_idx)
                context_output = concatenated_heads @ layer_context_weights
                layer_input = outputs.hidden_states[layer_idx][0, pos]
                layer_output = context_output + layer_input

                # Calculate distance from initial embedding
                dist = self.get_distances(initial, layer_output.unsqueeze(0), distance_type).item()
                token_distances.append(dist)

            distances_by_token[pos] = token_distances

        return {
            "all_distances": distances_by_token,
            "token_strings": token_strings,
            "selected_token": selected_token,
            "distance_type": distance_type
        }

    def get_all_token_rankings(self, text, selected_token, distance_type="Cosine"):
        # Setup
        if selected_token is None:
            selected_token = 1  # Default to first real token (after <s>)
        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        outputs = self.transformer.model(**inputs)
        num_layers = self.transformer.config.num_hidden_layers

        # Track rankings for all tokens
        rankings_by_token = {}
        token_strings = {}

        # Calculate rankings for each token position
        for pos in range(len(input_ids)):
            # Get token info
            tok_id = input_ids[pos]
            tok_str = self.transformer.tokenizer.convert_ids_to_tokens(tok_id.unsqueeze(0))[0]
            token_strings[pos] = tok_str

            # Get initial embedding for this token
            initial = outputs.hidden_states[0].squeeze(0)[pos].detach()
            token_rankings = []

            # Calculate ranking at each layer
            for layer_idx in range(num_layers):
                # Get layer output for this token
                post_attention = self.compute_post_attention_values(layer_idx, outputs)
                concatenated_heads = post_attention[pos].reshape(-1)
                layer_context_weights = self.transformer.get_layer_context_weights(layer_idx)
                context_output = concatenated_heads @ layer_context_weights
                layer_input = outputs.hidden_states[layer_idx][0, pos]
                layer_output = context_output + layer_input

                # Get distances to all vocab embeddings
                vocab_embeddings = self.transformer.get_raw_vocab_embeddings(
                    selected_token_idx=pos,
                    apply_positional_embeddings=False  # Since we're looking for original token
                )
                distances = self.get_distances(layer_output.unsqueeze(0), vocab_embeddings, distance_type)

                # Sort distances and find rank of original token
                sorted_indices = torch.argsort(distances)
                rank = (sorted_indices == tok_id).nonzero().item()
                token_rankings.append(rank)

            rankings_by_token[pos] = token_rankings

        return {
            "all_rankings": rankings_by_token,
            "token_strings": token_strings,
            "selected_token": selected_token,
            "distance_type": distance_type
        }