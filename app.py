import gradio as gr
import logging
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)

class TransformerManager:
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self._load_model(model_name)
        self._setup_embeddings()
        self._init_attributes()

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

    def _init_attributes(self):
        self.position_adjusted_vocab = None
        self.normalized_position_adjusted_vocab = None

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

        if apply_positional_embeddings:
            layer0_value_encodings = self.project_to_value_by_head(
                self.transformer.normalized_position_adjusted_vocab,
                layer_0_value_weights,
            )
        else:
            layer0_value_encodings = self.project_to_value_by_head(
                self.transformer.normalized_vocab_embeddings,
                layer_0_value_weights,
            )
        return layer0_value_encodings

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
                input_ids[0, encodings].tolist()
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
        input_ids = inputs["input_ids"]
        decoded_tokens = f"{self.transformer.tokenizer.convert_ids_to_tokens(input_ids[0, selected_token].item())} ({input_ids[0, selected_token]})"

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
        closest_df = []
        closest_outputs_df = []
        for head in range(self.transformer.config.num_attention_heads):
            closest_for_head = self.get_closest(
                self.k,
                self.transformer,
                None,
                post_attention_encodings[selected_token, head, :],
                layer_0_value_encodings[:, head, :],
                distance_type=distance_type,
            )
            closest_for_head = [f"{token} ({dist})" for token, dist in closest_for_head]
            closest_df.append(pd.DataFrame(closest_for_head, columns=[f"head {head}"]))

            closest_outputs_for_head = self.get_closest(
                len(input_ids[0]),
                self.transformer,
                input_ids,
                post_attention_encodings[selected_token, head, :],
                post_attention_encodings[:, head, :],
                distance_type=distance_type,
            )
            closest_outputs_for_head = [
                f"{token} ({dist})" for token, dist in closest_outputs_for_head
            ]
            closest_outputs_df.append(
                pd.DataFrame(closest_outputs_for_head, columns=[f"head {head}"])
            )

        closest_df = pd.concat(closest_df, axis=1)
        closest_outputs_df = pd.concat(closest_outputs_df, axis=1)
        return (decoded_tokens, closest_df, closest_outputs_df)


class AttentionAnalyzerUI:
    def __init__(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer
        self.k = analyzer.k

    def get_intro_markdown(self):
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            position = intro_markdown.find("# Token")
            if position != -1:
                intro_markdown = intro_markdown[position:]
        return intro_markdown

    def update_token_display(self, text):
        tokens = self.analyzer.get_tokens_for_text(text)
        return gr.update(samples=tokens, components=["text"])

    def launch(self):
        intro_markdown = self.get_intro_markdown()

        custom_css = """
                    .combined span { font-size: 9px !important; padding: 2px }
                    .combined div { min-height: 10px !important}
        """
        custom_js = """document.querySelector('.combined').click();"""

        with gr.Blocks(css=custom_css) as demo:
            initial_text = "Time flies like an arrow. Fruit flies like a banana."
            with gr.Row():
                text = gr.Textbox(
                    placeholder="Text to process",
                    value=initial_text,
                    scale=4,
                    label="Input text",
                )
                selected_layer = gr.Dropdown(
                    choices=range(12),
                    label="Layer number (0-11)",
                    show_label=True,
                    value=0,
                    scale=0,
                )
                distance_type = gr.Dropdown(
                    choices=["Euclidean", "Cosine"],
                    label="Distance type",
                    show_label=True,
                    value="Euclidean",
                    scale=0,
                )
                use_positional = gr.Checkbox(
                    label="Use positional embeddings",
                    show_label=True,
                    value=True,
                    scale=0,
                )
            with gr.Row():
                show_tokens = gr.Button(
                    "Tokenize", elem_classes="get-tokens", size="sm", scale=0
                )
                tokens = gr.Dataset(
                    components=["text"],
                    samples=self.analyzer.get_tokens_for_text(initial_text),
                    type="index",
                    label="Select a token from here to see its closest tokens after attention processing. Click one to see its neighbors below.",
                    scale=5,
                    samples_per_page=1000,
                )
                selected_token_str = gr.Textbox(
                    label="Selected Token", show_label=True, scale=0
                )

            with gr.Row():
                combined_dataframe = gr.DataFrame(
                    label="Nearest initial token value encodings (with distance)",
                    show_label=True,
                    elem_classes="combined",
                    col_count=12,
                    headers=[f"head {head}" for head in range(12)],
                )
            with gr.Row():
                intertoken_dataframe = gr.DataFrame(
                    label="Nearest sequence tokens (with distance)",
                    show_label=True,
                    elem_classes="combined",
                    col_count=12,
                    headers=[f"head {head}" for head in range(12)],
                )

            tokens.click(
                fn=self.analyzer.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            selected_layer.change(
                fn=self.analyzer.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            distance_type.change(
                fn=self.analyzer.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            use_positional.change(
                fn=self.analyzer.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            demo.load(
                fn=self.analyzer.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )

            show_tokens.click(
                self.update_token_display, inputs=[text], outputs=[tokens]
            )
            text.submit(self.update_token_display, inputs=[text], outputs=[tokens])

            gr.Markdown(intro_markdown)
            try:
                combined_dataframe.change(_js=custom_js)
            except:
                combined_dataframe.change(js=custom_js)
        demo.launch(server_name="0.0.0.0")


class App:
    def __init__(self):
        transformer = TransformerManager()
        analyzer = AttentionAnalyzer(transformer)
        self.ui = AttentionAnalyzerUI(analyzer)

    def launch(self):
        self.ui.launch()


if __name__ == "__main__":
    App().launch()
