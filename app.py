import gradio as gr
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

debug = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RobertaTransformer:
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
        self.normalized_vocab_embeddings = self.layer_norm(
            self.vocab_base_embeddings
        )

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

        if debug:
            print(selected_token_idx)
            print(f"{self.vocab_base_embeddings[selected_token_idx][:5]=}")
            print(
                f"{self.positional_embeddings[selected_token_idx + self.position_embedding_start_idx][:5]=}"
            )
            print(f"{self.position_adjusted_vocab[selected_token_idx][:5]=}")

    def project_to_value_by_head(self, embeddings, value_weights):
        # compute value encodings for all words
        x = torch.einsum("nd,df->nf", embeddings, value_weights)
        # reshape into individual head value encodings
        x_heads = x.view(x.size(0), self.config.num_attention_heads, self.head_size)
        return x_heads

    def get_layer_value_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.value.weight

    def get_layer_attention_weights(self, outputs, selected_layer):
        attentions = outputs.attentions[selected_layer]
        return attentions

    def get_all_layer_0_value_encodings(self, apply_positional_embeddings=True):
        layer_0_value_weights = self.get_layer_value_weights(0)

        if apply_positional_embeddings:
            layer0_value_encodings = self.project_to_value_by_head(
                self.normalized_position_adjusted_vocab,
                layer_0_value_weights,
            )
        else:
            layer0_value_encodings = self.project_to_value_by_head(
                self.normalized_vocab_embeddings,
                layer_0_value_weights,
            )
        return layer0_value_encodings

    def tokens_for_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_list = [[token] for token in tokens]
        return token_list

    def compute_post_attention_values(self, selected_layer, outputs):
        """
        Compute the post-attention encodings for all tokens at a specified layer.
        Outputs a tensor of shape <seq_len, num_heads, embedding_size>.
        """
        selected_layer = int(selected_layer)
        # get weights for selected_layer
        selected_layer_value_weights = self.get_layer_value_weights(
            selected_layer
        )

        # actual input embeddings (not all-same-position-encoded)
        # <batch_size, seq_len, hidden_size>, the input for selected_layer
        layer_sequence_inputs = outputs.hidden_states[selected_layer]

        # multiply input encodings by value weights to get value encodings
        # so values for all tokens, all heads
        # <batch_size, seq_len, hidden_size/num_heads>
        value_encodings = torch.einsum(
            "bsd,dv->bsv", layer_sequence_inputs, selected_layer_value_weights
        )

        # get all attention weights for all heads, single layer, single token
        # <batch_size, num_heads, seq_len, seq_len>
        layer_attention_weights = self.get_layer_attention_weights(outputs, selected_layer)

        # split value encoding into per-head encodings
        # <batch_size, seq_len, num_heads, hidden_size/num_heads>
        encodings_per_head = value_encodings.view(
            value_encodings.size(0),
            value_encodings.size(1),
            self.config.num_attention_heads,
            self.head_size,
        )

        # <batch_size, seq_len, num_heads, hidden_size/num_heads>
        post_attn_layer_encodings = torch.einsum(
            "bhij,bjhv->bihv", layer_attention_weights, encodings_per_head
        )

        return post_attn_layer_encodings.squeeze(0)


def get_distances(
    post_attention_encoding, comparison_encodings, distance_type="Euclidean"
):
    """
    Compute the distances between a post-attention encoding and a set of comparison encodings.
    """
    if distance_type == "Euclidean":
        return F.pairwise_distance(post_attention_encoding, comparison_encodings)
    else:
        return 1 - F.cosine_similarity(
            post_attention_encoding, comparison_encodings, dim=-1
        )


def get_closest(
    k,
    transformer,
    input_ids,
    post_attention_encoding,
    comparison_encodings,
    distance_type="Euclidean",
):
    encoding_distances = get_distances(
        post_attention_encoding, comparison_encodings, distance_type
    )
    distances, encodings = torch.topk(encoding_distances, k, largest=False)

    if input_ids is not None:  # the indexes are into input_ids instead of encodings
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(
            input_ids[0, encodings].tolist()
        )
    else:
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(encodings.tolist())
    distances = ["{0:.3f}".format(distance.item()) for distance in distances]

    return list(zip(encodings_str, distances))


class App:
    def __init__(self):
        self.transformer = RobertaTransformer()
        self.k = 15

    def update_tokens(self, text):
        tokens = self.transformer.tokens_for_text(text)
        return gr.update(samples=tokens, components=["text"])

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
        if debug:
            print(f"{selected_token=}")
        self.transformer.adjust_vocab_to_token_position(selected_token)

        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        decoded_tokens = f"{self.transformer.tokenizer.convert_ids_to_tokens(input_ids[0, selected_token].item())} ({input_ids[0, selected_token]})"

        # get all value encodings for all tokens at layer 0
        layer_0_value_encodings = self.transformer.get_all_layer_0_value_encodings(
            apply_positional_embeddings=apply_positional_embeddings
        )

        # run the model to get the attention weights and post-attention encodings
        outputs = self.transformer.model(**inputs)


        # get the post-attention encodings for all tokens at the selected layer
        post_attention_encodings = self.transformer.compute_post_attention_values(
            selected_layer, outputs
        )
        closest_df = []
        closest_outputs_df = []
        for head in range(self.transformer.config.num_attention_heads):
            closest_for_head = get_closest(
                self.k,
                self.transformer,
                None,
                post_attention_encodings[selected_token, head, :],
                layer_0_value_encodings[:, head, :],
                distance_type=distance_type,
            )
            closest_for_head = [f"{token} ({dist})" for token, dist in closest_for_head]
            closest_df.append(pd.DataFrame(closest_for_head, columns=[f"head {head}"]))

            closest_outputs_for_head = get_closest(
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

    def get_intro_markdown(self):
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            position = intro_markdown.find("# Token")
            if position != -1:
                intro_markdown = intro_markdown[position:]
        return intro_markdown

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
                    samples=self.transformer.tokens_for_text(initial_text),
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
                fn=self.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            selected_layer.change(
                fn=self.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            distance_type.change(
                fn=self.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            use_positional.change(
                fn=self.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )
            demo.load(
                fn=self.closest_to_all_values,
                inputs=[text, tokens, selected_layer, distance_type, use_positional],
                outputs=[selected_token_str, combined_dataframe, intertoken_dataframe],
            )

            show_tokens.click(self.update_tokens, inputs=[text], outputs=[tokens])
            text.submit(self.update_tokens, inputs=[text], outputs=[tokens])

            gr.Markdown(intro_markdown)
            try:
                combined_dataframe.change(_js=custom_js)
            except:
                combined_dataframe.change(js=custom_js)
        demo.launch(server_name="0.0.0.0")


App().launch()
