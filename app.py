from transformers import AutoTokenizer, AutoModel, AutoConfig
import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F

# %load_ext gradio

debug = False


class RobertaTransformer:
    def __init__(self, model_name="roberta-base"):
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        self.positional_embedding_offset = 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.config = AutoConfig.from_pretrained(model_name)
        self.head_size = self.config.hidden_size // self.config.num_attention_heads
        self.all_word_embeddings = self.model.embeddings.word_embeddings.weight.data
        self.positional_embeddings = self.model.embeddings.position_embeddings.weight
        self.layer_norm = self.model.embeddings.LayerNorm
        self.normalized_word_embeddings_without_position = self.layer_norm(
            self.all_word_embeddings
        )
        self.all_word_embeddings_with_position = None
        self.normalized_word_embeddings_with_position = None

    def create_positional_embeddings(self, selected_token):
        self.all_word_embeddings_with_position = (
            self.all_word_embeddings
            + self.positional_embeddings[
                selected_token + self.positional_embedding_offset
            ]
        )
        self.normalized_word_embeddings_with_position = self.layer_norm(
            self.all_word_embeddings_with_position
        )

        if debug:
            print(selected_token)
            print(f"{self.all_word_embeddings[selected_token][:5]=}")
            print(
                f"{self.positional_embeddings[selected_token + self.positional_embedding_offset][:5]=}"
            )
            print(f"{self.all_word_embeddings_with_position[selected_token][:5]=}")

    def get_all_encodings_from_weight(self, embeddings, kqv_weights):
        # compute kqv encodings for all words
        x = torch.einsum("nd,df->nf", embeddings, kqv_weights)
        # reshape into individual head kqv encodings
        x_heads = x.view(x.size(0), self.config.num_attention_heads, self.head_size)
        return x_heads

    def get_head_encodings_from_weight(self, embeddings, kqv_weights, selected_head):
        X_heads = self.get_all_encodings_from_weight(embeddings, kqv_weights)
        X_selected_head = X_heads[:, selected_head, :]
        return X_selected_head

    def get_selected_layer_value_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.value.weight

    def get_attention_weights_all_heads(self, outputs, selected_layer, selected_token):
        attentions_l_t = outputs.attentions[selected_layer][:, :, selected_token, :]
        if debug:
            print(f"{attentions_l_t[:, 0, :]=}")
        if debug:
            print(f"{torch.sum(attentions_l_t[:, 0, :])=}")
        return attentions_l_t

    def get_all_layer_0_value_encodings(self, use_positional=True):
        layer_0_value_weights = self.get_selected_layer_value_weights(0)

        if use_positional:
            # V_single_position has all tokens' value encodings using _selected_token's position_ for comparison to attention-weighted output encodings

            V_all_layer_0 = self.get_all_encodings_from_weight(
                self.normalized_word_embeddings_with_position,
                layer_0_value_weights,
            )
        else:
            V_all_layer_0 = self.get_all_encodings_from_weight(
                self.normalized_word_embeddings_without_position,
                layer_0_value_weights,
            )
        return {"V_all_layer_0": V_all_layer_0}

    def tokens_for_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_list = [[token] for token in tokens]
        return token_list

    def get_intermediate_output(self, selected_layer, selected_token, outputs):
        selected_layer = int(selected_layer)
        # get weights for selected_layer
        selected_layer_value_weights = self.get_selected_layer_value_weights(
            selected_layer
        )
        if debug:
            print(f"{selected_layer_value_weights.shape=}")
        # if debug: print(f"{selected_layer_value_weights[selected_token]=}")

        # <batch_size, seq_len, hidden_size>, the input for selected_layer
        layer_sequence_inputs = outputs.hidden_states[
            selected_layer
        ]  # actual input embeddings (not all-same-position-encoded)
        if debug:
            print(f"{layer_sequence_inputs.shape=}")  # [1, 8, 768]
        if debug:
            print(f"{layer_sequence_inputs[:, selected_token, :5]=}")

        # mutiply input encodings by value weights to get value encodings
        # so values for all tokens, all heads
        value_encodings = torch.einsum(
            "bsd,dv->bsv", layer_sequence_inputs, selected_layer_value_weights
        )
        if debug:
            print(
                f"{value_encodings.shape=}"
            )  # (1,8,768)<batch_size, seq_len, hidden_size>

        # get all attention weights for all heads, single layer, single token
        attentions_l_t = self.get_attention_weights_all_heads(
            outputs, selected_layer, selected_token
        )
        if debug:
            print(f"{attentions_l_t.shape=}")  # (1,12,8)

        # split value encoding into per-head encodings
        encodings_for_single_token_all_heads = value_encodings.view(
            value_encodings.size(0),
            value_encodings.size(1),
            self.config.num_attention_heads,
            self.head_size,
        )
        if debug:
            print(f"{encodings_for_single_token_all_heads.shape=}")  # (([1, 8, 12, 64]))

        post_attn_layer_value_encodings = torch.einsum(
            "bhs,bshv->bhv", attentions_l_t, encodings_for_single_token_all_heads
        )
        if debug:
            print(f"{post_attn_layer_value_encodings.shape=}")  # (1,12,64)

        return post_attn_layer_value_encodings


k = 15


def get_distances_euclidean(post_attention_encoding, comparison_encodings, p=2.0):
    return F.pairwise_distance(post_attention_encoding, comparison_encodings, p=p)


def get_distances_cosine(post_attention_encoding, comparison_encodings):
    return 1 - F.cosine_similarity(
        post_attention_encoding, comparison_encodings, dim=-1
    )


def get_closest(
    k,
    transformer,
    input_ids,
    post_attention_encoding,
    comparison_encodings,
    type="euclidean",
):
    if type == "Euclidean":
        encoding_distances = get_distances_euclidean(
            post_attention_encoding, comparison_encodings
        )
    else:
        encoding_distances = get_distances_cosine(
            post_attention_encoding, comparison_encodings
        )
    distances, encodings = torch.topk(encoding_distances, k, largest=False)
    if debug:
        print(f"{post_attention_encoding.shape=}")
    if debug:
        print(f"{comparison_encodings.shape=}")
    if debug:
        print(f"{encodings.shape=}")
    if input_ids is not None:  # the indexes are into input_ids instead of encodings
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist()
        )
    else:
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(encodings.tolist())
    distances = ["{0:.3f}".format(distance.item()) for distance in distances]

    return list(zip(encodings_str, distances))


class App:
    def __init__(self):
        self.transformer = RobertaTransformer()

    def closest_to_all_values(
        self,
        text="Time flies like an arrow.",
        selected_token=2,
        selected_layer=0,
        distance_type="Euclidean",
        use_positional=True,
    ):
        if debug:
            print(f"{selected_token=}")
        if not selected_token:
            selected_token = 1
        self.transformer.create_positional_embeddings(selected_token)

        inputs = self.transformer.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]

        token_str = f"{self.transformer.tokenizer.convert_ids_to_tokens(input_ids[0, selected_token].item())} ({input_ids[0, selected_token]})"

        outputs = self.transformer.model(**inputs)

        layer_0_value_encodings = self.transformer.get_all_layer_0_value_encodings(use_positional=use_positional)

        post_attention_encoding = self.transformer.get_intermediate_output(
            selected_layer, selected_token, outputs
        )
        closest_df = pd.DataFrame()
        for head in range(self.transformer.config.num_attention_heads):
            closest_for_head = get_closest(
                k,
                self.transformer,
                None,
                post_attention_encoding[:, head, :].squeeze(0),
                layer_0_value_encodings["V_all_layer_0"][:, head, :],
                type=distance_type,
            )
            closest_for_head = [f"{token} ({dist})" for token, dist in closest_for_head]
            closest_for_head_df = pd.DataFrame(closest_for_head)
            closest_for_head_df.columns = [f"head {head}"]
            closest_df = pd.concat([closest_df, closest_for_head_df], axis=1)
        return (token_str, closest_df)

    def launch(self):
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            position = intro_markdown.find("# Token")
            if position != -1:
                intro_markdown = intro_markdown[position:]

        custom_css = """.combined span { font-size: 9px !important; padding: 2px }
                    .combined div { min-height: 10px !important}
        """
        with gr.Blocks(css=custom_css) as demo:
            initial_text = "Time flies like an arrow. Fruit flies like a banana."
            with gr.Row():
                text = gr.Text(
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
                    headers=[[f"head {head}"] for head in range(12)],
                )
            display_values = (
                self.closest_to_all_values,
                [text, tokens, selected_layer, distance_type, use_positional],
                [selected_token_str, combined_dataframe],
            )
            # display_tokens = (self.transformer.tokens_for_text, [text], [tokens])
            tokens.click(*display_values)
            selected_layer.change(*display_values)
            distance_type.change(*display_values)
            use_positional.change(*display_values)

            display_tokens = (self.transformer.tokens_for_text, [text], [tokens])
            show_tokens.click(*display_tokens)
            text.submit(*display_tokens)

            gr.Markdown(intro_markdown)
        demo.launch()


App().launch()
