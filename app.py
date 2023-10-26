# %%
from transformers import AutoTokenizer, AutoModel, AutoConfig
import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F
# %load_ext gradio

debug = False

# %%
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

    def create_positional_embeddings(self, selected_token):
        self.all_word_embeddings_with_position = (
            self.all_word_embeddings
            + self.positional_embeddings[
                selected_token + self.positional_embedding_offset
            ]
        )
        self.layer_norm = self.model.embeddings.LayerNorm
        self.normalized_word_embeddings_with_position = self.layer_norm(
            self.all_word_embeddings_with_position
        )

    def get_all_encodings_from_weight(self, embeddings, kqv_weights):
        # compute kqv encodings for all words
        X = torch.einsum("nd,df->nf", embeddings, kqv_weights)
        # reshape into individual head kqv encodings
        X_heads = X.view(X.size(0), self.config.num_attention_heads, self.head_size)
        return X_heads

    def get_head_encodings_from_weight(self, embeddings, kqv_weights, selected_head):
        X_heads = self.get_all_encodings_from_weight(embeddings, kqv_weights)
        X_selected_head = X_heads[:, selected_head, :]
        return X_selected_head

    def get_selected_layer_value_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.value.weight

    def get_selected_layer_key_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.key.weight

    def get_selected_layer_query_weights(self, selected_layer):
        return self.model.encoder.layer[selected_layer].attention.self.query.weight

    def get_attention_weights(
        self, outputs, selected_layer, selected_head, selected_token
    ):
        """
        Extracts and returns attention weights for a specific layer, head, and token.

        Parameters:
            outputs: Transformer model outputs, which include attentions.
            selected_layer (int): The index of the layer from which to extract the attention weights.
            selected_head (int): The index of the head from which to extract the attention weights.
            selected_token (int): The index of the token for which to extract the attention weights.

        Returns:
            torch.Tensor: Attention weights for the specified layer, head, and token.]

            Really a static method, but leaving it like this for now.
        """
        attentions_l_h_t = outputs.attentions[selected_layer][
            :, selected_head, selected_token, :
        ]
        return attentions_l_h_t

    def get_attention_weights_all_heads(
        self, outputs, selected_layer, selected_token
    ):
        """
        Extracts and returns attention weights for a specific layer, head, and token.

        Parameters:
            outputs: Transformer model outputs, which include attentions.
            selected_layer (int): The index of the layer from which to extract the attention weights.
            selected_head (int): The index of the head from which to extract the attention weights.
            selected_token (int): The index of the token for which to extract the attention weights.

        Returns:
            torch.Tensor: Attention weights for the specified layer, head, and token.]

            Really a static method, but leaving it like this for now.
        """
        attentions_l_t = outputs.attentions[selected_layer][
            :, :, selected_token, :
        ]
        return attentions_l_t

# %%
def get_all_layer_0_value_encodings(transformer):
    layer_0_value_weights = transformer.get_selected_layer_value_weights(0)

    # V_single_position has all tokens' value encodings using _selected_token's position_ for comparison to attention-weighted output encodings

    V_all_layer_0 = transformer.get_all_encodings_from_weight(
        transformer.normalized_word_embeddings_with_position,
        layer_0_value_weights,
    )
    return {"V_all_layer_0": V_all_layer_0}


def get_intermediate_output(transformer, selected_layer, selected_token, outputs):
    selected_layer = int(selected_layer)
    # get weights for selected_layer
    selected_layer_value_weights = transformer.get_selected_layer_value_weights(
        selected_layer
    )
    if debug: print(f"{selected_layer_value_weights.shape=}")

    # <batch_size, seq_len, hidden_size>, the input for selected_layer
    layer_sequence_inputs = outputs.hidden_states[
        selected_layer
    ]  # actual input embeddings (not all-same-position-encoded)
    if debug: print(f"{layer_sequence_inputs.shape=}") # [1, 8, 768]

    # mutiply input encodings by value weights to get value encodings
    # so values for all tokens, all heads
    value_encodings = torch.einsum(
        "bsd,dv->bsv", layer_sequence_inputs, selected_layer_value_weights
    )
    if debug: print(f"{value_encodings.shape=}") # (1,8,768)<batch_size, seq_len, hidden_size>


    # get all attention weights for all heads, single layer, single token
    attentions_l_t = transformer.get_attention_weights_all_heads(
        outputs, selected_layer, selected_token
    )
    if debug: print(f"{attentions_l_t.shape=}") # (1,12,8)


    # split value encoding into per-head encodings
    encodings_for_single_token_all_heads = value_encodings.view(
        value_encodings.size(0),
        value_encodings.size(1),
        transformer.config.num_attention_heads,
        transformer.head_size,
    )
    if debug: print(f"{encodings_for_single_token_all_heads.shape=}") # (([1, 8, 12, 64]))

    post_attn_layer_value_encodings = torch.einsum("bhs,bshv->bhv", attentions_l_t, encodings_for_single_token_all_heads)
    if debug: print(f"{post_attn_layer_value_encodings.shape=}") # (1,12,64)

    return post_attn_layer_value_encodings

# %%
k=15

def get_distances_euclidean(post_attention_encoding, comparison_encodings, p=2.0):
    return F.pairwise_distance(post_attention_encoding, comparison_encodings, p=p)

def get_distances_cosine(post_attention_encoding, comparison_encodings):
    return 1 - F.cosine_similarity(post_attention_encoding, comparison_encodings, dim=-1)

def get_closest(k, transformer, input_ids, post_attention_encoding, comparison_encodings, type="euclidean"):
    if type == "Euclidean":
        encoding_distances = get_distances_euclidean(post_attention_encoding, comparison_encodings)
    else:
        encoding_distances = get_distances_cosine(post_attention_encoding, comparison_encodings)
    distances, encodings = torch.topk(encoding_distances, k, largest=False)
    if debug: print(f"{post_attention_encoding.shape=}")
    if debug: print(f"{comparison_encodings.shape=}")
    if debug: print(f"{encodings.shape=}")
    if input_ids is not None: # the indexes are into input_ids instead of encodings
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    else:
        encodings_str = transformer.tokenizer.convert_ids_to_tokens(encodings.tolist())
    distances = ["{0:.3f}".format(distance.item()) for distance in distances]

    return list(zip(encodings_str, distances))

# %%
transformer = RobertaTransformer()

# %%


def closest_to_all_values(
    text="Time flies like an arrow.",
    selected_token=5,
    selected_layer=0,
    distance_type="Euclidean"
):
    if debug: print(f"{selected_token=}")
    if not selected_token: selected_token=1
    global transformer # gross gross gross
    # transformer = RobertaTransformer()
    transformer.create_positional_embeddings(selected_token)

    inputs = transformer.tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]

    token_str = f"{transformer.tokenizer.convert_ids_to_tokens(input_ids[0, selected_token].item())} ({input_ids[0, selected_token]})"

    outputs = transformer.model(**inputs)

    layer_0_value_encodings = get_all_layer_0_value_encodings(transformer)

    post_attention_encoding = get_intermediate_output(
        transformer, selected_layer, selected_token, outputs
    )
    closest_df = pd.DataFrame()
    for head in range(transformer.config.num_attention_heads):
        closest_for_head=get_closest(
                k,
                transformer,
                None,
                post_attention_encoding[:, head, :].squeeze(0),
                layer_0_value_encodings["V_all_layer_0"][:, head, :],
                type=distance_type,
            )
        closest_for_head = [f"{token} ({dist})" for token, dist in closest_for_head]
        closest_for_head_df = pd.DataFrame(closest_for_head)
        closest_for_head_df.columns = [f"head {head}"]
        closest_df = pd.concat([closest_df,closest_for_head_df], axis=1)
    return (token_str, closest_df)

if debug: closest_to_all_values()

# %%
def tokens_for_text (text):
    global transformer # yuck
    inputs = transformer.tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    tokens = transformer.tokenizer.convert_ids_to_tokens(input_ids[0])
    token_list = [[token] for token in tokens]
    return token_list
tokens_for_text("Time flies like an arrow. Fruit flies like a banana.")

# %%
intro_markdown = """
# Token Attention Across All Layers of a RoBERTa Model

## Introduction

This quick experiment offers an exploratory look into the transformation of individual sequence tokens across the attention layers of a pre-trained RoBERTa model. The focus is on understanding the immediate post-attention output for a specific sequence position and identifying the "nearest" vocabulary tokens in the value encoding space.

## Usage Instructions

1. Input some text to be tokenized. Hit return or click "Get Tokens".
2. Click on one of the displayed tokens.
3. Choose a layer number (starting from 0) to see how the post-attention encoding at the specific level compares to the initial vocabulary embeddings.
4. Optionally, switch between the Euclidean and cosine distance metrics for a different perspective.

## Methodology

- The encodings considered are post-attention, but pre-concatenation and feed-forward network.
- The Hugging Face API doesn’t natively offer these encodings. To obtain them, output encodings from the preceding layer are projected into the value space and then the attention weights are applied.
- For the specified token:
    * The corresponding positional embedding is added to all vocabulary embeddings. Since adding positional embeddings causes significant changes to the original token embedding, it seemed best to add the selected token's positional embedding to the entire vocabulary.
    * The resulting vocabulary encodings are then projected into the transformer's Value space at the input layer.
    * By comparing the post-attention encoding against this set, the tokens corresponding to the "closest" token embeddings are showcased.
    * This procedure is replicated for each attention head.

## Why RoBERTa?

RoBERTa was chosen due to its encoder-based architecture. Additionally, avoiding BERT's segment embeddings seemed helpful.

## Interpretation

While token encodings in a transformer model typically assume a direct and specific relationship to each originating token, they are thought to transform into more general or abstract concepts as they progress through layers. Initial expectations were that higher-level encodings might correlate to some broad, interpretable concepts. However, many seem to be simply noise--try any token at layer 11. Intriguingly, initial layers often show that token encodings are closer to **another token in the sequence** than to their initial value encoding.

## Personal Thoughts

This observation suggests that later token encodings might not be firmly linked to their original tokens. It's plausible that concepts are dispersed among all encodings at each layer. This raises an interesting possibility: Perhaps the number of encodings at each layer doesn't need to be fixed. While architectures like CNNs and fully-connected networks benefit from varying neuron counts across layers, transformers might also benefit from experimenting with different numbers of output encodings per layer. Though there are potential efficiency challenges, I'm going to cleverly wave my hands in a distracting way and note that it's an area ripe for exploration.

## Future Directions

- **Models**: Exploring other models like GPT-2 (especially with the causal attention mask disabled) might be useful.
- **Visualization**: Implementing visual tools like t-SNE or PCA maps might demystify the perceived noise in token encodings.
- **Comparisons**: While it doesn't make as much obvious sense as comparing post-attention encodings to value encodings, comparing them with key or query vocabulary encodings could be interesting.

Any feedback welcome! Feel free to get in touch at [encodings@danavery.com](mailto:encodings@danavery.com)!
"""

# %%
# %%blocks

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
            choices=['Euclidean','Cosine'],
            label="Distance type",
            show_label=True,
            value="Euclidean",
            scale=0,
        )
    with gr.Row():
        show_tokens = gr.Button("Get Tokens", elem_classes="get-tokens", size="sm", scale=0)
        tokens = gr.Dataset(
            components=["text"],
            samples = tokens_for_text(initial_text),
            type="index",
            label="Select a token from here to see its closest tokens after attention processing. Click one to see its neighbors below.",
            scale=5,
            samples_per_page=1000,
        )
        selected_token_str = gr.Textbox(label="selected_token", show_label=True, scale=0)

    with gr.Row():
        combined_dataframe = gr.DataFrame(
            label=f"Nearest initial value encodings with distance, with positional embedding included",
            show_label=True,
            elem_classes="combined",
            col_count=12,
            headers=[[f"head {head}"] for head in range(12)]
        )
    tokens.click(
        closest_to_all_values,
        [text, tokens, selected_layer],
        [selected_token_str, combined_dataframe],
    )
    selected_layer.change(
        closest_to_all_values,
        [text, tokens, selected_layer, distance_type],
        [selected_token_str, combined_dataframe],
    )
    distance_type.change(
        closest_to_all_values,
        [text, tokens, selected_layer, distance_type],
        [selected_token_str, combined_dataframe],
    )
    show_tokens.click(tokens_for_text, [text], [tokens])
    text.submit(tokens_for_text, [text], [tokens])

    gr.Markdown(intro_markdown)
demo.launch()
# %%


# %%



