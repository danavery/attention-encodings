---
title: Attention Encodings
emoji: ⚡
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
---

# Token Encodings Across All Attention Layers of a RoBERTa Model

## Introduction

This quick experiment offers an exploratory look into the transformation of individual sequence tokens across the attention layers of a pre-trained RoBERTa model. The focus is on understanding the immediate post-attention output for a specific sequence position and identifying the "nearest" vocabulary tokens in the value encoding space.

## Usage Instructions

1. Input some text to be tokenized. Hit return or click "Tokenize".
2. Click on one of the displayed tokens.
3. Choose a layer number (starting from 0) to see how the post-attention encoding at the specific level compares to the initial vocabulary embeddings.
4. Optionally, switch between the Euclidean and cosine distance metrics for a different perspective.
5. Also optionally, turn applying the selected token's positional encoding to the model vocabulary on or off.
5. Safari sometimes displays a mostly-blank table on first run. Clicking on a cell seems to fix things.

## Methodology

- The encodings considered are post-attention, but pre-concatenation and feed-forward network.
- The Hugging Face API doesn’t natively offer these encodings. To obtain them, output encodings from the preceding layer are projected into the value space and then the attention weights are applied.
- For the specified token:
    * If "Use positional embeddings" is selected, the corresponding positional embedding is added to all vocabulary embeddings. Since adding positional embeddings causes significant changes to the original token embedding, it seems best to add the selected token's positional embedding to the entire vocabulary, but the option is provided to skip this step.
    * The resulting vocabulary encodings are then projected into the transformer's Value space at the input layer.
    * By comparing the post-attention encoding against this set, the tokens corresponding to the "closest" token embeddings are displayed.
    * This procedure is replicated for each attention head.

## Why RoBERTa?

RoBERTa was chosen due to its encoder-based architecture. Additionally, avoiding BERT's segment embeddings seemed helpful. The pre-trained weights are from the "roberta-base" Hugging Face model.

## Interpretation

Descriptions of the transformer architecture typically assume that later token encodings have a direct and specific relationship to each originating token--they are thought to transform into more general or abstract concepts as they progress through layers.
Initial expectations were that higher-level encodings might correlate to some broad, interpretable concepts.
However, many seem to be simply noise--try any token at layer 11.

Intriguingly, the first layer often creates new token encodings that are closer to **another token in the sequence** than to the original token. Note that in "Time flies like an arrow, fruit flies like a banana," using cosine distance, the token "Time" comes out of the attention weighting closer to "flies" than to any other vocabulary token in four of the twelve heads, and comparatively far away from its original encoding.

## Personal Thoughts

The results here suggest that later token encodings might not be firmly linked to their original tokens. It's plausible that concepts are distributed among all encodings at each layer. This raises an interesting possibility: Perhaps the number of encodings at each layer doesn't need to be fixed. While architectures like CNNs and fully-connected networks benefit from varying neuron counts across layers, transformers might also benefit from experimenting with different numbers of output encodings per layer. Though there are potential efficiency challenges, I'm going to cleverly wave my hands in a distracting way and note that it's an area ripe for exploration.

## Future Directions

- **Models**: Exploring other models like GPT-2 (especially with the causal attention mask disabled) might be useful.
- **Visualization**: Implementing visual tools like t-SNE or PCA maps might demystify the perceived noise in token encodings.
- **Comparisons**: While it doesn't make as much obvious sense as comparing post-attention encodings to value encodings, comparing them with key or query vocabulary encodings could be interesting.
- **Attention Weights**: Displaying the attention weights for the token at each head could be interesting. That's largely a UI problem, as there are already a not-insignificant number of words and numbers on the page.

Any feedback welcome! Feel free to get in touch at [encodings@danavery.com](mailto:encodings@danavery.com)!