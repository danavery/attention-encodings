# Token Encodings Across All Attention Layers of a RoBERTa Model

## Introduction

This experiment challenges conventional assumptions, suggesting that transformers may not retain hierarchical interpretability across layers as often assumed.

This quick experiment offers an exploratory look into the transformation of individual sequence tokens across the attention layers of a pre-trained RoBERTa model. The focus is on understanding the immediate post-attention output for a specific sequence position and identifying the "nearest" vocabulary tokens in the value encoding space.

Descriptions of the transformer architecture typically assume that later token encodings have a direct and specific relationship to each originating token--they are thought to transform into more general or abstract concepts as they progress through layers. But this is not what we see here.

## Usage Instructions

1. Input some text to be tokenized. Hit return or click "Tokenize".
2. Click on one of the displayed tokens.
3. Choose a layer number (starting from 0) to see how the post-attention encoding at the specific level compares to the initial vocabulary embeddings.
4. Optionally, switch between the Euclidean and cosine distance metrics for a different perspective.
5. Also optionally, turn applying the selected token's positional encoding to the model vocabulary on or off.
5. **Safari sometimes displays a mostly-blank table on first run. Clicking on a cell seems to fix things.** This seems to be a Safari/Gradio issue.

## Methodology

- The encodings considered are post-attention, but pre-concatenation and feed-forward network.
- The Hugging Face API doesn’t natively offer these encodings. To obtain them, output encodings from the preceding layer are projected into the value space and then the attention weights are applied.
- For the specified token:
    * If "Use positional embeddings" is selected, the corresponding positional embedding is added to all vocabulary embeddings. Since adding positional embeddings causes significant changes to the original token embedding, it seems best to add the selected token's positional embedding to the entire vocabulary, but the option is provided to skip this step.
    * The resulting vocabulary encodings are then projected into the transformer's Value space at the input layer. This way they stay roughly in the same conceptual space (i.e., their original high-dimensional structure) as the original tokens, and closer to the space of the post-attention encodings.
    * By comparing the post-attention encoding against this set, the tokens corresponding to the "closest" token embeddings are displayed.
    * This procedure is replicated for each attention head.
    * Distances are also computed from the selected post-attention mechanism token encoding at the selected layer to all sequence tokens at the same layer.

## Why RoBERTa?

RoBERTa was chosen due to its encoder-based architecture. Additionally, avoiding BERT's segment embeddings seemed helpful. The pre-trained weights are from the "roberta-base" Hugging Face model.

## Interpretation

Conventional descriptions of transformer architectures suggest that higher-level encodings might correlate to some broad, interpretable concepts. However, many appear to be simply noise--try any token at layer 11. They're obviously in a totally different conceptual space than the original tokens, and far from their original meanings.

It also seems like the information being conveyed is distributed widely among the encodings at each layer, implying that the individual sequence positions are only used to group and mix information between tokens, and don't hold any real connection to the original tokens at those positions, just like a feed-forward network mashes and distributes information throughout the network. In both cases, the focus is on distributing information across the entire network, rather than preserving individual feature identities. This also introduces the same issues with interpretability as feed-forward networks.

This distributed structure suggests that transformers’ representations may not have the clear, hierarchical interpretability often ascribed to them.

Intriguingly, **the first layer often creates new token encodings that are closer to the value encoding of another token in the sequence** than to the original token. Note that in "Time flies like an arrow, fruit flies like a banana," using cosine distance, the token "Time" comes out of the attention weighting closer to "flies" than to any other vocabulary token in four of the twelve heads, and comparatively far away from its original encoding.

Interestingly—and somewhat unexpectedly—head 7 preserves the original token encoding for the first layer. But it's not preserved for the next layer or certainly any later layers.


## Personal Thoughts

The results here suggest that later token encodings are not linked to their original tokens, and that information is distributed among all encodings at each layer. *Attention layers seem to be using the sequence positions solely as tools to group and mix information between encodings from the very beginning*, rather than preserving individual feature identities.

This raises a couple of interesting possibilities:

* *The number of encodings at each layer doesn't need to be fixed.* While architectures like CNNs and fully-connected networks benefit from varying neuron counts across layers, transformers might also benefit from experimenting with different numbers of output encodings per layer.
* *The dimensionality of the encodings doesn't need to be fixed between layers.* Since the attention layers seem to be using the sequence positions to group and mix information between encodings from the very beginning, rather than preserving individual feature identities, there appears to be no obvious reason to keep the encoding dimensionality fixed (at 768 in BERT's case).

Though there are potential efficiency challenges, I'm going to cleverly wave my hands in a distracting way and note that it's an area ripe for exploration. If transformers are indeed less focused on token-specific identity than widely believed, this could open up new avenues in architecture design, especially around flexibility in layer structure and dimensionality.


## Future Directions

- **Models**: Exploring other models like GPT-2 (especially with the causal attention mask disabled) might reveal whether this mixing behavior is unique to RoBERTa or common across transformer architectures.
- **Visualization**: Implementing visual tools like t-SNE or PCA maps might demystify the perceived noise in token encodings.
- **Comparisons**: While it doesn't make as much obvious sense as comparing post-attention encodings to value encodings, comparing them with key or query vocabulary encodings could be interesting.
- **Attention Weights**: Exploring per-head attention weights could shed light on each head’s specific role in information distribution and token blending, though displaying this in the UI presents a challenge due to space.

I believe these insights offer a fresh perspective on transformers and open up intriguing directions for future research. Any feedback welcome! Feel free to get in touch at [encodings@danavery.com](mailto:encodings@danavery.com)!
