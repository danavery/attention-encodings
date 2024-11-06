(3 Nov 2024: Demo is offline for a minor revision in the next 24-48 hours)
Current demo available [here](https://demo.danavery.com/attention-encodings).

# Distributed Far and Wide: Rethinking Layered Representation in Transformer Models

## Introduction

This experiment provides an exploratory look into the transformation of individual sequence tokens across the attention layers of a pre-trained RoBERTa model. The focus is on understanding the immediate post-attention output for specific sequence positions and identifying the "nearest" vocabulary tokens in the value encoding space.

Descriptions of transformer architecture often assume that later token encodings have a direct and specific relationship to each originating token, gradually transforming into more abstract or generalized concepts. This holds true for the most part in this analysis—but there are some intriguing exceptions.

## Key Findings

1. *Token Movement Through Layers:* Token encodings shift both toward and away from their original embeddings as they progress through the transformer layers. This movement is measured by the number of tokens closer to each encoding than its original embedding.
2. *Context vs. Function Words:* By the final layer, context-dependent words—those that rely on surrounding words to define their meaning—deviate significantly from their original embeddings, conceptually drifting far from their initial representations. In contrast, function words, like "and" or "the," remain much closer to their original embeddings, indicating that the attention mechanism alters their rankings minimally.
3. *Encoding Space Density:* Interestingly, the cosine distances between the original embeddings and the final layer embeddings show only slight variation between context and function words. This suggests that context words may occupy a denser region in the encoding space than function words, potentially accounting for their larger shifts in relative rankings.
4. *Radical Shift in Layer 5:* In this RoBERTa model, layer 5 exhibits a dramatic shift, pushing token encodings far from their original embeddings. However, by layer 6, they revert closer to their initial positions. Whether this is a particular quirk of the RoBERTa model or a general property of transformers is unclear. It's also unclear if the shift implies a particular recontextualization action or is simply a place where the model goes "off-track" temporarily. The layer 5 shift happens regardless of the token sequence processed.

This behavior suggests that while the attention layers impact the rankings of "context" words more than function words, this may be attributed more to differences in encoding space density for the two types of tokens than to inherent differences in movement of tokens through the encoding space.


## Usage Instructions

1. Input some text to be tokenized. Hit return or click "Tokenize"
2. Click on one of the displayed tokens
3. Choose a layer number (0-11) to see:
    * How far the token's encoding has moved from related vocabulary terms
    * How tightly it clusters with other sequence tokens
4. Use the Journey tab to visualize how the token moves with its sequence through the layers for each head
5. Switch between distance metrics and toggle positional embeddings to explore different aspects of the clustering

## Methodology

To understand how token encodings evolve through attention layers, we analyze them in two ways:

1. Distance Measurements
    * Compare post-attention encodings to the initial vocabulary embeddings projected into value space
        * The encodings considered are post-attention, but pre-concatenation and feed-forward network
        * The Hugging Face API doesn’t natively offer these encodings. To obtain them, output encodings from the preceding layer are projected into the value space and then the attention weights are applied.
    * Track how quickly token encodings diverge from semantically related terms
    * Measure distances to tokens that had similar meanings in the original embedding space

2. PCA Visualization
    * Project the high-dimensional token encodings into 2D space
    * Track a single token's journey through all layers
    * Show other tokens' encodings at the current layer to reveal clustering
    * Observe how entire clusters move between layers

This reveals both the loss of original semantic relationships and the emergence of tight sequence-level clustering from the very first layer.

## Technical Notes

* The encodings considered are post-attention, but pre-concatenation and feed-forward network.
* The Hugging Face API doesn’t natively offer these encodings. To obtain them, output encodings from the preceding layer are projected into the value space and then the attention weights are applied.
* For the specified token:
    * If "Use positional embeddings" is selected, the corresponding positional embedding is added to all vocabulary embeddings. Since adding positional embeddings causes significant changes to the original token embedding, it seems best to add the selected token's positional embedding to the entire vocabulary, but the option is provided to skip this step.
    * The resulting vocabulary encodings are then projected into the transformer's Value space at the input layer. This way they stay roughly in the same conceptual space (i.e., their original high-dimensional structure) as the original tokens, and closer to the space of the post-attention encodings.
    * By comparing the post-attention encoding against this set, the tokens corresponding to the "closest" token embeddings are displayed.
    * This procedure is replicated for each attention head.
    * Distances are also computed from the selected post-attention mechanism token encoding at the selected layer to all sequence tokens at the same layer.

## Why RoBERTa?

RoBERTa was chosen due to its encoder-based architecture. Additionally, avoiding BERT's segment embeddings seemed helpful. The pre-trained weights are from the "roberta-base" Hugging Face model.

## Interpretation

Conventional descriptions of transformer architectures suggest that higher-level encodings might correlate to some broad, interpretable concepts derived from the original token embeddings. But the evidence suggests something different.

1. From the first layer, tokens show high distances to their original value encodings and to semantically-related terms
2. Rather than developing interpretable features, the encodings appear to immediately begin distributing information across all sequence positions
3. Entire groups of encodings move dramatically through the embedding space together, maintaining tight clustering but losing individual semantic meaning

This behavior is more reminiscent of how feed-forward networks distribute information throughout their layers than the commonly assumed hierarchical feature development. The sequence positions appear to function primarily as containers for distributed information rather than as carriers of increasingly sophisticated token-level meaning, just as in feed-forward networks, except these networks manipulate tensors rather than scalar values. This also introduces the same issues with interpretability as feed-forward networks.

This distributed structure suggests that transformers’ representations may not have the clear, hierarchical interpretability often ascribed to them. Instead of building up abstract features from concrete ones, they seem to immediately begin mixing information across all positions in service of processing the sequence as a whole.

Intriguingly, **the first layer often creates new token encodings that are closer to the value encoding of another token in the sequence** than to the original token. For example, in 'Time flies like an arrow, fruit flies like a banana,' using cosine distance, the token 'Time' moves immediately away from its original meaning and clusters with its sequence neighbors in four of the twelve heads, showing how quickly the distribution of information begins.

Interestingly—and somewhat unexpectedly—head 7 preserves much of the original token encoding for the first layer. But it's not preserved for the next layer or certainly any later layers.


## Personal Thoughts

The results here suggest that later token encodings are not linked to their original tokens, and that information is distributed among all encodings at each layer. *Attention layers seem to be using the sequence positions solely as tools to group and mix information between encodings from the very beginning*, rather than preserving individual feature identities.

This raises a few interesting possibilities:

* *The number of encodings at each layer doesn't need to be fixed.* While architectures like CNNs and fully-connected networks benefit from varying neuron counts across layers, transformers might also benefit from experimenting with different numbers of output encodings per layer.
* *The dimensionality of the encodings doesn't need to be fixed between layers.* Since the attention layers seem to be using the sequence positions to group and mix information between encodings from the very beginning, rather than preserving individual feature identities, there appears to be no obvious reason to keep the encoding dimensionality fixed (at 768 in BERT's case).
* If one of the primary breakthroughs of attention layers is that they act primarily like feed-forward networks, but by manipulating tensors rather than scalars, this means that *attention mechanisms might be useful in other types of networks*. Other architectures could be enhanced by adding group-wise combination to their existing scalar operations.

Though there are potential efficiency challenges, I'm going to cleverly wave my hands in a distracting way and note that it's an area ripe for exploration. If transformers are indeed less focused on token-specific identity than widely believed, this could open up new avenues in architecture design, especially around flexibility in layer structure and dimensionality.


## Future Directions

- **Models**: Exploring other models like GPT-2 (especially with the causal attention mask disabled) might reveal whether this mixing behavior is unique to RoBERTa or common across transformer architectures.
- **Comparisons**: While it doesn't make as much obvious sense as comparing post-attention encodings to value encodings, comparing them with key or query vocabulary encodings could be interesting.
- **Attention Weights**: Exploring per-head attention weights could shed light on each head’s specific role in information distribution and token blending, though displaying this in the UI presents a challenge due to space constraints.

I believe these insights offer a fresh perspective on transformers and open up intriguing directions for future research. Any feedback welcome! Feel free to get in touch at [encodings@danavery.com](mailto:encodings@danavery.com)!
