import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from attention_analyzer import AttentionAnalyzer


class AttentionAnalyzerUI:
    def __init__(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer
        self.k = analyzer.k

    def get_intro_markdown(self):
        # read in README.md and strip out metadata if present
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            position = intro_markdown.find("# Distributed")
            if position != -1:
                intro_markdown = intro_markdown[position:]
        return intro_markdown

    def update_tabs(self, text, tokens, selected_layer, distance_type, use_positional, head_selector):
        """
        Update the tabs with the new data.
        """
        residual_distance_dataframe = self.analyzer.get_residual_distances_df(
            text, tokens, distance_type, use_positional
        )

        selected_token_str, combined_dataframe, intertoken_dataframe = (
            self.analyzer.closest_to_all_values(
                text, tokens, selected_layer, distance_type, use_positional
            )
        )
        pca_plot = self.create_pca_plot(text, tokens, selected_layer, head_selector)
        return (
            selected_token_str,
            combined_dataframe,
            intertoken_dataframe,
            pca_plot,
            residual_distance_dataframe,
        )

    def create_pca_plot(self, text, token_idx, layer, head):
        """
        Visualize token's journey through layers using PCA.
        """
        # Get the token's embeddings through layers and all tokens at current layer
        journey_data = self.analyzer.get_token_journey(text, token_idx, layer, head)
        embeddings = journey_data["embeddings"]
        token_info = journey_data["token_info"]

        # Convert embeddings to numpy array for PCA
        embedding_array = np.stack([e.numpy() for e in embeddings])

        # Apply PCA to journey points first
        pca = PCA(n_components=2)
        journey_points = pca.fit_transform(embedding_array)

        # Transform current layer points using the same PCA
        current_layer_points = pca.transform(journey_data["all_current"].numpy())

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot current layer context points in gray
        ax.scatter(current_layer_points[:, 0], current_layer_points[:, 1],
                    color='gray', alpha=0.3, label=f'Other encodings at layer {layer}')

        # Plot journey points in blue
        ax.scatter(journey_points[:, 0], journey_points[:, 1],
                color='blue', alpha=0.5,
                label='Token journey')

        # Highlight current layer in red
        ax.scatter(journey_points[layer + 1, 0], journey_points[layer + 1, 1],
                    color='red', s=100, label=f'Layer {layer}')

        # Add layer numbers next to points
        for i, (x, y) in enumerate(journey_points):
            label = "Initial" if i == 0 else f"L{i-1}"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')

        ax.set_title(f'Token "{token_info["string"]}" Journey Through Layers')
        ax.set_xlabel('First PCA Component')
        ax.set_ylabel('Second PCA Component')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        return fig

    def update_token_display(self, text):
        tokens = self.analyzer.get_tokens_for_text(text)
        return gr.update(samples=tokens, components=["text"])

    def launch(self):
        intro_markdown = self.get_intro_markdown()

        custom_css = """
                    .combined span { font-size: 9px !important; padding: 2px }
                    .combined div { min-height: 10px !important}
        """
        # This is a hack to make Safari render the dataframe on load
        head = """
        <script>
        setTimeout(() => {
            window.scrollTo(0, 10);
            window.scrollTo(0, 0);
        }, 1500);
        </script>"""

        with gr.Blocks(css=custom_css, head=head) as demo:
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
                    value="Cosine",
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

            with gr.Tabs():
                with gr.Tab("Distances"):
                    with gr.Row():
                        residual_distance_dataframe = gr.DataFrame(
                            label="Distance between the post-attention layer residual output and the original vocabulary token embeddings",
                            show_label=True,
                            elem_classes="combined",
                            col_count=12,
                            headers=[f"layer {layer}" for layer in range(12)],
                        )
                with gr.Tab("Attention Deltas"):
                    with gr.Row():
                        combined_dataframe = gr.DataFrame(
                            label="Nearest initial token value encodings to selected post-attention token encoding",
                            show_label=True,
                            elem_classes="combined",
                            col_count=12,
                            headers=[f"head {head}" for head in range(12)],
                        )
                    with gr.Row():
                        intertoken_dataframe = gr.DataFrame(
                            label="Nearest sequence position post-attention encodings (shown as original tokens at that position)",
                            show_label=True,
                            elem_classes="combined",
                            col_count=12,
                            headers=[f"head {head}" for head in range(12)],
                        )
                with gr.Tab("Journey"):
                    head_selector = gr.Dropdown(
                        choices=range(12),
                        label="Attention Head",
                        show_label=True,
                        value=0
                    )
                    pca_plot = gr.Plot()

            tokens.click(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )
            selected_layer.change(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )
            distance_type.change(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )
            use_positional.change(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )
            head_selector.change(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )
            demo.load(
                fn=self.update_tabs,
                inputs=[text, tokens, selected_layer, distance_type, use_positional, head_selector],
                outputs=[
                    selected_token_str,
                    combined_dataframe,
                    intertoken_dataframe,
                    pca_plot,
                    residual_distance_dataframe,
                ],
            )

            show_tokens.click(
                self.update_token_display, inputs=[text], outputs=[tokens]
            )
            text.submit(self.update_token_display, inputs=[text], outputs=[tokens])

            gr.Markdown(intro_markdown)

        demo.launch(server_name="0.0.0.0")
