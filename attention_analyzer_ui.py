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

    def update_tabs(
        self,
        text,
        selected_token,
        distance_type,
        use_positional,
    ):
        """
        Update the tabs with the new data.
        """
        residual_metrics = self.analyzer.get_all_token_metrics(
            text, selected_token, distance_type
        )
        token_string, residual_distance_dataframe = (
            self.analyzer.get_residual_distances_df(residual_metrics, selected_token)
        )

        residual_distance_plot = self.create_residual_distance_plot(residual_metrics)
        residual_rank_plot = self.create_residual_rank_plot(residual_metrics)

        return (
            token_string,
            residual_distance_dataframe,
            residual_distance_plot,
            residual_rank_plot,
        )

    def create_residual_rank_plot(self, residual_metrics):

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each token's ranking journey
        for pos, rankings in residual_metrics["all_rankings"].items():
            token = residual_metrics["token_strings"][pos]
            if pos == residual_metrics["selected_token"]:
                # Selected token - bold blue line with markers
                ax.plot(range(12), rankings, "b-o", linewidth=2, label=token)
            else:
                # Other tokens - thin gray lines
                ax.plot(range(12), rankings, color="gray", alpha=0.3)

        # Add token labels at the end of each line
        for pos, rankings in residual_metrics["all_rankings"].items():
            token = residual_metrics["token_strings"][pos]
            # Add text slightly offset from the final point
            ax.annotate(
                token,
                (11, rankings[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                alpha=1.0 if pos == residual_metrics["selected_token"] else 0.3,
            )

        # Flip y-axis since lower rank = better
        ax.invert_yaxis()

        ax.set_xlabel("Layer")
        ax.set_ylabel("Rank of Original Token in Vocabulary")
        ax.set_title("Token Rankings Across Layers")
        ax.grid(True)
        ax.legend()

        return fig

    def create_residual_distance_plot(self, residual_metrics):

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each token's journey
        for pos, distances in residual_metrics["all_distances"].items():
            token = residual_metrics["token_strings"][pos]
            if pos == residual_metrics["selected_token"]:
                ax.plot(range(12), distances, "b-o", linewidth=2, label=token)
            else:
                ax.plot(range(12), distances, color="gray", alpha=0.3)

        ax.set_xlabel("Layer")
        ax.set_ylabel(
            f'Distance from Initial Embedding ({residual_metrics["distance_type"]})'
        )
        ax.set_title("Token Distances from Initial Embeddings Across Layers")
        ax.grid(True)
        ax.legend()

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
        }, 4000);
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
                with gr.Tab("Distances and Rankings"):
                    with gr.Row():
                        residual_distance_dataframe = gr.DataFrame(
                            label="Distance between the post-attention layer residual output and the original vocabulary token embeddings",
                            show_label=True,
                            elem_classes="combined",
                            col_count=12,
                            headers=[f"layer {layer}" for layer in range(12)],
                        )
                with gr.Tab("Distance across Layers"):
                    residual_distance_plot = gr.Plot()
                with gr.Tab("Rankings across Layers"):
                    residual_rank_plot = gr.Plot()

            update_handler_params = {
                "fn": self.update_tabs,
                "inputs": [
                    text,
                    tokens,
                    distance_type,
                    use_positional,
                ],
                "outputs": [
                    selected_token_str,
                    residual_distance_dataframe,
                    residual_distance_plot,
                    residual_rank_plot,
                ],
            }
            tokens.click(
                **update_handler_params,
            )
            distance_type.change(**update_handler_params)
            use_positional.change(**update_handler_params)
            demo.load(**update_handler_params)

            show_tokens.click(
                self.update_token_display, inputs=[text], outputs=[tokens]
            )
            text.submit(self.update_token_display, inputs=[text], outputs=[tokens])

            gr.Markdown(intro_markdown)

        demo.launch(server_name="0.0.0.0")
