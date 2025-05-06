import gradio as gr
import matplotlib.pyplot as plt

from attention_analyzer import AttentionAnalyzer
from transformer_manager import TransformerManager


class AttentionAnalyzerUI:
    def __init__(self, analyzer: AttentionAnalyzer):
        self.setup(analyzer)

    def setup(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer
        self.k = analyzer.k
        self.num_layers = analyzer.num_layers
        self.sim_fig = None
        self.rank_fig = None

    def change_model(self, text, selected_token, use_positional, model_name):
        transformer = TransformerManager(model_name)
        analyzer = AttentionAnalyzer(transformer)
        self.setup(analyzer)
        return(self.update_tabs(text, selected_token, use_positional))

    def get_intro_markdown(self):
        # read in README.md and strip out metadata if present
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            start_position = intro_markdown.find("# Dense")
            end_position = intro_markdown.find("## Screenshots")
            if start_position != -1:
                intro_markdown = intro_markdown[start_position:end_position]
        return intro_markdown

    def update_tabs(
        self,
        text,
        selected_token,
        use_positional,
    ):
        """
        Update the tabs with the new data.
        """
        if not selected_token:
            selected_token = 1
        selected_token = int(selected_token)
        token_metrics = self.analyzer.get_all_token_metrics(text, use_positional)
        token_string, similarity_dataframe = self.analyzer.get_similarities_df(
            token_metrics, selected_token
        )

        similarity_plot = self.create_similarity_plot(token_metrics, selected_token)
        rank_plot = self.create_rank_plot(token_metrics, selected_token)
        tokens = gr.Dataset(samples=self.analyzer.get_tokens_for_text(text))
        return (token_string, similarity_dataframe, similarity_plot, rank_plot, tokens)

    def create_rank_plot(self, token_metrics, selected_token):
        if hasattr(self, "rank_fig"):
            plt.close(self.rank_fig)
        rank_fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each token's ranking journey
        for pos in range(len(token_metrics["faiss_results"])):
            rankings = AttentionAnalyzer.get_rankings_for_token(
                pos, token_metrics["faiss_results"], token_metrics["token_ids"]
            )
            token_string = token_metrics["token_strings"][pos]

            if pos == selected_token:
                # Selected token - bold blue line with markers
                ax.plot(range(self.num_layers), rankings, "b-o", linewidth=2, label=token_string)
            else:
                # Other tokens - thin gray lines
                ax.plot(range(self.num_layers), rankings, color="gray", alpha=0.3)

            # Add token labels at the end of each line
            ax.annotate(
                token_string,
                (self.num_layers - 1, rankings[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                alpha=1.0 if pos == selected_token else 0.3,
            )

        # Flip y-axis since lower rank = better
        ax.invert_yaxis()

        ax.set_xlabel("Layer")
        ax.set_ylabel("Rank of Original Token in Vocabulary")
        ax.set_title("Token Rankings Across Layers")
        ax.grid(True)
        ax.legend()

        return rank_fig

    def create_similarity_plot(self, token_metrics, selected_token):
        if hasattr(self, "sim_fig"):
            plt.close(self.sim_fig)
        sim_fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each token's journey by layer
        for pos in range(len(token_metrics["faiss_results"])):
            similarities = AttentionAnalyzer.get_similarities_for_token(
                pos, token_metrics["faiss_results"], token_metrics["token_ids"]
            )
            token_string = token_metrics["token_strings"][pos]
            if pos == selected_token:
                ax.plot(
                    range(self.num_layers), similarities, "b-o", linewidth=2, label=token_string
                )
            else:
                ax.plot(range(self.num_layers), similarities, color="gray", alpha=0.3)

            # Add token labels at the end of each line
            ax.annotate(
                token_string,
                (self.num_layers - 1, similarities[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                alpha=1.0 if pos == selected_token else 0.3,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Similarity to Initial Embedding (Cosine)")
        ax.set_title("Token Similarities to Initial Embeddings Across Layers")
        ax.grid(True)
        ax.legend()

        return sim_fig

    def update_token_display(self, text):
        tokens = self.analyzer.get_tokens_for_text(text)
        return gr.update(samples=tokens, components=["text"])

    def launch(self):
        intro_markdown = self.get_intro_markdown()

        custom_css = """
                    .combined span { font-size: 9px !important; padding: 2px }
                    .combined div { min-height: 10px !important}
                    .combined .cell-wrap { padding: 0 !important }
        """

        with gr.Blocks(css=custom_css) as demo:
            initial_text = "Time flies like an arrow. Fruit flies like a banana."
            with gr.Row():
                text = gr.Textbox(
                    placeholder="Text to process",
                    value=initial_text,
                    scale=4,
                    label="Input text",
                )

                use_positional = gr.Checkbox(
                    label="Use positional embeddings",
                    show_label=True,
                    value=True,
                    scale=0,
                )
                model = gr.Dropdown(
                    choices=["roberta-base", "roberta-large"],
                    value="roberta-base",
                    interactive=True,
                )
            with gr.Row():
                show_tokens = gr.Button("Tokenize", elem_classes="get-tokens", size="sm", scale=0)
                tokens = gr.Dataset(
                    components=["text"],
                    samples=[],
                    type="index",
                    label="Select a token from here to see its closest tokens after attention processing. Click one to see its neighbors below.",
                    scale=5,
                    samples_per_page=1000,
                )
                selected_token_str = gr.Textbox(label="Selected Token", show_label=True, scale=0)

            with gr.Tabs():
                with gr.Tab("Similarity and Rankings"):
                    with gr.Row():
                        similarity_dataframe = gr.DataFrame(
                            label="Similarities between the post-attention layer post-residual-add output and the original vocabulary token embeddings",
                            show_label=True,
                            elem_classes="combined",
                            col_count=self.num_layers,
                            headers=[f"layer {layer}" for layer in range(self.num_layers)],
                        )
                with gr.Tab("Similarity across Layers"):
                    similarity_plot = gr.Plot()
                with gr.Tab("Rankings across Layers"):
                    rank_plot = gr.Plot()

            update_handler_params = {
                "fn": self.update_tabs,
                "inputs": [
                    text,
                    tokens,
                    use_positional,
                ],
                "outputs": [
                    selected_token_str,
                    similarity_dataframe,
                    similarity_plot,
                    rank_plot,
                    tokens,
                ],
            }
            tokens.click(
                **update_handler_params,
            )
            use_positional.change(**update_handler_params)
            demo.load(**update_handler_params)

            show_tokens.click(self.update_token_display, inputs=[text], outputs=[tokens])
            text.submit(self.update_token_display, inputs=[text], outputs=[tokens])

            model_change_params = {
                "fn": self.change_model,
                "inputs": update_handler_params["inputs"] + [model],
                "outputs": update_handler_params["outputs"]
            }
            print(model_change_params)
            model.change(**model_change_params)

            gr.Markdown(intro_markdown)

        demo.launch(server_name="0.0.0.0")
