import gradio as gr

from attention_analyzer import AttentionAnalyzer


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
            combined_dataframe.change(js=custom_js)
        demo.launch(server_name="0.0.0.0")
