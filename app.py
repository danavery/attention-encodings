import logging
import os

from attention_analyzer import AttentionAnalyzer
from attention_analyzer_ui import AttentionAnalyzerUI
from transformer_manager import TransformerManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)


class App:
    def __init__(self):
        transformer = TransformerManager()
        analyzer = AttentionAnalyzer(transformer)
        self.ui = AttentionAnalyzerUI(analyzer)

    def launch(self):
        self.ui.launch()


if __name__ == "__main__":
    App().launch()
