# from pprint import pprint
import matplotlib.pyplot as plt

from attention_analyzer import AttentionAnalyzer
from transformer_manager import TransformerManager


class RankingsTest:
    def __init__(self):
        self.transformer = TransformerManager(model_name="roberta-base")
        self.analyzer = AttentionAnalyzer(self.transformer)

    def print_metrics(
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
        n = 100
        last_n_set = None
        overlap_counts = []
        for layer in range(12):
            top_n_rankings = token_metrics["faiss_results"][selected_token][layer][1][0][0:n]
            print(top_n_rankings)
            n_set = set(top_n_rankings)
            if last_n_set:
                overlap = last_n_set & n_set
                overlap_counts.append(len(overlap))
            last_n_set = n_set
        print(overlap_counts)

        # Plot from layer 1–11 (since overlap compares current to previous layer)
        layers = list(range(1, len(overlap_counts) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(layers, overlap_counts, marker='o', linewidth=2)
        plt.title(f"Top-{n} Neighbor Overlap Between Layers (Token Index {selected_token})")
        plt.xlabel("Layer (compared to previous)")
        plt.ylabel(f"Count of Shared Top-{n} Neighbors")
        plt.xticks(layers)
        plt.ylim(0, n)  # since the max possible overlap is n
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("token_overlap_plot.png", dpi=300)



if __name__ == "__main__":
    rankings_test = RankingsTest()
    rankings_test.print_metrics("Time flies like an arrow. Fruit flies like a banana.", 3, True)
