from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from utils import load_json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def pca(path):
    metric_to_scores = load_json(path)
    df = (pd.DataFrame(metric_to_scores)).T
    print(df.iloc[:5, :5])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    print(f"explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"PC1: {pca.components_[0]}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1])
    for i, label in enumerate(df.index):
        plt.text(
            principal_components[i, 0],
            principal_components[i, 1],
            label,
            fontsize=6,
            ha="right",
            va="bottom",
            color="red",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA for rainy images found at {path}")
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="grey", linestyle="--", linewidth=0.7)
    plt.grid(True)
    plt.show()


def visualize_separate_figs(path, image_names, standardise=True):
    metric_to_scores = load_json(path)
    scaler = StandardScaler()
    for metric, scores in metric_to_scores.items():
        xs = []
        ys = []
        for img in image_names:
            xs.append(int(img[:-4]))
            ys.append(scores[img])
        if standardise:
            ys_norm = scaler.fit_transform(np.array(ys).reshape(-1, 1))
        else:
            ys_norm = ys
        plt.scatter(xs, ys_norm, color="red", s=10)
        plt.xlabel("Noise severity")
        plt.ylabel("Score")
        plt.title(f"{metric}")
        plt.show()


def all_metrics(path):
    metric_to_scores = load_json(path)
    return list(metric_to_scores.keys())


def all_images():
    return [f"{i}.png" for i in range(100)]


def visualize_one_fig(
    path,
    metrics,
    image_names,
    standardise=True,
    save_plots=False,
    save_path="plot.png",
    fig_info="",
):
    metric_to_scores = load_json(path)
    scaler = StandardScaler()
    colors = ["red", "blue", "green", "purple", "orange", "brown"]

    plt.figure(figsize=(10, 8))
    for idx, metric in enumerate(metrics):
        scores = metric_to_scores[metric]
        xs = []
        ys = []
        for img in image_names:
            xs.append(int(img[:-4]))
            ys.append(scores[img])
        if standardise:
            ys_norm = scaler.fit_transform(np.array(ys).reshape(-1, 1))
        else:
            ys_norm = ys
        plt.scatter(xs, ys_norm, color=colors[idx % len(colors)], s=10, label=metric)

    plt.xlabel("Noise severity")
    plt.ylabel("Score")
    plt.title(f"Scores for IQA metrics at different {fig_info} severities")
    plt.legend(title="Metrics", loc="upper right")
    if save_plots:
        file_name = "_".join(map(str, metrics))
        full_path = os.path.join(save_path, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, bbox_inches="tight")
        print(f"Plot saved as {save_path}")
    else:
        plt.show()


def run_plots():
    # path = "output/albumentations/rainy/combined_results.json"
    path = "output/albumentations/foggy/combined_foggy_results.json"
    image_names = all_images()
    metrics = all_metrics(path)
    metrics_chunks = np.array_split(metrics, 12)
    for chunk in metrics_chunks:
        visualize_one_fig(
            path,
            chunk,
            image_names,
            standardise=True,
            save_plots=True,
            save_path="output/foggy_plots",
            fig_info="Rain",
        )


if __name__ == "__main__":
    pca("output/albumentations/rainy/combined_results.json")
    # pca("output/albumentations/foggy/combined_foggy_results.json")
