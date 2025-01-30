from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from utils import load_json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def construct_df(path):
    metric_to_scores = load_json(path)
    df = pd.DataFrame(metric_to_scores)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    print(f"explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"PC1: {pca.components_[0]}")
    
    #print(f"pca.components_: {pca.components_}")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1])
    for i, label in enumerate(df.index):
        plt.text(principal_components[i, 0], 
                principal_components[i, 1], 
                label, fontsize=6, ha='right', va='bottom', color='red')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    construct_df("output/albumentations/rainy/combined_results.json")
    #construct_df("output/albumentations/foggy/combined_foggy_results.json")



