import json
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np
import argparse
from dotenv import load_dotenv
import os
from wandb_results import get_coefficients_table
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from scipy.stats import wilcoxon
from tabulate import tabulate
from itertools import permutations
from statsmodels.stats.multitest import multipletests


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical tests script")

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Path to the env file containing wandb info",
    )
    return parser.parse_args()


def calculate_combined_df(coeff_dfs):
    combined_df = pd.DataFrame(
        {
            "Image_set": coeff_dfs[0]["location"].astype(str)
            + "_"
            + coeff_dfs[0]["prediction_key"].astype(str)
            + "_"
            + coeff_dfs[1]["weather"].astype(str)
        }
    )
    for df in coeff_dfs:
        combined_df[f"{df.name}_SRCC"] = df["SRCC"]
    return combined_df


def manual_wilcoxon_signed_rank(combined_df, col1, col2):
    diff = combined_df[col1] - combined_df[col2]
    diff = diff[diff != 0]  # remove zero differences

    abs_diff = abs(diff)
    ranks = abs_diff.rank()
    signed_ranks = ranks.copy()
    signed_ranks[diff < 0] = -signed_ranks[diff < 0]

    positive_sum = signed_ranks[signed_ranks > 0].sum()
    negative_sum = signed_ranks[signed_ranks < 0].sum()

    print(f"Positive sum: {positive_sum}. Negative sum: {negative_sum}")
    print("Result", min(abs(positive_sum), abs(negative_sum)))


def wilcoxon_signed_rank(combined_df):
    pairs = list(permutations(combined_df.columns, 2))
    p_values = [-1.0] * len(pairs)
    stat_values = [-1.0] * len(pairs)
    index_to_pair = {}
    for i, pair in enumerate(pairs):
        col1, col2 = combined_df[pair[0]], combined_df[pair[1]]
        index_to_pair[i] = pair
        w_stat_one_side, w_p_one_side = wilcoxon(col1, col2, alternative="greater")
        w_stat, w_p = wilcoxon(col1, col2)
        assert (
            w_stat == w_stat_one_side
        ), f"w_stat: {w_stat}. w_stat_one_side: {w_stat_one_side}"
        # assert w_p == w_p_one_side
        # wilcoxon_matrix.loc[col1, col2] = (w_stat, w_p)
        # wilcoxon_matrix.loc[col2, col1] = (w_stat, w_p)
        p_values[i] = w_p
        stat_values[i] = w_stat

    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    for i, pair in index_to_pair.items():
        col1, col2 = pair
        # wilcoxon_matrix.loc[col1, col2] = pvals_corrected[i]
        print(f"{col1} vs {col2}: p={pvals_corrected[i]:.5f}. W={stat_values[i]:.5f}")
        """
        print(
            f"{col1}: Mean: {combined_df[col1].mean()}. Median: {combined_df[col1].median()}."
        )
        print(
            f"{col2}: Mean: {combined_df[col2].mean()}. Median: {combined_df[col2].median()}."
        )
        """
        manual_wilcoxon_signed_rank(combined_df, col1, col2)
        print("\n\n")

    # print("pvals_corrected", pvals_corrected)
    # print("rejected", rejected)


def wilcoxon_signed_rank2(combined_df):
    wilcoxon_qalign = combined_df[f"qalign_SRCC"]
    wilcoxon_ilniqe = combined_df[f"ilniqe_SRCC"]
    print("\n\nWilcoxon test between Q-Align and Ilniqe")
    wilcoxon_statistic, wilcoxon_p_value = wilcoxon(wilcoxon_qalign, wilcoxon_ilniqe)
    print(f"Wilcoxon statistic = {wilcoxon_statistic:.3f}")
    print(f"p-value = {wilcoxon_p_value:.3e}")
    return wilcoxon_statistic, wilcoxon_p_value


def friedman_nemenyi_test(combined_df):
    ranked_df = combined_df.drop(combined_df.columns[0], axis=1)

    ranked_df = ranked_df.rank(axis=1, ascending=False)

    mean_ranks = ranked_df.mean()
    mean_ranks = mean_ranks.round(3)
    print("mean_ranks", mean_ranks)

    combined_df = combined_df.drop(combined_df.columns[0], axis=1)

    data_array = combined_df.to_numpy().T

    friedman_stat, friedman_p_value = friedmanchisquare(*data_array)

    print(f"Friedman chi-square statistic = {friedman_stat:.3f}")
    print(f"p-value = {friedman_p_value:.3e}")

    print("\n\nNemenyi test")
    nemenyi_p_matrix = sp.posthoc_nemenyi_friedman(combined_df)
    nemenyi_p_matrix = nemenyi_p_matrix.round(4)

    print(nemenyi_p_matrix)


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(dotenv_path=args.env)
    topiq_nr_id = os.getenv("TOPIQ_NR_ANALYSIS")
    dbcnn_id = os.getenv("DBCNN_ANALYSIS")
    qualiclip_id = os.getenv("QUALICLIP_ANALYSIS")
    qalign_id = os.getenv("QALIGN_ANALYSIS")
    ilniqe_id = os.getenv("ILNIQE_ANALYSIS")
    entity = os.getenv("ENTITY")
    api = wandb.Api()

    analysis_pairs = [
        ("topiq_nr", topiq_nr_id),
        ("dbcnn", dbcnn_id),
        ("qualiclip", qualiclip_id),
        ("qalign", qalign_id),
        ("ilniqe", ilniqe_id),
    ]
    coeff_dfs = []
    for pair in analysis_pairs:
        coeff_dfs.append(get_coefficients_table(entity, pair[0], pair[1]))

    combined_df = calculate_combined_df(coeff_dfs)
    combined_df = combined_df.drop(combined_df.columns[0], axis=1)
    # friedman_nemenyi_test(combined_df)
    wilcoxon_signed_rank(combined_df)
    # manual_wilcoxon_signed_rank(combined_df)
    # wilcoxon_signed_rank(combined_df)
    # wilcoxon_signed_rank2(combined_df)
