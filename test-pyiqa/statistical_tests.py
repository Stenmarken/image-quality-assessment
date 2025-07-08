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
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def concatenate_dfs(coeff_dfs, combine_fogginess):
    filtered_dfs = []
    for df in coeff_dfs:
        name = df.name
        if not combine_fogginess:
            df = df[df["Weather"] != "foggy_2"].reset_index(drop=True)
            df.name = name
        filtered_dfs.append(df)

    combined_df = pd.DataFrame(
        {
            "Image_set": filtered_dfs[0]["Location"].astype(str)
            + "_"
            + filtered_dfs[0]["Weather"].astype(str)
            + "_"
            + filtered_dfs[1]["Reference image"].astype(str)
        }
    )
    for df in filtered_dfs:
        combined_df[f"{df.name}_SRCC"] = df["SRCC"]
    return combined_df


def friedman_test(combined_df):
    data_array = combined_df.to_numpy().T
    friedman_stat, friedman_p_value = friedmanchisquare(*data_array)
    print(f"Friedman_stat: {friedman_stat}. friedman_p: {friedman_p_value}")


def manual_wilcoxon_signed_rank(combined_df, col1, col2):
    diff = combined_df[col1] - combined_df[col2]
    diff = diff[diff != 0]  # remove zero differences

    abs_diff = abs(diff)
    ranks = abs_diff.rank()
    signed_ranks = ranks.copy()
    signed_ranks[diff < 0] = -signed_ranks[diff < 0]

    positive_sum = signed_ranks[signed_ranks > 0].sum()
    negative_sum = signed_ranks[signed_ranks < 0].sum()
    W = min(abs(positive_sum), abs(negative_sum))
    return W, positive_sum, negative_sum


def pairwise_wilcoxon(combined_df):
    pairs = list(combinations(combined_df.columns, 2))
    ps = [-1.0] * len(pairs)
    stats = [-1.0] * len(pairs)
    index_to_pair = {}
    for i, (m_1, m_2) in enumerate(pairs):
        col1, col2 = combined_df[m_1], combined_df[m_2]
        index_to_pair[i] = (m_1, m_2)
        w_stat, w_p = wilcoxon(col1, col2)
        ps[i] = w_p
        stats[i] = w_stat
    _, ps_corr, _, _ = multipletests(ps, alpha=0.05, method="holm")
    for i, (m_1, m_2) in index_to_pair.items():
        W, positive_sum, negative_sum = manual_wilcoxon_signed_rank(
            combined_df, m_1, m_2
        )
        assert W == stats[i]
        print(
            f"{m_1} - {m_2}. W = {W}, R+ = {positive_sum}, R- = {negative_sum}, p = {ps_corr[i]:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical tests script")

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Path to the env file containing wandb info",
    )
    parser.add_argument(
        "--combine_fogginess",
        required=False,
        default=False,
        action="store_true",
        help="If true, include both foggy^1=foggy_0.0_1.0_0.7 and foggy^2=foggy_0.5_0.9",
    )
    return parser.parse_args()


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

    concat_df = concatenate_dfs(coeff_dfs, args.combine_fogginess)
    concat_df = concat_df.drop("Image_set", axis=1)
    friedman_test(concat_df)
    pairwise_wilcoxon(concat_df)
