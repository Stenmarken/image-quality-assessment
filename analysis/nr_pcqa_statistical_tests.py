import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import rankdata
import argparse


def read_csv_file(path):
    df = pd.read_csv(path, usecols=["SRCC"])
    return df["SRCC"].tolist()


def wilcoxon_signed_rank(fst, snd):
    w_stat, w_p = wilcoxon(fst, snd)
    print("W_stat", w_stat)
    print("W_p", w_p)


def manual_wilcoxon_signed_rank(fst, snd):
    diff = np.array(fst) - np.array(snd)
    print("fst mean", np.mean(fst))
    print("snd mean", np.mean(snd))
    diff = diff[diff != 0]  # remove zero differences

    abs_diff = np.abs(diff)
    ranks = rankdata(abs_diff)
    signed_ranks = ranks.copy()
    signed_ranks[diff < 0] = -signed_ranks[diff < 0]

    positive_sum = signed_ranks[signed_ranks > 0].sum()
    negative_sum = signed_ranks[signed_ranks < 0].sum()

    print(f"Positive sum: {positive_sum}. Negative sum: {negative_sum}")
    print("Result", min(abs(positive_sum), abs(negative_sum)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run statistical tests")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the CSV files containing the scores",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    ms_pcqe = f"{parser.base_path}/ms-pcqe_coefficients.csv"
    mm_pcqa = f"{parser.base_path}/mm-pcqa_coefficients.csv"

    ms_pcqe = read_csv_file(ms_pcqe)
    mm_pcqa = read_csv_file(mm_pcqa)

    manual_wilcoxon_signed_rank(ms_pcqe, mm_pcqa)
    wilcoxon_signed_rank(ms_pcqe, mm_pcqa)
