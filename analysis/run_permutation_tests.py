import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau
from statsmodels.stats.multitest import multipletests
import argparse
import yaml


def control_multiple_hypotheses(p_vals):
    _, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
    return pvals_corrected


def run_permutation_test(
    n_permutations, observed_srcc, observed_krcc, higher_is_better
):

    if higher_is_better:
        y_true = [100 - i for i in range(100)]
    else:
        y_true = [i for i in range(100)]
    permuted_mean_srccs = []
    permuted_mean_krccs = []
    num_sets = 60

    for _ in range(n_permutations):
        permuted_srccs = []
        permuted_krccs = []
        for _ in range(num_sets):
            y_true_perm = np.random.permutation(y_true)

            srcc, _ = spearmanr(y_true_perm, y_true)
            krcc, _ = kendalltau(y_true_perm, y_true)
            permuted_srccs.append(srcc)
            permuted_krccs.append(krcc)
        permuted_mean_srccs.append(np.mean(permuted_srccs))
        permuted_mean_krccs.append(np.mean(permuted_krccs))

    permuted_mean_srccs = np.array(permuted_mean_srccs)
    permuted_mean_krccs = np.array(permuted_mean_krccs)
    srcc_p_value = np.mean(np.abs(permuted_mean_srccs) >= np.abs(observed_srcc))
    krcc_p_value = np.mean(np.abs(permuted_mean_krccs) >= np.abs(observed_krcc))

    return max(srcc_p_value, 1 / (n_permutations + 1)), max(
        krcc_p_value, 1 / (n_permutations + 1)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Permutation tests")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the YAML file containing info about mean SRCC and KRCC values",
    )
    return parser.parse_args()


def read_yaml_file(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


if __name__ == "__main__":
    args = parse_args()
    yaml_data = read_yaml_file(args.config)
    mean_srccs = sorted(yaml_data["observed_mean_srccs"])
    mean_krccs = sorted(yaml_data["observed_mean_krccs"])
    assert len(mean_srccs) == len(mean_krccs)
    uncorrected_p_srcc = [0.0] * len(mean_srccs)
    uncorrected_p_krcc = [0.0] * len(mean_srccs)
    n_permutations = yaml_data["n_permutations"]
    higher_is_better = yaml_data["higher_is_better"]

    for i in range(len(mean_srccs)):
        srcc_pair = mean_srccs[i]
        krcc_pair = mean_krccs[i]

        assert srcc_pair[0] == krcc_pair[0]
        metric = srcc_pair[0]
        print("Running metric", metric)
        srcc_p, krcc_p = run_permutation_test(
            n_permutations, srcc_pair[1], krcc_pair[1], higher_is_better[metric]
        )
        uncorrected_p_srcc[i] = srcc_p
        uncorrected_p_krcc[i] = krcc_p
    corrected_p_srcc = control_multiple_hypotheses(uncorrected_p_srcc)
    corrected_p_krcc = control_multiple_hypotheses(uncorrected_p_krcc)

    for i in range(len(mean_srccs)):
        print(f"Metric: {mean_srccs[i][0]}")
        print(f"Corrected SRCC_p: {max(corrected_p_srcc[i], 1/n_permutations)}")
        print(f"Corrected KRCC_p: {max(corrected_p_krcc[i], 1/n_permutations)}\n\n")
