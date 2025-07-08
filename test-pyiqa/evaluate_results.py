import json
from scipy.stats import spearmanr, kendalltau
import wandb
import numpy as np
import os
import argparse
from evaluate_results_property_tests import (
    test_log_location_wise,
    test_log_weather_wise,
    test_mean_munkkivuori,
    test_mean_otaniemi,
)
from utils import filter_by_key, filter_entries
import yaml
import pandas as pd


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def log(msg, bcolor_type):
    print(f"{bcolor_type}{msg}{bcolors.ENDC}")


def ground_truth(metric):
    if lower_better(metric):
        ground_truth = [i for i in range(100)]
    else:
        ground_truth = [100 - i for i in range(100)]
    return ground_truth


def lower_better(metric):
    lower_better_dict = {
        "ilniqe": True,
        "qalign": False,
        "qualiclip": False,
        "topiq_nr": False,
        "dbcnn": False,
    }
    return lower_better_dict[metric]


def read_json(path):
    with open(path, "r") as f:
        first_results = json.load(f)
    return first_results


def calculate_coeffs_df(metric, scores_df):
    rows = []
    columns = [f"{i}.png" for i in range(100)]
    true_values = ground_truth(metric)
    for _, row in scores_df.iterrows():
        pred_scores = row[columns].tolist()
        srcc, srcc_p = spearmanr(true_values, pred_scores)
        krcc, krcc_p = kendalltau(true_values, pred_scores)
        rows.append(
            {
                "Location": row["Location"],
                "Weather": row["Weather"],
                "Reference image": row["Reference image"],
                "SRCC": srcc,
                "SRCC_p": srcc_p,
                "KRCC": krcc,
                "KRCC_p": krcc_p,
            }
        )
    return pd.DataFrame(rows)


def nested_dict_to_df(results_dict):
    rows = []
    for location, location_data in results_dict.items():
        for weather, weather_data in location_data.items():
            for ref_image, ref_img_data in weather_data.items():
                row = {
                    "Location": location,
                    "Weather": weather,
                    "Reference image": ref_image,
                }
                row.update(ref_img_data)  # Each key is like "0.png": score
                rows.append(row)
    return pd.DataFrame(rows)


def log_mean_std(prefix, srcc_values, krcc_values):
    log(prefix, bcolor_type=bcolors.HEADER)
    print(f"{prefix}. SRCC mean: {np.mean(srcc_values)}")
    print(f"{prefix}. SRCC STD: {np.std(srcc_values)}")

    print(f"{prefix}. KRCC mean: {np.mean(krcc_values)}")
    print(f"{prefix}. KRCC STD: {np.std(krcc_values)}")


def log_srcc_krcc_combined_fog(df):
    log("\n\n\nResults using both foggy_1 and foggy_2", bcolor_type=bcolors.HEADER)

    groups = [
        ("All values", df),
        ("Otaniemi", df[df["Location"] == "Otaniemi"]),
        ("Munkkivuori", df[df["Location"] == "Munkkivuori"]),
        (
            "Foggy both locations",
            df[(df["Weather"] == "foggy_1") | (df["Weather"] == "foggy_2")],
        ),
        ("Rainy both locations", df[df["Weather"] == "rainy"]),
        (
            "Foggy Otaniemi",
            df[
                ((df["Weather"] == "foggy_1") | (df["Weather"] == "foggy_2"))
                & (df["Location"] == "Otaniemi")
            ],
        ),
        (
            "Foggy Munkkivuori",
            df[
                ((df["Weather"] == "foggy_1") | (df["Weather"] == "foggy_2"))
                & (df["Location"] == "Munkkivuori")
            ],
        ),
        (
            "Rainy Otaniemi",
            df[(df["Weather"] == "rainy") & (df["Location"] == "Otaniemi")],
        ),
        (
            "Rainy Munkkivuori",
            df[(df["Weather"] == "rainy") & (df["Location"] == "Munkkivuori")],
        ),
        (
            "Foggy_1 Otaniemi",
            df[(df["Weather"] == "foggy_1") & (df["Location"] == "Otaniemi")],
        ),
        (
            "Foggy_1 Munkkivuori",
            df[(df["Weather"] == "foggy_1") & (df["Location"] == "Munkkivuori")],
        ),
        (
            "Foggy_2 Otaniemi",
            df[(df["Weather"] == "foggy_2") & (df["Location"] == "Otaniemi")],
        ),
        (
            "Foggy_2 Munkkivuori",
            df[(df["Weather"] == "foggy_2") & (df["Location"] == "Munkkivuori")],
        ),
    ]
    for prefix, subset in groups:
        log_mean_std(prefix, subset["SRCC"].tolist(), subset["KRCC"].tolist())


def log_srcc_krcc_one_fogginess(df):
    log("\n\n\nResults using only foggy_1", bcolor_type=bcolors.HEADER)
    df = df[df["Weather"] != "foggy_2"]

    groups = [
        ("All values", df),
        ("Otaniemi", df[df["Location"] == "Otaniemi"]),
        ("Munkkivuori", df[df["Location"] == "Munkkivuori"]),
        ("Foggy both locations", df[df["Weather"] == "foggy_1"]),
        ("Rainy both locations", df[df["Weather"] == "rainy"]),
        (
            "Foggy Otaniemi",
            df[(df["Weather"] == "foggy_1") & (df["Location"] == "Otaniemi")],
        ),
        (
            "Foggy Munkkivuori",
            df[(df["Weather"] == "foggy_1") & (df["Location"] == "Munkkivuori")],
        ),
        (
            "Rainy Otaniemi",
            df[(df["Weather"] == "rainy") & (df["Location"] == "Otaniemi")],
        ),
        (
            "Rainy Munkkivuori",
            df[(df["Weather"] == "rainy") & (df["Location"] == "Munkkivuori")],
        ),
    ]
    for prefix, subset in groups:
        log_mean_std(prefix, subset["SRCC"].tolist(), subset["KRCC"].tolist())


def log_metric(metric, results_path):
    results = read_json(results_path)
    scores_df = nested_dict_to_df(results)
    coeff_df = calculate_coeffs_df(metric, scores_df)
    log_srcc_krcc_one_fogginess(coeff_df)
    log_srcc_krcc_combined_fog(coeff_df)

    scores_table = wandb.Table(dataframe=scores_df)
    wandb.log({"Scores_table": scores_table})
    coeff_table = wandb.Table(dataframe=coeff_df)
    wandb.log({"Coefficients_table": coeff_table})
    wandb.finish()


def evaluate_metric(metric, base_path, json_name):
    wandb.login()
    wandb.init(project=metric, name=f"Final analysis", resume=False)

    full_path = os.path.join(base_path, json_name)
    print(f"Evaluating metric: {metric}")
    print(f"Using scores from file: {full_path}")

    log_metric(metric, full_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate results script")

    parser.add_argument(
        "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing NR-IQA information",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.c, "r") as f:
        config = yaml.safe_load(f)
    doubles = [tuple(t) for t in config["doubles"]]

    for metric, json_name in doubles:
        evaluate_metric(metric, config["base_path"], json_name)
