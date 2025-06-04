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


def log_srcc_krcc(id_base, label, entries):
    log_stats(f"{id_base}", f"SRCC of {label}", filter_by_key(entries, "SRCC"))
    log_stats(f"{int(id_base) + 1}", f"KRCC of {label}", filter_by_key(entries, "KRCC"))


def log_stats(num_prefix, prefix, values):
    if values:
        mean = np.mean(values)
        std = np.std(values)
        print(f"{prefix}: {mean}")
        print(f"STD {prefix}: {std}")
        wandb.summary[f"{num_prefix}_{prefix}"] = mean
        wandb.summary[f"{num_prefix}_STD {prefix}"] = std
    else:
        print("ERROR: Values not found.")


def reformat_dict(d):
    """
    orig_dict: Dictionary containing all rainy results and half the foggy ones.
    The keys in orig_dict have to be changed as they do not specify the
    exact parameters of fogginess.
    """
    d["Otaniemi"]["foggy_0.7_0.0_1.0"] = d["Otaniemi"]["foggy"]
    d["Otaniemi"].pop("foggy")

    d["Munkkivuori"]["foggy_0.5_0.9"] = d["Munkkivuori"]["foggy"]
    d["Munkkivuori"].pop("foggy")
    return d


def get_alpha_coef_fog_intensity(s):
    if s == "foggy_0.7_0.0_1.0":
        fog_intensity = "0.7"
        alpha_coef = "[0.0, 1.0]"
    elif s == "foggy_0.5_0.9":
        fog_intensity = "[0.5, 0.9]"
        alpha_coef = "[0.5, 0.9]"
    else:
        fog_intensity = "Not applicable"
        alpha_coef = "Not applicable"
    return {"fog_intensity": fog_intensity, "alpha_coef": alpha_coef}


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


def calculate_srcc_krcc(metric, first_results_path, second_results_path):
    """
    first_results_path: Path to the results of the first execution with
    all rainy results and half of the foggy results. Here, the Munkkivuori
    images were distorted by letting fog intensity and alpha_coef vary from
    0.5 to 0.9. Otaniemi had fog_intensity of 0.7 and alpha_coef changed from
    0.0 to 1.0.

    second_results_path: Path to the results of the second execution with half of the
    foggy results. Munkkivuori has fog_intensity of 0.7 and alpha_coef varies
    from 0.0 to 1.0. Otaniemi had fog_intensity and alpha_coef vary from
    0.5 to 0.9.
    """

    first_results = reformat_dict(read_json(first_results_path))
    second_results = read_json(second_results_path)

    results_files = [first_results, second_results]

    # If lower is better then assign 0.png the highest score and 100.png the lowest.
    # Otherwise, do the opposite.
    if lower_better(metric):
        ground_truth = [i for i in range(100)]
    else:
        ground_truth = [100 - i for i in range(100)]

    results_table = wandb.Table(
        columns=[
            "location",
            "weather",
            "alpha_coef",
            "fog_intensity",
            "prediction_key",
            "SRCC",
            "SRCC p-value",
            "KRCC",
            "KRCC p-value",
        ]
    )
    results = []
    for result in results_files:
        for location, weather_dict in result.items():

            for weather, predictions in weather_dict.items():
                for key, score_dict in predictions.items():
                    pred_scores = [score_dict[f"{i}.png"] for i in range(100)]
                    srcc, srcc_p = spearmanr(ground_truth, pred_scores)
                    krcc, krcc_p = kendalltau(ground_truth, pred_scores)

                    coeffs = get_alpha_coef_fog_intensity(weather)

                    results.append(
                        {
                            "location": location,
                            "weather": weather,
                            "alpha_coef": coeffs["alpha_coef"],
                            "fog_intensity": coeffs["fog_intensity"],
                            "prediction_key": key,
                            "SRCC": srcc,
                            "SRCC p-value": srcc_p,
                            "KRCC": krcc,
                            "KRCC p-value": krcc_p,
                        }
                    )
                    # print(f"Logging: {location}, {weather}, {key}, alpha_coef={coeffs['alpha_coef']}, fog_intensity={coeffs['fog_intensity']} SRCC={srcc}, KRCC={krcc} SRCC p-value: {srcc_p}")
                    results_table.add_data(
                        str(location),
                        str(weather),
                        str(coeffs["alpha_coef"]),
                        str(coeffs["fog_intensity"]),
                        str(key),
                        float(srcc),
                        float(srcc_p),
                        float(krcc),
                        float(krcc_p),
                    )
    wandb.log({"Coefficients_table": results_table})
    return results


def log_results(results):
    test_mean_otaniemi(results)
    test_log_location_wise(results)
    test_mean_munkkivuori(results)
    test_log_weather_wise(results)

    log_combined_results(results)
    log_location_wise(results)
    log_weather_wise(results)
    log_specific_munkkivuori(results)
    log_specific_otaniemi(results)
    log_munkkivuori_fst_fog(results)
    log_otaniemi_fog_fst(results)
    log_munkkivuori_snd_fog(results)
    log_otaniemi_snd_fog(results)


def log_combined_results(results):
    log("\n\nCombined results", bcolors.HEADER)
    log_srcc_krcc("0", "Combined", results)


def log_location_wise(results):
    log("\n\nResults of Otaniemi only", bcolors.HEADER)
    otaniemi = filter_entries(results, location="Otaniemi")
    log_srcc_krcc("2", "Otaniemi", otaniemi)

    log("\n\nResults of Munkkivuori only", bcolors.HEADER)
    munkkivuori = filter_entries(results, location="Munkkivuori")
    log_srcc_krcc("4", "Munkkivuori", munkkivuori)


def log_weather_wise(results):

    log("\n\nResults of rainy images on both datasets", bcolors.HEADER)
    all_rainy = filter_entries(results, weather="rainy")
    log_srcc_krcc("6", "Rainy images on both sites", all_rainy)

    log("\n\nResults of foggy images on both datasets", bcolors.HEADER)
    all_foggy = filter_entries(results, weather="foggy_0.5_0.9") + filter_entries(
        results, weather="foggy_0.7_0.0_1.0"
    )
    log_srcc_krcc("8", "All foggy images", all_foggy)


def log_specific_otaniemi(results):
    otaniemi = filter_entries(results, location="Otaniemi")
    log("\n\nResults on rainy Otaniemi images", bcolors.HEADER)
    otaniemi_rainy = filter_entries(otaniemi, weather="rainy")
    log_srcc_krcc("12", "Rainy Otaniemi images", otaniemi_rainy)

    log("\n\nResults on foggy Otaniemi images", bcolors.HEADER)
    otaniemi = filter_entries(results, location="Otaniemi")
    otaniemi_foggy = filter_entries(otaniemi, weather="foggy_0.5_0.9") + filter_entries(
        otaniemi, weather="foggy_0.7_0.0_1.0"
    )
    log_srcc_krcc("10", "Foggy Otaniemi images", otaniemi_foggy)


def log_specific_munkkivuori(results):
    munkkivuori = filter_entries(results, location="Munkkivuori")

    log("\n\nResults on foggy Munkkivuori images", bcolors.HEADER)

    munkkivuori_foggy = filter_entries(
        munkkivuori, weather="foggy_0.5_0.9"
    ) + filter_entries(munkkivuori, weather="foggy_0.7_0.0_1.0")
    log_srcc_krcc("14", "Foggy Munkkivuori images", munkkivuori_foggy)

    log("\n\nResults on rainy Munkkivuori images", bcolors.HEADER)
    munkkivuori_rainy = filter_entries(munkkivuori, weather="rainy")
    log_srcc_krcc("16", "Rainy Munkkivuori images", munkkivuori_rainy)


def log_munkkivuori_fst_fog(results):
    munkkivuori = filter_entries(results, location="Munkkivuori")

    log(
        "\n\nResults on foggy Munkkivuori images with alpha_coef=[0.0, 1.0], fog_intensity=0.7",
        bcolors.HEADER,
    )
    munkkivuori_foggy_fst = filter_entries(munkkivuori, weather="foggy_0.7_0.0_1.0")
    log_srcc_krcc(
        "18",
        "Foggy Munkkivuori images (alpha_coef=[0.0, 1.0], fog_intensity=0.7)",
        munkkivuori_foggy_fst,
    )


def log_otaniemi_fog_fst(results):
    otaniemi = filter_entries(results, location="Otaniemi")
    log(
        "\n\nResults on foggy Otaniemi images with alpha_coef=[0.0, 1.0], fog_intensity=0.7 ",
        bcolors.HEADER,
    )
    otaniemi_foggy_fst = filter_entries(otaniemi, weather="foggy_0.7_0.0_1.0")
    log_srcc_krcc(
        "20",
        "Foggy Otaniemi images (alpha_coef=[0.0, 1.0], fog_intensity=0.7)",
        otaniemi_foggy_fst,
    )


def log_munkkivuori_snd_fog(results):
    munkkivuori = filter_entries(results, location="Munkkivuori")

    log(
        "\n\nResults on foggy Munkkivuori images with alpha_coef=[0.5, 0.9], fog_intensity=[0.5, 0.9]",
        bcolors.HEADER,
    )
    munkkivuori_foggy_snd = filter_entries(munkkivuori, weather="foggy_0.5_0.9")
    log_srcc_krcc(
        "23",
        "Foggy Munkkivuori images (alpha_coef=[0.5, 0.9], fog_intensity=[0.5, 0.9])",
        munkkivuori_foggy_snd,
    )


def log_otaniemi_snd_fog(results):
    otaniemi = filter_entries(results, location="Otaniemi")
    log(
        "\n\nResults on foggy Otaniemi images with alpha_coef=[0.5, 0.9], fog_intensity=[0.5, 0.9]",
        bcolors.HEADER,
    )
    otaniemi_foggy_snd = filter_entries(otaniemi, weather="foggy_0.5_0.9")
    log_srcc_krcc(
        "24",
        "Foggy Otaniemi images (alpha_coef=[0.5, 0.9], fog_intensity=[0.5, 0.9])",
        otaniemi_foggy_snd,
    )


def evaluate_metric(metric, base_path, first_file, second_file):
    wandb.login()
    wandb.init(project=metric, name=f"Combined analysis", resume=False)

    first_full_path = os.path.join(base_path, first_file)
    second_full_path = os.path.join(base_path, second_file)
    print(f"Evaluating metric: {metric}")
    print(f"Using scores from file: {first_full_path}")
    print(f"Using scores from file: {second_full_path}")

    results = calculate_srcc_krcc(metric, first_full_path, second_full_path)

    log_results(results)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate results script")

    parser.add_argument(
        "--c",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing NR-IQA information",
    )
    args = parser.parse_args()

    with open(args.c, "r") as f:
        config = yaml.safe_load(f)

    triplets = [tuple(t) for t in config["triplets"]]

    for pair in triplets:
        evaluate_metric(
            pair[0],
            config["base_path"],
            pair[1],
            pair[2],
        )
