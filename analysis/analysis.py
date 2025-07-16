import argparse
import wandb
import pandas as pd
import shutil
import json
from pathlib import Path
import re
from utils.utils import log, bcolors
from scipy.stats import spearmanr, kendalltau
from utils.utils import load_yaml, write_to_json


def clean_name(artifact):
    """
    Artifacts names are of type: run-xxxxx-name_of_table:version_number
    and here we only want name_of_table
    """
    match = re.search(r"run-[^-]+-(.*):", artifact.name)
    if match:
        file_name = match.group(1)
    else:
        log("No match found for the file name", bcolor_type=bcolors.FAIL)
        raise ValueError("No match found for the file name")
    return file_name


def ground_truth(lower_better, n):
    if lower_better:
        ground_truth = [i for i in range(n)]
    else:
        ground_truth = [n - i for i in range(n)]
    return ground_truth


def artifact_table_to_df(artifact):
    print(f"{artifact.name=} {artifact.type=}")
    path = Path(artifact.download())

    clean_artifact = clean_name(artifact)
    clean_artifact += ".table.json"

    with open(path / clean_artifact, "r") as f:
        table_data = json.load(f)

    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
    shutil.rmtree(path)
    return df


def load_all_tables(run):
    tables = {}
    for artifact in run.logged_artifacts():
        if artifact.type == "run_table":
            tables[artifact.name] = artifact_table_to_df(artifact)
    return tables


def calculate_SRCC_KRCC(tables, lower_better):
    scores = {}
    for key, df in tables.items():
        log(f"Scoring table {key}", bcolor_type=bcolors.OKBLUE)
        num_scores = df.shape[0]
        true_values = ground_truth(lower_better, num_scores)
        pred_scores = df["score"].tolist()
        srcc, srcc_p = spearmanr(true_values, pred_scores)
        krcc, krcc_p = kendalltau(true_values, pred_scores)
        scores[key] = {"srcc": srcc, "srcc_p": srcc_p, "krcc": krcc, "krcc_p": krcc_p}
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="pyiqa runnner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to where the config .yaml file is located",
    )
    args = parser.parse_args()
    return args


def main():
    api = wandb.Api()
    arguments = parse_args()
    config = load_yaml(arguments.config)
    entity = config["entity"]
    project = config["project"]
    run_id = config["run_id"]
    output_path = config["output_path"]

    run = api.run(f"{entity}/{project}/{run_id}")

    tables = load_all_tables(run)
    scores = calculate_SRCC_KRCC(tables, False)
    log(f"Logging scores to {output_path}", bcolor_type=bcolors.OKBLUE)
    write_to_json(output_path, scores)


if __name__ == "__main__":
    main()
