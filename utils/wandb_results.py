import wandb
import pandas as pd


def get_coefficients_table(entity, metric, id):
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{metric}/run-{id}-Coefficients_table:v0")
    table = artifact.get("Coefficients_table")  # replace with actual table name
    df = pd.DataFrame(table.data, columns=table.columns)
    df.name = metric
    return df


def extract_filename(key):
    # Split the string by '-' and re-join from the third part onward
    parts = key.split("-")
    if len(parts) < 3:
        return None  # or raise an error if the format is unexpected
    remainder = "-".join(parts[2:])

    # Find the .png part and slice accordingly
    if ".png" in remainder:
        end_index = remainder.find(".png") + 4  # include '.png'
        return remainder[:end_index]
    return None


def get_scores_table(entity, metric, id):
    api = wandb.Api()
    tables = []

    run = api.run(f"{entity}/{metric}/{id}")
    for artifact in run.logged_artifacts():
        if not artifact.name.endswith(".png:v0"):
            print("Skipping", artifact.name)
            continue

        # Use the artifact (downloads it if needed)
        # used_artifact = api.artifact(f"{entity}/{metric}/{artifact.name}:v0")
        used_artifact = api.artifact(f"{entity}/{metric}/{artifact.name}")

        table_name = extract_filename(artifact.name)
        table = used_artifact.get(table_name)

        # Convert to DataFrame and collect
        df = pd.DataFrame(table.data, columns=table.columns)
        tables.append(df)

    combined_df = pd.concat(tables, ignore_index=True)
    combined_df.name = metric
    return combined_df
