import wandb
import pandas as pd
import argparse
from dotenv import load_dotenv
import os
from wandb_results import get_coefficients_table
import plotly.express as px
import numpy as np
import pandas as pd
import numpy as np
import plotly.express as px


def generate_coefficients_figure(dfs, output_path, one_fogginess):

    # Combine all dataframes and assign a name column
    df_all = pd.concat(
        [df.assign(metric_name=df.name) for df in dfs], ignore_index=True
    )
    if one_fogginess:
        df_all = df_all[~(df_all["weather"] == "foggy_0.5_0.9")]
        # Rename fog variants to readable labels
        weather_rename_map = {
            "foggy_0.7_0.0_1.0": "foggy",
        }
        # Define desired legend order
        desired_order = [
            "Otaniemi-rainy",
            "Otaniemi-foggy",
            "Munkkivuori-rainy",
            "Munkkivuori-foggy",
        ]
    else:
        weather_rename_map = {
            "foggy_0.7_0.0_1.0": "foggy¹",
            "foggy_0.5_0.9": "foggy²",
        }
        desired_order = [
            "Otaniemi-rainy",
            "Otaniemi-foggy¹",
            "Otaniemi-foggy²",
            "Munkkivuori-rainy",
            "Munkkivuori-foggy¹",
            "Munkkivuori-foggy²",
        ]
    df_all["weather"] = df_all["weather"].replace(weather_rename_map)

    # Create full setup for legend and hover info
    df_all["full_setup"] = df_all["location"] + "-" + df_all["weather"]
    df_all["setup"] = df_all["full_setup"]

    df_all["setup"] = pd.Categorical(
        df_all["setup"], categories=desired_order, ordered=True
    )
    df_all = df_all.sort_values("setup")

    # Map metric names to x positions and add jitter
    unique_names = df_all["metric_name"].unique()
    name_to_x = {name: i for i, name in enumerate(unique_names)}
    df_all["x_base"] = df_all["metric_name"].map(name_to_x)
    df_all["x_jitter"] = df_all["x_base"] + np.random.uniform(
        -0.2, 0.2, size=len(df_all)
    )

    # Create scatter plot
    fig = px.scatter(
        df_all,
        x="x_jitter",
        y="SRCC",
        color="setup",
        symbol_sequence=["circle"],
        hover_data=["full_setup", "prediction_key", "metric_name", "SRCC"],
    )

    # Increase marker size
    fig.update_traces(marker=dict(size=10))

    # Update layout: remove title, move legend, increase text
    fig.update_layout(
        title={
            "text": "SRCC values across location and distortion type",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(
            tickmode="array",
            tickvals=list(name_to_x.values()),
            ticktext=list(name_to_x.keys()),
            title="NR-IQA metric",
        ),
        yaxis_title="SRCC",
        legend_title="Setup",
        width=800,
        height=700,
        font=dict(size=20),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=18)
        ),
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        margin=dict(l=5, r=5, t=45, b=60),
    )

    fig.show()
    if output_path:
        fig.write_image(output_path, scale=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots script")

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        default=None,
        help="Path to the env file containing wandb info",
    )

    parser.add_argument(
        "--output_path", type=str, required=False, help="Path to the output image file"
    )

    parser.add_argument(
        "--one_fogginess",
        type=str,
        required=False,
        default=False,
        help="Parameter deciding if only foggy^1=foggy_0.0_1.0_0.7 should be used",
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

    generate_coefficients_figure(coeff_dfs, args.output_path, args.one_fogginess)
