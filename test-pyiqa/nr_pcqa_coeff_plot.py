from scipy.stats import spearmanr, kendalltau
import wandb
import pandas as pd
from pathlib import Path
import plotly.express as px
import numpy as np
import argparse
from wandb_results import get_coefficients_table
from dotenv import load_dotenv
import os


def nr_pcqa_coefficients_figure(df_coeffs, output_path):
    # Combine all dataframes and assign a name column
    df_all = pd.concat(
        [df.assign(metric_name=df.name) for df in df_coeffs], ignore_index=True
    )

    # Show what actual Weather and Time values are
    print("Unique Weather values:", df_all["Weather"].unique())
    print("Unique Time values:", df_all["Time"].unique())

    # Create jittered x-position by mapping metric_name to a numeric x and adding noise
    unique_names = df_all["metric_name"].unique()
    name_to_x = {name: i for i, name in enumerate(unique_names)}
    df_all["x_base"] = df_all["metric_name"].map(name_to_x)
    df_all["x_jitter"] = df_all["x_base"] + np.random.uniform(
        -0.1, 0.1, size=len(df_all)
    )

    # Normalize weather names (adjust this mapping as needed based on print output)
    df_all["Weather_normalized"] = (
        df_all["Weather"]
        .str.lower()
        .replace(
            {
                "clear_weather": "clear",
                "rainy_weather": "rainy",
                "rain": "rainy",
                "foggy_weather": "foggy",
            }
        )
    )

    # Combine weather and time
    df_all["WeatherTime"] = (
        df_all["Weather_normalized"] + ", " + df_all["Time"].str.lower()
    )

    # Keep only the four target combinations
    valid_combinations = ["clear, day", "clear, night", "rainy, day", "rainy, night"]
    df_all = df_all[df_all["WeatherTime"].isin(valid_combinations)]

    # Create the jitter plot using only circles
    fig = px.scatter(
        df_all,
        x="x_jitter",
        y="SRCC",
        color="WeatherTime",
        hover_data=["File_name", "metric_name", "SRCC"],
    )

    fig.update_traces(
        marker_symbol="circle", marker=dict(size=10)
    )  # Force all markers to be circles

    fig.update_layout(
        title={
            "text": "SRCC values across weather and time conditions",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(
            tickmode="array",
            tickvals=list(name_to_x.values()),
            ticktext=list(name_to_x.keys()),
            title="NR-PCQA metric metric",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(dotenv_path=args.env)
    mm_pcqa_id = os.getenv("MM_PCQA_ANALYSIS")
    ms_pcqe_id = os.getenv("MS_PCQE_ANALYSIS")
    entity = os.getenv("ENTITY")
    api = wandb.Api()

    analysis_pairs = [
        ("mm-pcqa", mm_pcqa_id),
        ("ms-pcqe", ms_pcqe_id),
    ]
    coeff_dfs = []
    for pair in analysis_pairs:
        coeff_dfs.append(get_coefficients_table(entity, pair[0], pair[1]))

    nr_pcqa_coefficients_figure(coeff_dfs, args.output_path)
