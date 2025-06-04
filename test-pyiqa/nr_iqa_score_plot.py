import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import json
import argparse
import os


def retrieve_data(fst_execution_path, snd_execution_path):
    """
    First experiment:
    Munkkivuori = 0.5_0.9
    Otaniemi = 0.7_0.0_1.0

    Second experiment:
    Munkkivuori = 0.7_0.0_1.0 or foggy¹
    Otaniemi =  0.5_0.9 or foggy²
    """
    with open(fst_execution_path, "r") as f:
        fst_execution = json.load(f)
    fst_execution["Otaniemi"]["foggy¹"] = fst_execution["Otaniemi"].pop("foggy")
    fst_execution["Munkkivuori"]["foggy²"] = fst_execution["Munkkivuori"].pop("foggy")

    with open(snd_execution_path, "r") as f:
        snd_execution = json.load(f)

    fst_execution["Otaniemi"]["foggy²"] = snd_execution["Otaniemi"].pop("foggy_0.5_0.9")
    fst_execution["Munkkivuori"]["foggy¹"] = snd_execution["Munkkivuori"].pop(
        "foggy_0.7_0.0_1.0"
    )
    return fst_execution


def construct_df(data):
    rows = []
    for location, weathers in data.items():
        for weather, image_groups in weathers.items():
            for (
                group_filename,
                image_scores,
            ) in image_groups.items():  # group_filename = "1702455546884733518.png"
                for image_file, score in image_scores.items():
                    image_index = int(image_file.replace(".png", ""))
                    rows.append(
                        {
                            "Location-weather": f"{location}-{weather}",
                            "image_index": image_index,
                            "score": score,
                            "source_filename": group_filename,  # <- Add this for hover
                        }
                    )

    return pd.DataFrame(rows)


import plotly.express as px


def generate_violin_plot(metric, data, output_dir, one_fogginess, every_other):
    df = construct_df(data)

    if one_fogginess:
        df = df[~(df["Location-weather"] == "Munkkivuori-foggy²")]
        df = df[~(df["Location-weather"] == "Otaniemi-foggy²")]
        df["Location-weather"] = df["Location-weather"].replace(
            "Munkkivuori-foggy¹", "Munkkivuori-foggy"
        )
        df["Location-weather"] = df["Location-weather"].replace(
            "Otaniemi-foggy¹", "Otaniemi-foggy"
        )

    df = df[df["image_index"] % every_other == 0]

    # Create violin plot
    fig = px.violin(
        df,
        x="image_index",
        y="score",
        color="Location-weather",
        box=True,  # Show embedded box plot
        points="outliers",  # Show individual points for outliers
        hover_data=["source_filename"],
        title=f"{metric} Score Distribution per Image Index (Violin Plot)",
    )

    fig.update_layout(
        xaxis_title="Image Index",
        yaxis_title="Score",
        xaxis=dict(tickmode="linear", dtick=2, range=[0, 99]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            itemwidth=80,
            itemsizing="constant",
            tracegroupgap=5,
        ),
        title=dict(
            text=f"{metric_to_metric(metric)} Score Distribution per Image Index (Violin Plot)",
            font=dict(size=20),
        ),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=16),
        legend_font=dict(size=16),
        legend_title=None,
        margin=dict(l=10, r=10, t=60, b=100),
        height=600,
    )

    if output_dir:
        fig.write_image(
            f"{output_dir}/{metric}_scores_A100_violinplot.pdf",
            scale=3,
        )


def generate_box_plot(metric, data, output_dir, one_fogginess, every_other):
    df = construct_df(data)
    if one_fogginess:
        df = df[~(df["Location-weather"] == "Munkkivuori-foggy²")]
        df = df[~(df["Location-weather"] == "Otaniemi-foggy²")]
        df["Location-weather"] = df["Location-weather"].replace(
            "Munkkivuori-foggy¹", "Munkkivuori-foggy"
        )
        df["Location-weather"] = df["Location-weather"].replace(
            "Otaniemi-foggy¹", "Otaniemi-foggy"
        )

    df = df[df["image_index"] % every_other == 0]

    # Create box plot with image_index on x-axis and scores grouped by Location-weather
    fig = px.box(
        df,
        x="image_index",
        y="score",
        color="Location-weather",
        points="outliers",  # or "all" if you want to see every point
        hover_data=["source_filename"],
        title=f"{metric} Score Distribution per Image Index",
    )

    fig.update_layout(
        xaxis_title="Image Index",
        yaxis_title="Score",
        xaxis=dict(tickmode="linear", dtick=5, range=[-2, 99]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            itemwidth=80,
            itemsizing="constant",
            tracegroupgap=5,
        ),
        title=dict(
            text=f"{metric_to_metric(metric)} Score Distribution per Image Index",
            font=dict(size=20),
        ),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=16),
        legend_font=dict(size=16),
        legend_title=None,
        margin=dict(l=10, r=10, t=60, b=100),
        height=600,
    )

    if output_dir:
        fig.write_image(f"{output_dir}/{metric}_scores_A100_boxplot.pdf", scale=3)


def generate_score_plot(metric, data, output_dir, one_fogginess, every_other):
    df = construct_df(data)
    if one_fogginess:
        df = df[~(df["Location-weather"] == "Munkkivuori-foggy²")]
        df = df[~(df["Location-weather"] == "Otaniemi-foggy²")]
        df["Location-weather"] = df["Location-weather"].replace(
            "Munkkivuori-foggy¹", "Munkkivuori-foggy"
        )
        df["Location-weather"] = df["Location-weather"].replace(
            "Otaniemi-foggy¹", "Otaniemi-foggy"
        )

    fig = px.scatter(
        df,
        x="image_index",
        y="score",
        color="Location-weather",
        title=f"{metric} Score vs Image Index",
        hover_data=["source_filename"],  # <- Show the group filename in tooltip
    )

    # Rest of the layout and style remain unchanged
    fig.update_layout(
        xaxis_title="Image Index",
        yaxis_title="Score",
        xaxis=dict(tickmode="linear", dtick=5, range=[0, 99]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            itemwidth=80,
            itemsizing="constant",
            tracegroupgap=5,
        ),
        title=dict(
            text=f"{metric_to_metric(metric)} Score vs Image Index",
            font=dict(size=20),
        ),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=16),
        yaxis_tickfont=dict(size=16),
        legend_font=dict(size=16),
        legend_title=None,
        margin=dict(l=10, r=10, t=60, b=100),
        height=600,
    )

    fig.update_traces(marker=dict(size=5))

    if output_dir:
        fig.write_image(f"{output_dir}/{metric}_scores_A100_plot.pdf", scale=3)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots script")

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to be plotted",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base directory containing JSON data"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Path to output directory",
    )

    parser.add_argument(
        "--one_fogginess",
        required=False,
        action="store_true",
        help="Parameter deciding if only foggy^1=foggy_0.0_1.0_0.7 should be used",
    )

    parser.add_argument(
        "--every_other",
        type=int,
        required=False,
        default=1,
        help="Only plot every other image index",
    )
    return parser.parse_args()


def metric_to_metric(metric):
    metric_names = {
        "dbcnn": "DB-CNN",
        "qalign": "Q-Align",
        "topiq_nr": "TOPIQ",
        "ilniqe": "IL-NIQE",
        "qualiclip": "QualiCLIP",
    }
    return metric_names[metric]


if __name__ == "__main__":
    args = parse_args()
    first = f"{args.base_path}/first_execution/{args.metric}_A100_results.json"
    second = f"{args.base_path}/second_execution/{args.metric}_A100_results.json"
    combined = retrieve_data(first, second)

    metrics = ["qalign", "ilniqe", "topiq_nr", "dbcnn", "qualiclip"]
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for metric in metrics:
        first = f"/Users/stenmark/Skola/thesis/repos/image-quality-assessment/test-pyiqa/output/production/json_files/first_execution/{metric}_A100_results.json"
        second = f"/Users/stenmark/Skola/thesis/repos/image-quality-assessment/test-pyiqa/output/production/json_files/second_execution/{metric}_A100_results.json"
        combined = retrieve_data(first, second)
        generate_score_plot(
            metric, combined, args.output_dir, args.one_fogginess, args.every_other
        )
        generate_box_plot(
            metric, combined, args.output_dir, args.one_fogginess, args.every_other
        )
