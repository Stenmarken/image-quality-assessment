import pandas as pd
import plotly.express as px
import argparse


def extract_base_filename(name):
    # Extracts the part before the last underscore (e.g., 000_162 from 000_162_0.01)
    return "_".join(name.split("_")[:-1])


def generate_score_plot(metric, csv_path, output_path):
    # Load your CSV file
    df = pd.read_csv(csv_path)

    # Combine Time and Weather for grouping
    df["Time_Weather"] = df["Weather"] + ", " + df["Time"]

    # Extract base filename group (used to connect lines)
    df["Base_File"] = df["File_name"].apply(extract_base_filename)

    # Optional: convert distortion to numeric if it's string-like
    df["Distortion"] = df["Distortion"].astype(float)

    # Sort so lines are drawn in order of distortion
    df = df.sort_values(by=["Base_File", "Distortion"])

    # Create line plot with markers; group lines by Base_File
    fig = px.line(
        df,
        x="Distortion",
        y="Score",
        color="Time_Weather",
        line_group="Base_File",  # This connects same file across distortions
        markers=True,
        hover_data=df.columns,
        title=f"{metric} Score vs Distortion Level",
    )

    # Layout customization
    fig.update_layout(
        xaxis_title="Distortion Severity",
        yaxis_title="Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.5,
        ),
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        legend_title_text="",
        font=dict(size=20),
        width=800,
        height=600,
        margin=dict(l=5, r=5, t=45, b=60),
    )

    fig.update_traces(marker=dict(size=10))

    fig.show()
    if output_path:
        fig.write_image(output_path, scale=3)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots script")

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to be plotted",
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to csv file containing scores",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_score_plot(args.metric, args.csv_path, args.output_path)
