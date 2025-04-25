import random
import shutil
from pathlib import Path
import argparse


def select_and_copy_files(
    source_directory, target_directory, n_files=50, seed=42, filetype=".png"
):
    """
    Randomly select n PNG files from source directory and copy them to target directory.

    Args:
        source_directory (str): Path to the source directory containing PNG files
        target_directory (str): Path to the target directory where files will be copied
        n_files (int): Number of files to select
        seed (int): Random seed for reproducibility

    Returns:
        list: List of selected file paths in the target directory
    """
    # Set random seed
    random.seed(seed)

    # Convert paths to Path objects
    source_path = Path(source_directory)
    target_path = Path(target_directory)

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Get all PNG files from directory
    files = list(source_path.glob(f"*{filetype}"))

    # Check if we have enough files
    if len(files) < n_files:
        raise ValueError(
            f"Directory contains only {len(files)} files, cannot select {n_files}"
        )

    # Randomly select files
    selected_files = random.sample(files, n_files)

    # Copy files to target directory
    copied_files = []
    for file in selected_files:
        destination = target_path / file.name
        shutil.copy2(file, destination)
        copied_files.append(str(destination))

    return copied_files


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selection of images")
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        help="Path to directory where the source images are.",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        help="Path to directory where the images will be copied to.",
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        help="Number of images that are randomly selected.",
        default=10,
    )
    parser.add_argument("-f",
                        "--filetype",
                        type=str,
                        help="File type to be moved",
                        default=".png")
    parser.add_argument("--seed", type=int, help="Randomness seed", default=42)
    args = parser.parse_args()

    copied_files = select_and_copy_files(
        args["source_dir"],
        args["target_dir"],
        n_files=args["num_images"],
        seed=args["seed"],
        filetype=args["filetype"]
    )

    print(f"Successfully copied {len(copied_files)} files to {args["target_dir"]}")
    for i, file in enumerate(copied_files, 1):
        print(f"{i}. {file}")
