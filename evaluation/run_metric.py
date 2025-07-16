import torch
import time
from pathlib import Path
import json
import pyiqa
import pyiqa.utils
import torch
import time
import json
import wandb
from utils.utils import bcolors, log, load_yaml
import argparse


class MetricRunner:
    def __init__(
        self,
        metric,
        input_dir,
        output_dir,
        device,
        use_prev_results=False,
        prev_results_path="",
    ):
        self.metric = metric
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_prev_results = use_prev_results
        self.prev_results_path = prev_results_path
        self.prepare_inputs()
        self.prepare_logging()
        self.device = device

    def prepare_inputs(self):
        self.input_sub_dirs = [p for p in self.input_dir.iterdir() if p.is_dir()]
        self.output_file_path = self.output_dir / f"{self.metric}_results.json"
        self.output_timing_path = self.output_dir / f"{self.metric}_timing_results.json"
        self.results_dict = {}
        if self.use_prev_results and self.prev_results_path:
            with open(self.output_file_path, "r") as f:
                self.results_dict = json.load(f)

    def prepare_logging(self):
        self.timing_data = {
            "total_wall_time_seconds": 0.0,
            "total_model_inference_time_seconds": 0.0,
        }

    def get_images(self, dir):
        # Assumes that dir is of type Path
        return [
            f for f in dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

    def append_results(self, dir, distorted_images_scores):
        # Convert the key to a string as JSON doesn't support posix paths.
        self.results_dict[str(dir)] = distorted_images_scores
        with open(self.output_file_path, "w") as f:
            json.dump(self.results_dict, f, indent=4)

    def log_time_data(self, total_wall_time, num_imgs):
        log(f"Total wall time: {total_wall_time}", bcolor_type=bcolors.HEADER)
        log(
            f"Total inference time: {self.timing_data["total_model_inference_time_seconds"]}",
            bcolor_type=bcolors.HEADER,
        )
        self.timing_data["total_wall_time_seconds"] = total_wall_time
        with open(self.output_timing_path, "w") as f:
            json.dump(self.timing_data, f, indent=4)
        wandb.log(
            {
                "total_inference_time": self.timing_data[
                    "total_model_inference_time_seconds"
                ],
                "total_wall_time": total_wall_time,
                "f_processed": num_imgs,
            }
        )

    def run_metric(self):
        num_imgs = 0
        with torch.no_grad():
            model = pyiqa.create_metric(self.metric, device=self.device)
            log(
                f"{self.metric}: lower better: {model.lower_better}",
                bcolor_type=bcolors.HEADER,
            )
            log(
                f"{self.metric}: score range: {model.score_range}",
                bcolor_type=bcolors.HEADER,
            )
            start_wall_time = time.time()
            for dir in self.input_sub_dirs:
                if dir in self.results_dict:
                    log(f"Already ran {dir}. Skipping", bcolor_type=bcolors.WARNING)
                    continue
                log(f"Running {self.metric} on {dir}", bcolor_type=bcolors.OKBLUE)

                distorted_images_scores = {}
                imgs = self.get_images(dir)
                num_imgs += len(imgs)
                results_table = wandb.Table(columns=["sample_id", "score"])
                for img in imgs:
                    img_str = str(img)
                    img_tensor = pyiqa.utils.img_util.imread2tensor(img_str).unsqueeze(
                        0
                    )
                    img_tensor = img_tensor.to(self.device)
                    start_inference = time.time()
                    score = model(img_tensor).item()
                    end_inference = time.time()
                    distorted_images_scores[img.name] = score
                    log(f"{img.name}: {score}", bcolor_type=bcolors.OKBLUE)
                    del img_tensor
                    torch.cuda.empty_cache()

                    self.timing_data["total_model_inference_time_seconds"] += (
                        end_inference - start_inference
                    )
                    results_table.add_data(img_str, score)

                wandb.log({dir.name: results_table})
                self.append_results(dir, distorted_images_scores)
            total_wall_time = time.time() - start_wall_time
            self.log_time_data(total_wall_time, num_imgs)


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
    args = parse_args()
    if not args.config:
        raise argparse.ArgumentTypeError("Directory path must be specified")
    else:
        config = load_yaml(args.config)

    wandb.login()
    wandb.init(project="test", name="test-run", resume=False)
    metric = config["metric"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    use_prev_results = config["use_prev_results"]
    prev_results_path = config["prev_results_path"]

    log(f"Running metric: {metric}", bcolor_type=bcolors.HEADER)
    log(f"Using device: {device}", bcolor_type=bcolors.HEADER)
    log(f"Input directory: {input_dir}", bcolor_type=bcolors.HEADER)
    log(f"Output directory: {output_dir}", bcolor_type=bcolors.HEADER)
    log(f"Using previous results: {use_prev_results}", bcolor_type=bcolors.HEADER)
    if use_prev_results:
        log(f"Previous results path: {prev_results_path}", bcolor_type=bcolors.HEADER)
    runner = MetricRunner(
        metric=config["metric"],
        input_dir=config["input_dir"],
        output_dir=config["output_dir"],
        device=device,
        use_prev_results=config["use_prev_results"],
        prev_results_path=config["prev_results_path"],
    )
    runner.run_metric()
    wandb.finish()


if __name__ == "__main__":
    main()
