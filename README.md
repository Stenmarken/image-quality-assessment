# Image quality assessment

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Distorting images

Distorting images is done using the `corrupt_images.py` script. It requires a configuration file that specifies the distortion parameters. It can be run using the following command:

```python
python3 -m distortion.corrupt_images -c configs/disstortion_configs/distortion_config.yaml
```

## Running the image quality assessment

To run the image quality assessment, use the `run_metric.py` script. It requires a configuration file that specifies the metric, input, output directories, and other parameters. It can be run using the following command:

```python
python3 -m evaluation.run_metric -c configs/metric_configs/metric_config.yaml
```
