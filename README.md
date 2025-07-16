# Image quality assessment

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Distorting images
Distorting images is done using the `corrupt_images.py` script. It requires a configuration file that specifies the distortion parameters. It can be run using the following command:

```python
python3 -m distortion.corrupt_images -c configs/example_config.yaml
```


