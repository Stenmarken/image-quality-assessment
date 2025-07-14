### evaluate_results.py

This script calculates SRCC and KRCC values of the raw scores calculated earlier. It reqiures a YAML configuration file that describes where the JSON files containing the score data is to be found. Look in `configs/production_wandb_configs/production.yaml` for more information.

The script can be run with the following command:

```
python3 evaluate_results.py --c configs/production/wandb_configs/production.yaml --one_fogginess
```

### run_permutation_tests.py

This script runs permutation tests for the various NR-IQA metrics. It doesn't require any input but there are hardcoded values in the file that potentially has to be changed.

The script can be run with the following command:

```
python3 run_permutation_tests.py
```

### statistical_tests.py

This script runs the Friedman, Nemenyi and Wilcoxon signed rank tests. It requires an env file that contains the IDs of the runs of the various metrics in Wandb. Look in `configs/production_wandb_configs/production_wandb_configs/coefficient_nr_iqa.env` for more information.

The script can be run with the following command:

```
python3 statistical_tests.py --env configs/production_wandb_configs/coefficient_nr_iqa.env
```

### nr_iqa_coeff_plot.py

The script generates a plot of the coefficients of the various NR-IQA metrics. It requires an env configuration file that describes the IDs of the runs in Wandb. Look in `configs/production_wandb_configs/production_wandb_configs/coefficient_nr_iqa.env` for more information. You can also specify the output path for the plot.
The script can be run with the following command:

```
python3 nr_iqa_coeff_plot.py --env configs/production_wandb_configs/coefficient_nr_iqa.env --output_path output/nr_iqa_coefficients_plot.png
```

### nr_pcqa_coeff_plot.py

This script generates a plot of the coefficients of the various NR-PCQA metrics. It requires an env configuration file that describes the IDs of the runs in Wandb. Look in `configs/production_wandb_configs/production_wandb_configs/coefficient_nr_pcqa.env` for more information. You can also specify the output path for the plot.

```
python3 nr_pcqa_coeff_plot.py --env configs/production_wandb_configs/production_wandb_configs/coefficient_nr_pcqa.env --output_path output/nr_pcqa_coefficients_plot.png
```

### nr_iqa_score_plot.py

This script generates a plot of the scores for a metric. It requires a metric name and potentially some extra configuration as the paths are hardcoded in the script. You can also specify an output directory. If you don't, it won't save the plot The script can be run with the following command:

```
python3 nr_iqa_score_plot.py --metric topiq_nr --output_dir output/
```

### run_permutation_tests.py

This script runs permutation tests for the various NR-IQA/NR-PCQA metrics. It takes as input a YAML configuration file that describes the paths to the JSON files containing the score data. Look in `configs/production_wandb_configs/production.yaml` for more information.
The script can be run with the following commands:

```
python3 run_permutation_tests.py --c configs/production_wandb_configs/permutation_nr_iqa.yaml
python3 run_permutation_tests.py --c configs/production_wandb_configs/permutation_nr_pcqa.yaml

```
