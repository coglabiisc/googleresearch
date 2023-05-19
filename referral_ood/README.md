
## Avoiding Catastrophic Referral Failures in Medical Images Under Domain Shift

This repository contains the code for the paper:

**Avoiding Catastrophic Referral Failures in Medical Images Under Domain Shift** <br>
Anuj Srivastava, Pradeep Shenoy, Devarajan Sridharan <br>
*ICLR 2023 Workshop on Domain Generalization* <br>

### Usage

The main script for all methods is run using the following command:

```
python main.py \
	--seed $seed \
	--validation_fraction $validation_fraction \
	--results_path $results_path \
	--plots_path $plots_path
```

`$results_path` is the path to the directory containing the classification results in the format specified by the RETINA Benchmark, with the following structure:
```
$results_path
└── <distribution-shift>
    └── <model-name>
        └── <single-or-ensemble>
            └── <test-domain>
                └── eval_results_<seed>
                    ├── y_logit.npy
                    └── y_true.npy
```

Performance vs coverage curves will be plotted for each setting in `$plots_path`



### Citation

If you find our methods useful, please cite:

```
@misc{Srivastava_2023_ICLR_DG,
    author    = {Anuj Srivastava and Pradeep Shenoy and Devarajan Sridharan},
    title     = {Avoiding Catastrophic Referral Failures in Medical Images Under Domain Shift},
    booktitle = {ICLR Workshop on Domain Generalization},
    year      = {2023}
}
```
