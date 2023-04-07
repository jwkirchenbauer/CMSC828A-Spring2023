## Starter Code for Homework Assignment 2

"No Warranty" i.e. this script should work as provided, but there could be a minor bug here or there.
It assumes that the `io_utils.py` and `task_splits.json` files are available in the working directory
and that your python environment has up to date versions of the packages in the `requirements.txt` file.

Take a look at the arguments in the argparse block to see what controls/features you have already implemented.
Notably, the `--with_tracking` flag lets you log to a gui experiment tracker like (`wandb` by default)

Basic Usage
```
python run_ensemble_no_trainer.py \
    --output_dir ./output \
    --model_name_or_path microsoft/resnet-18 \
    --with_tracking
```