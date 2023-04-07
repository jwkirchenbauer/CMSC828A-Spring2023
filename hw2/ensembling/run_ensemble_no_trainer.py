# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Adapted by the CMSC828A-Spring2023-Development team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning an ensemble of 🤗 Transformers models for image classification."""
import argparse
import logging
import os
import random

import datasets
from datasets import DatasetDict
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils.versions import require_version

from io_utils import read_json

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Train an ensemble of image classfication models on the Maysee/tiny-imagenet dataset")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to run evaluation only. Assumes previously saved models are in the output directory.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="microsoft/resnet-18",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None, help="Total number of training steps to perform per epoch, per task, rather than covering the entire task dataset.")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")    
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

# Ensemble aggegation method that takes in a test example and either selects
# a single model using a score based scheme such as "confidence" or "entropy"
# or aggregates the predictions of all models using an real aggregation function
# or neighbor scheme. If the neighbor method is used, then a 
# kNN model fit to the featurized training data is also required.
def ensemble_prediction(test_example, 
                        task_models, 
                        aggregation_method="confidence",
                        knn_model=None):
    if aggregation_method == "confidence":
        # for each model, get the confidence score for the top prediction
        # then select the prediction from the model with the highest confidence score
        # as the final prediction

        raise NotImplementedError("confidence based selection not yet implemented")

    elif aggregation_method == "entropy":
        # for each model, get the entropy of the prediction distribution
        # then select the prediction from the model with the lowest output entropy
        # as the final prediction

        raise NotImplementedError("entropy based selection not yet implemented")

    elif aggregation_method == "neighbor_majority":
        raise NotImplementedError("knn based majority voting aggregation not yet implemented")
    
    else:
        raise ValueError("Invalid aggregation method")

    return prediction


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(**accelerator_log_kwargs)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets

    # Task split creation
    task_split_meta = read_json("./task_splits.json")

    task_to_class_list = task_split_meta["task_to_class_list"]
    task_to_class_list = {k: set(v) for k, v in task_to_class_list.items()} # just to make membership check faster

    # make label_to_task dict using the task_to_class_list for oracle method
    label_to_task = {}
    for task in task_to_class_list:
        for label in task_to_class_list[task]:
            label_to_task[label] = task

    # Prepare label mappings.
    # these are originally from the tinyimagenet dataset
    label2id = task_split_meta["label2id"]
    id2label = task_split_meta["id2label"]

    # if args.preprocessed_task_train_datasets_path is None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset("Maysee/tiny-imagenet", task="image-classification")

    task_train_datasets = DatasetDict(
        {k: dataset["train"].filter(lambda x: x["labels"] in task_to_class_list[k]) for k in task_to_class_list.keys()}
    )
    task_eval_datasets = DatasetDict(
        {k: dataset["valid"].filter(lambda x: x["labels"] in task_to_class_list[k]) for k in task_to_class_list.keys()}
    )
    combined_eval_dataset = dataset["valid"]
    
    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label2id),
        i2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
    )
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    task_models = {k: AutoModelForImageClassification.from_config(config=config) for k in task_to_class_list.keys()}

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    # This is the same normalization as standard imagenet data and so we should use it for tinyimagenet as well
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    with accelerator.main_process_first():
        task_train_datasets = DatasetDict(
            {k: v.with_transform(preprocess_train) for k, v in task_train_datasets.items()}
        )
        task_eval_datasets = DatasetDict(
            {k: v.with_transform(preprocess_val) for k, v in task_eval_datasets.items()}
        )
        combined_eval_dataset = combined_eval_dataset.with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    task_train_dataloaders = {
        task: DataLoader(
            task_train_datasets[task], shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
        )
        for task in task_train_datasets
    }
    task_eval_dataloaders = {
        task: DataLoader(
            task_eval_datasets[task], collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
        )
        for task in task_eval_datasets
    }
    combined_eval_dataloader = DataLoader(
        combined_eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
    )

    # make one optimizer per task model
    no_decay = ["bias", "LayerNorm.weight"]
    task_optimizers = {
        task: torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in task_models[task].named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in task_models[task].named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ],
            lr=args.learning_rate,
        )
        for task in task_models
    }

    if args.max_train_steps_per_epoch is None:
        task_steps_per_epoch = {
            task: len(task_train_dataloaders[task])
            for task in task_train_dataloaders
        }
    else:
        task_steps_per_epoch = {
            task: min(args.max_train_steps_per_epoch, len(task_train_dataloaders[task]))
            for task in task_train_dataloaders
        }

    # make one lr_scheduler per task as the num steps will be different
    lr_schedulers = {
        task: get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=task_optimizers[task],
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=task_steps_per_epoch[task] * args.num_train_epochs,
        )
        for task in task_train_datasets
    }

    # prepare the train and eval dataloaders and lr_schedulers, optimizers and models
    task_train_dataloaders = {task: accelerator.prepare(task_train_dataloaders[task]) for task in task_train_dataloaders}
    task_eval_dataloaders = {task: accelerator.prepare(task_eval_dataloaders[task]) for task in task_eval_dataloaders}
    combined_eval_dataloader = accelerator.prepare(combined_eval_dataloader)
    lr_schedulers = {task: accelerator.prepare(lr_schedulers[task]) for task in lr_schedulers}
    task_optimizers = {task: accelerator.prepare(task_optimizers[task]) for task in task_optimizers}
    # task_models = {task: accelerator.prepare(task_models[task]) for task in task_models}

    # send all models to cpu, we will move them to the correct device when we need them
    task_models = {task: task_models[task].cpu() for task in task_models}

    total_train_steps = sum(task_steps_per_epoch.values()) * args.num_train_epochs

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Get the metric function
    metric = evaluate.load("accuracy")

    ##### TRAINING #####

    if not args.eval_only:

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Num Epochs per task = {args.num_train_epochs}")
        logger.info(f"  Num optimization steps per task = {task_steps_per_epoch}")
        logger.info(f"  Total optimization steps across all tasks = {total_train_steps}")

        for task in task_train_dataloaders:
            train_dataloader = task_train_dataloaders[task]
            eval_dataloader = task_eval_dataloaders[task]
            lr_scheduler = lr_schedulers[task]
            optimizer = task_optimizers[task]
            
            # pull out model, move to gpu, then return to cpu after finished with all epochs
            model = task_models[task]
            model.to(accelerator.device)

            completed_steps = 0
            total_train_step_this_task = task_steps_per_epoch[task] * args.num_train_epochs
            task_progress_bar = tqdm(range(total_train_step_this_task), disable=not accelerator.is_local_main_process, desc=f"Task {task} training")
            for epoch in range(args.num_train_epochs):
                model.train()
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):

                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    task_progress_bar.update(1)
                    completed_steps += 1

                    if step >= task_steps_per_epoch[task]-1:
                        break
                
                if completed_steps >= total_train_step_this_task:
                    task_progress_bar.close()

                ####### PER TASK EVALUATION #######
                model.eval()
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_metric = metric.compute()
                logger.info(f"Task {task} epoch {epoch}: {eval_metric}")

                if args.with_tracking:
                    accelerator.wait_for_everyone()
                    metrics_to_log = {
                            f"task_{task}_val_accuracy": eval_metric["accuracy"],
                            f"task_{task}_train_loss": total_loss.item() / task_steps_per_epoch[task],
                            f"task_{task}_lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "step": completed_steps,
                        }
                    accelerator.log(
                        metrics_to_log,
                    )
            model.to("cpu")

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            # unwrap and save each model
            for task in task_models:
                unwrapped_model = accelerator.unwrap_model(task_models[task])
                unwrapped_model.save_pretrained(
                    f"{args.output_dir}/task_{task}", is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )

            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
    
    ###### COMBINED EVALUATION ######

    # load the previously saved models, will fail if not trained and saved as expected above
    for task in task_models:
        task_models[task] = accelerator.unwrap_model(task_models[task])
        task_models[task].load_state_dict(torch.load(f"{args.output_dir}/task_{task}/pytorch_model.bin"))
        task_models[task].to(accelerator.device)
        task_models[task].eval()

    # set the seed so that any randomness in the eval steps is reproducible
    if args.seed is not None:
        set_seed(args.seed)

    # evaluate all individual models on combined eval dataset
    # pull out model, move to gpu, then return to cpu after finished
    for task in tqdm(task_models, desc="Evaluating task models on combined eval dataset"):
        
        model = task_models[task]
        model.to(accelerator.device)

        model.eval()
        for step, batch in enumerate(combined_eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"Task {task} model combined eval: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    f"task_{task}_model_combined_val_accuracy": eval_metric["accuracy"],
                }
            )

        model.to("cpu")


    # next evaluate the ensemble using the random model scheme
    # randomly select a model for each image in the combined eval dataset

    # note, these require enough vram to hold all the models in gpu memory for efficiency
    for task in task_models:
        task_models[task].to(accelerator.device)

    # make a dataloader for the combined eval dataset with normal batchsize, but shuffle it
    # so that effectively, a random model is selected for each image
    combined_eval_dataloader_shuffled = accelerator.prepare(DataLoader(combined_eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size, shuffle=True))

    for step, batch in tqdm(enumerate(combined_eval_dataloader_shuffled), desc="Evaluating ensemble on combined eval dataset (random model scheme)"):
        # randomly select a model
        task = random.choice(list(task_models.keys()))
        model = task_models[task]

        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    logger.info(f"Ensemble model combined eval (random model scheme): {eval_metric}")

    if args.with_tracking:
        accelerator.log(
            {
                f"ensemble_random_model_scheme_combined_val_accuracy": eval_metric["accuracy"],
            }
        )

    # oracle scheme: check which task the val image's label belongs to, then use that task's model to predict

    # For this we need a dataloader with the eval examples and bsz=1, which we'll reuse later too
    # make a dataloader for the combined eval dataset with batchsize 1
    combined_eval_dataloader_bsz_1 = accelerator.prepare(DataLoader(combined_eval_dataset, collate_fn=collate_fn, batch_size=1, shuffle=False))

    for step, batch in tqdm(enumerate(combined_eval_dataloader_bsz_1), desc="Evaluating ensemble on combined eval dataset (oracle scheme)"):

        # get the task of the label
        label = batch["labels"][0].item()
        task = label_to_task[label]

        model = task_models[task]

        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    
    eval_metric = metric.compute()
    logger.info(f"Ensemble model combined eval (oracle scheme): {eval_metric}")
    if args.with_tracking:
        accelerator.log(
            {
                f"ensemble_oracle_scheme_combined_val_accuracy": eval_metric["accuracy"],
            }
        )

    # Selection and aggregation function based evaluation schemes
    # For each eval example, call the ensemble_prediction on the example
    for aggregation_method in ["confidence", "entropy", "neighbor_majority"]:
        for step, batch in tqdm(enumerate(combined_eval_dataloader_bsz_1), desc=f"Evaluating ensemble on combined eval dataset (aggregation='{aggregation_method}')"):
            prediction = ensemble_prediction(batch, task_models, aggregation_method)
            predictions, references = accelerator.gather_for_metrics((prediction, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )   
        eval_metric = metric.compute()
        logger.info(f"Ensemble combined eval (aggregation='{aggregation_method}'): {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    f"{aggregation_method}_ensemble_combined_val_accuracy": eval_metric["accuracy"],
                },
            )
    
    # TODO: Stacked feature base model construction and evaluation code,
    # which likely requires some earlier modifications to the code
    
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()