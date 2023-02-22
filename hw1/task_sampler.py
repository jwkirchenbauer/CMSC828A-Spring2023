import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer

from datasets import load_dataset, DatasetDict


class TaskSampler():
    """ 
    Class for sampling batches from a dictionary of dataloaders according to a weighted sampling scheme.

    Dynamic task weights can be externally computed and set using the set_task_weights method,
    or, this class can be extended with methods and state state to implement a more complex sampling scheme.

    You probably/shouldn't need to use this with multiple GPUs, but if you do, you'll may need
    to extend/debug it yourself since the current implementation is not distributed-aware.
    
    Args:
        dataloader_dict (dict[str, DataLoader]): Dictionary of dataloaders to sample from.
        task_weights (list[float], optional): List of weights for each task. If None, uniform weights are used. Defaults to None.
        max_iters (int, optional): Maximum number of iterations. If None, infinite. Defaults to None.
    """
    def __init__(self, 
                *,
                dataloader_dict: dict[str, DataLoader],
                task_weights=None,
                max_iters=None):
        
        assert dataloader_dict is not None, "Dataloader dictionary must be provided."

        self.dataloader_dict = dataloader_dict
        self.task_names = list(dataloader_dict.keys())
        self.dataloader_iterators = self._initialize_iterators()
        self.task_weights = task_weights if task_weights is not None else self._get_uniform_weights()
        self.max_iters = max_iters if max_iters is not None else float("inf")
    
    # Initialization methods
    def _get_uniform_weights(self):
        return [1/len(self.task_names) for _ in self.task_names]
    
    def _initialize_iterators(self):
        return {name:iter(dataloader) for name, dataloader in self.dataloader_dict.items()}
    
    # Weight getter and setter methods (NOTE can use these to dynamically set weights)
    def set_task_weights(self, task_weights):
        assert sum(self.task_weights) == 1, "Task weights must sum to 1."
        self.task_weights = task_weights
    
    def get_task_weights(self):
        return self.task_weights

    # Sampling logic
    def _sample_task(self):
        return np.random.choice(self.task_names, p=self.task_weights)
    
    def _sample_batch(self, task):
        try:
            return self.dataloader_iterators[task].__next__()
        except StopIteration:
            print(f"Restarting iterator for {task}")
            self.dataloader_iterators[task] = iter(self.dataloader_dict[task])
            return self.dataloader_iterators[task].__next__()
        except KeyError as e:
            print(e)
            raise KeyError("Task not in dataset dictionary.")
    
    # Iterable interface
    def __iter__(self):
        self.current_iter = 0
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iters:
            raise StopIteration
        else:
            self.current_iter += 1
        task = self._sample_task()
        batch = self._sample_batch(task)
        return task, batch



if __name__ == "__main__":

    # Demonstrate usage of TaskSampler with MultiNLI dataset genres as dummy "tasks"

    # NOTE : we're not addressing the fact that each task has a different head/model
    # associated with it, but you could use the task name to select the appropriate head
    # in your training loop. You could even extend this class to keep track of the
    # current heads/models and return them with the batch.

    import os
    os.environ['HF_HOME'] = "/cmlscratch/jkirchen/.cache/huggingface"

    mnli_all_ds = load_dataset("multi_nli", split="train")

    # use genere splits as makeshift tasks
    dataset_dict = DatasetDict({"government": mnli_all_ds.filter(lambda x: x["genre"] == "government"), 
                                "fiction": mnli_all_ds.filter(lambda x: x["genre"] == "fiction"), 
                                "telephone": mnli_all_ds.filter(lambda x: x["genre"] == "telephone")})
    
    genres = list(dataset_dict.keys())


    # identifier on huggingface.co/models
    hf_model_name_or_path = "bert-base-cased"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)

    # Preprocessing the raw_datasets
    # NOTE : this is an uber simplified version of the preprocessing functions
    # you'll find in most of the example scripts.
    # SQuAD and NER are more complicated so I'd use whats in the examples, 
    # but you can use the same general approach.

    sentence1_key, sentence2_key = ("premise", "hypothesis")

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        # the tokenization args here are flexible, you can change them if you want (refer to the examples and documentation)
        result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
        return result


    processed_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[genres[0]].column_names,
        desc="Running tokenizer on dataset",
    )


    # Task dataLoader(s) creation:
    # look up Dataloader "collator"s as well as the source of this actual
    # huggingface collator to understand what this is doing
    mnli_data_collator = DataCollatorWithPadding(tokenizer)
    
    # here we create a dictionary of dataloaders, one for each task
    # and this is the dictionary we'll pass to the TaskSampler)
    dataloader_dict = {key: DataLoader(dataset, shuffle=True, collate_fn=mnli_data_collator, batch_size=2) for key, dataset in processed_datasets.items()}

    # Examples of TaskSampler usage with different configurations
    # NOTE: these configuration might inspire a way to achieve 
    # the different task orderings/sequences you need to implement for experiments in Part 2 of the homework

    # task_sampler = TaskSampler(dataloader_dict=dataloader_dict)
    # task_sampler = TaskSampler(dataloader_dict=dataloader_dict, max_iters=10, task_weights=[1.0, 0.0, 0.0])

    print("Version 1, fixed weights")
    task_sampler = TaskSampler(dataloader_dict=dataloader_dict, max_iters=20, task_weights=[0.0, 0.5, 0.5])
    
    for task, batch in task_sampler:
        print(task_sampler.get_task_weights())
        print(task)
        # print(batch["input_ids"].shape)
        # print(tokenizer.batch_decode(batch["input_ids"])) # verbose


    # is equivalent to
    print("Version 2, 'dynamic' weights")
    task_sampler = TaskSampler(dataloader_dict=dataloader_dict)
    max_iters = 20
    task_iterator = iter(task_sampler)
    
    for i in range(max_iters):
        # NOTE : however, now we have a chance to set the weights 'dynamically'
        if i < 10:
            task_sampler.set_task_weights([0.0, 0.5, 0.5])
        else:
            task_sampler.set_task_weights([0.9, 0.1, 0.0])

        print(task_sampler.get_task_weights())
        task, batch = next(task_iterator)
        print(task)
        # print(batch["input_ids"].shape)
        # print(tokenizer.batch_decode(batch["input_ids"])) # verbose

    # So ... you could implement the dynamic weight computation either
    # externally or in the TaskSampler class 

