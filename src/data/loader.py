# Loads datasets and converts it into a HuggingFace DatasetDict format that our training loop expects.

from typing import Tuple, Dict
from dataclasses import dataclass

"""
Loads 20 Newsgroups dataset returns a "Bunch" object:
    .data   → list of raw text documents
    .target → list of integer labels
    .target_names → list of category names (strings)
"""
from sklearn.datasets import fetch_20newsgroups

"""
HuggingFace Dataset
DatasetDict → container holding multiple datasets { "train": Dataset, "test": Dataset }
"""
from datasets import Dataset, DatasetDict


@dataclass
class DatasetInfo:
    """
    Container for the dataset and its label mappings
    @dataclass automatically generates an initializer, so it can be constructed simply with: 
        DatasetInfo(dataset=..., label2id=..., id2label=...)
    """

    dataset: DatasetDict        # Holds 'train' and 'test' HF datasets
    label2id: Dict[str, int]    # Map from label string → integer ID
    id2label: Dict[int, str]    # Reverse map integer ID → label string


def load_20ng_dataset(remove=("headers", "footers", "quotes")) -> DatasetInfo:
    """
    Loads 20 Newsgroups using sklearn and wraps  into a HuggingFace DatasetDict.

    Returns:
        DatasetInfo with:
          - dataset: DatasetDict with 'train' and 'test'
          - label2id, id2label mappings
    """

    #Load raw 20NG using sklearn
    train_bunch = fetch_20newsgroups(subset="train", remove=remove)
    test_bunch  = fetch_20newsgroups(subset="test",  remove=remove)

    #Create label mappings
    label_names = train_bunch.target_names                      #List of 20 category strings
    id2label = {i: name for i, name in enumerate(label_names)}  #Map from integer ID → label string
    label2id = {name: i for i, name in id2label.items()}        #Reverse map label string → integer ID

    #Convert to HuggingFace DatasetDict
    train_ds = Dataset.from_dict(
        {
            "text": train_bunch.data,       # raw text documents
            "label": train_bunch.target,    # integer labels
        }
    )
    test_ds = Dataset.from_dict(
        {
            "text": test_bunch.data,    # raw text documents
            "label": test_bunch.target, # integer labels
        }
    )

    # Combine into DatasetDict
    dataset = DatasetDict(train=train_ds, test=test_ds) # HF Trainer expects DatasetDict with 'train' and 'test'

    # Return dataset and label mappings
    return DatasetInfo(dataset=dataset, label2id=label2id, id2label=id2label)
