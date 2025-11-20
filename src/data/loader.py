
from typing import Tuple, Dict
from dataclasses import dataclass

from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset, DatasetDict


@dataclass
class DatasetInfo:
    dataset: DatasetDict
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def load_20ng_dataset(remove=("headers", "footers", "quotes")) -> DatasetInfo:
    """
    Load 20 Newsgroups using sklearn and wrap it into a HuggingFace DatasetDict.

    Returns:
        DatasetInfo with:
          - dataset: DatasetDict with 'train' and 'test'
          - label2id, id2label mappings
    """
    train_bunch = fetch_20newsgroups(subset="train", remove=remove)
    test_bunch  = fetch_20newsgroups(subset="test",  remove=remove)

    label_names = train_bunch.target_names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}

    train_ds = Dataset.from_dict(
        {
            "text": train_bunch.data,
            "label": train_bunch.target,  # ints
        }
    )
    test_ds = Dataset.from_dict(
        {
            "text": test_bunch.data,
            "label": test_bunch.target,
        }
    )

    dataset = DatasetDict(train=train_ds, test=test_ds)

    return DatasetInfo(dataset=dataset, label2id=label2id, id2label=id2label)
