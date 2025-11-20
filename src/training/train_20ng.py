import os
from pathlib import Path

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.loader import load_20ng_dataset
from src.utils.config_loader import load_config


def build_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(batch, tokenizer, max_length: int = 512):
    """
    Tokenize a batch. No padding here — it will be handled dynamically by the data collator.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    config = load_config()

    # ---- Paths ----
    model_dir = Path(config["paths"].get("distilbert_20ng_dir", "models/bert_20ng"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Hyperparameters ----
    hparams = config["training"]["twenty_ng"]
    max_length = hparams.get("max_length", 512)
    batch_size = hparams.get("batch_size", 16)
    learning_rate = hparams.get("learning_rate", 5e-5)
    num_epochs = hparams.get("num_epochs", 3)

    # 1) Load dataset
    ds_info = load_20ng_dataset()
    dataset: DatasetDict = ds_info.dataset
    label2id = ds_info.label2id
    id2label = ds_info.id2label
    num_labels = len(label2id)

    # 2) Tokenizer
    tokenizer = build_tokenizer()

    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer, max_length=max_length)

    tokenized_datasets = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    #Dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_dir / "hf_checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        weight_decay=0.01,

        logging_steps=50,
        report_to="none",
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,  
        compute_metrics=compute_metrics,
    )

    # 6) Train
    trainer.train()

    # 7) Evaluate
    eval_results = trainer.evaluate()
    print("\nFinal evaluation:", eval_results)

    # 8) Save model + tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Optional detailed classification report
    preds_output = trainer.predict(tokenized_datasets["test"])
    preds = preds_output.predictions.argmax(-1)
    labels = tokenized_datasets["test"]["label"]
    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in range(num_labels)]
    ))


if __name__ == "__main__":
    main()
