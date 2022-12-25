import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score

from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
)


raw_dataset = load_dataset("csv", data_files="data.csv")

split = raw_dataset["train"].train_test_split(test_size=0.3, seed=42)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenizer_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  prediction = np.argmax(logits, axis=-1)
  acc = np.mean(prediction == labels)
  f1 = f1_score(labels, prediction, average='macro')

  return {'accuracy': acc, 'f1': f1}

tokenized_datasets = split.map(tokenizer_fn, batched=True)
config = AutoConfig.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)

training_args = TrainingArguments(
    checkpoint,
    evaluation_strategy='epoch',
    num_train_epochs=3,
    learning_rate=1e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    save_total_limit = 3,
    fp16 = False,
    push_to_hub = True,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()