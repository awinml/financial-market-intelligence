from huggingface_hub import login
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import nltk
import numpy as np
import torch


torch.cuda.empty_cache()

nltk.download("punkt")


# Model Config
TOKEN = ""
MODEL_CHECKPOINT = "t5-small"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
CSV_FILE = "flat.csv"
TRAIN_SIZE = 0.8
MODEL_NAME = (
    MODEL_CHECKPOINT.split("/")[-1] + "-" + CSV_FILE.split("/")[-1].split(".")[-2]
)
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
DECAY = 0.01
EPOCHS = 10
PUSH_TO_HUB = True
FP16 = False


def create_dataset(FILE, TRAIN_SIZE):
    try:
        data = pd.read_csv(FILE, na_values=" ")
    except:
        print(f"Cannot Open {FILE}.")
        return None
    else:
        data.fillna("-", inplace=True)

        raw_datasets = Dataset.from_pandas(data)

        raw_datasets = raw_datasets.train_test_split(train_size=TRAIN_SIZE)

        return raw_datasets


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT, model_max_length=MAX_INPUT_LENGTH
    )

    prefix = "summarize: "

    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def compute_metrics(eval_pred):

    # Loading Rouge Metric
    metric = load_metric("rouge")
    print("Loaded Rouge Metric")

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def huggingface_summarization():
    try:
        if PUSH_TO_HUB:
            login(token=TOKEN)
    except:
        print("Invalid Huggingface Access Token")
    else:
        # Loading csv and Splitting that data
        raw_datasets = create_dataset(CSV_FILE, TRAIN_SIZE)

        if raw_datasets != None:
            print(f"Loaded {CSV_FILE} Successfully(with Split Train Size {TRAIN_SIZE})")

            # Loading AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CHECKPOINT, model_max_length=MAX_INPUT_LENGTH
            )
            print(f"Loaded AutoTokenizer for {MODEL_CHECKPOINT} model")

            # Tokenize Data
            tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
            print("Tokenized Data Successfully")

            # Loading Model
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
            print(f"Loaded {MODEL_CHECKPOINT} successfully")

            # Loading Data Collator
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            args = Seq2SeqTrainingArguments(
                MODEL_NAME,
                evaluation_strategy="epoch",
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                weight_decay=DECAY,
                save_total_limit=3,
                num_train_epochs=EPOCHS,
                predict_with_generate=True,
                fp16=FP16,
                push_to_hub=PUSH_TO_HUB,
            )

            trainer = Seq2SeqTrainer(
                model,
                args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            # Training Model
            trainer.train()
            print("Trained Model Successfully")

            if PUSH_TO_HUB:
                trainer.push_to_hub()


huggingface_summarization()

model_name = "awinml/t5-small-sec-10K"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

summary = pipeline(model=model_name, tokenizer=model_name)
