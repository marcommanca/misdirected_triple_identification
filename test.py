import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer

from utils.generic_utils import print_metrics


def ml_model_test(model, x_test, y_test, xgb=False):
    pred = model.predict(x_test)
    print_metrics(y_test, pred, xgb)


def bert_model_test(trainer, test_ds, version_base=True):
    if version_base:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors='pt')

    test_tokenized = test_ds.map(tokenize_function, batched=True)
    pred = trainer.predict(test_tokenized)

    predictions = np.argmax(pred.predictions, axis=1)
    true_labels = pred.label_ids

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)

    if version_base:
        print(f"\n Accuracy: {accuracy} \n "
              f"F1_score: {f1} \n Recall: {recall} \n Precision: {precision}")

    else:
        print(f"\n Accuracy: {accuracy} \n "
              f"F1_score: {f1} \n Recall: {recall} \n Precision: {precision}")


def llama2_model_test(trainer, test_ds, llama_model_path):
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors='pt')

    test_tokenized = test_ds.map(tokenize_function, batched=True)

    pred = trainer.predict(test_tokenized)
    predictions = np.argmax(pred.predictions, axis=1)
    true_labels = pred.label_ids

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)

    print(f"\n Accuracy: {accuracy} "
          f"\n F1_score: {f1} \n Recall: {recall} \n Precision: {precision}")