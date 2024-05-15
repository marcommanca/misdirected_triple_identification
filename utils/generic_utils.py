import evaluate
import torch

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def print_metrics(gt, pred, xgb=True):
    if xgb:
        pos_label=1
    else:
        pos_label="left"
    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, pos_label=pos_label)
    recall = recall_score(gt, pred, pos_label=pos_label)
    precision = precision_score(gt, pred, pos_label=pos_label)

    print(
        f"Classifier Metrics: \n Accuracy: {accuracy} \n F1_score: {f1} \n Recall: {recall} \n Precision: {precision}")


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    recall = evaluate.load("recall")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    rec = recall.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    f1 = f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        'accuracy': acc["accuracy"],
        'precision': prec["precision"],
        'recall': rec["recall"],
        'f1': f1["f1"]
    }


class SaveScoreCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        f_name = f"{args.output_dir}/checkpoint-{state.global_step}/score.original_module.pt"
        torch.save(self.model.model.score.original_module.state_dict(), f_name)
