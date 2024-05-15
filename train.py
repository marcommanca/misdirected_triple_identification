import torch

import xgboost as xgb

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer,
                          BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from utils.generic_utils import SaveScoreCallback, compute_metrics


peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1,
                         bias="none")


# svm classifier training
def train_svm(x_train, y_train, kernel="sigmoid", random_state=42):
    svm_classifier = SVC(kernel=kernel, random_state=random_state)
    svm_classifier.fit(x_train, y_train)
    return svm_classifier


# random forest classifier training
def train_randomforest(x_train, y_train, n_estimators=300, random_state=42):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier


# Extreme Gradient Boosting classifier training
def train_xgboost(x_train, y_train, n_estimators=300, random_state=42):
    xgboost = xgb.XGBClassifier(n_estimators=n_estimators, random_state=random_state)
    xgboost.fit(x_train, y_train)
    return xgboost


# BERT-base classifier training
def train_bert(train_ds,
               test_ds,
               version_base=True,
               learning_rate=1e-4,
               batch_size=256,
               epochs=30,
               steps=100,
               output_dir="./tuned_models/bert-base-uncased-classifier",
               eval_dim=100,
               save=True):
    if version_base:
        model_name = 'bert-base-uncased'
    else:
        model_name = 'bert-large-uncased'

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors='pt')

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = test_ds.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=-1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=steps,
        logging_steps=steps,
        save_strategy="no",
        push_to_hub=False,
        bf16=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized.select([i for i in range(eval_dim)]),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics

    )
    trainer.add_callback(SaveScoreCallback(model))
    trainer.train()
    if save:
        trainer.save_model(output_dir)
    return trainer


def llama2_train(path_model,
                 train_ds,
                 test_ds,
                 learning_rate=2e-5,
                 batch_size=256,
                 epochs=1,
                 steps=100,
                 output_dir="./tuned_models/llama2-7b-classifier",
                 eval_dim=100,
                 save=True):
    llama2_model_folder = path_model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForSequenceClassification.from_pretrained(llama2_model_folder,
                                                               device_map='auto',
                                                               quantization_config=bnb_config,
                                                               num_labels=2)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(llama2_model_folder)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors='pt')

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = test_ds.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=-1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=steps,
        logging_steps=steps,
        save_strategy="no",
        push_to_hub=False,
        bf16=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized.select([i for i in range(eval_dim)]),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics

    )
    trainer.add_callback(SaveScoreCallback(model))
    trainer.train()
    if save:
        trainer.save_model(output_dir)
    return trainer
