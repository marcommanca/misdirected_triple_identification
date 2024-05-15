import os
import json
import spacy
import datasets

import pandas as pd


# computes embeddings for training and test data; it returns the embeddings and related labels in two separate lists
# so the output of this function is four lists, two lists of embeddings and two lists of labels
# the "label_map" parameter is used to convert labels to integers (label 'right' converted to 0, 'left' to 1)
def embedder_data(train_data, test_data):
    nlp = spacy.load("en_core_web_sm")

    def sentence_embedding(sentence):
        doc = nlp(sentence)
        return doc.vector

    train_data["emb"] = train_data.e1 + train_data.relation + train_data.e2
    test_data["emb"] = test_data.e1 + test_data.relation + test_data.e2

    x_train = train_data["emb"].apply(lambda word: sentence_embedding(word)).tolist()
    x_test = test_data["emb"].apply(lambda word: sentence_embedding(word)).tolist()
    y_train = train_data[["direction"]]
    y_test = test_data[["direction"]]
    return x_train, x_test, y_train, y_test


# loading of pre-processed data for training and testing of classifiers

# the boolean parameter 'emb' specifies whether the data is to be embeddable or not
# (set to True only if classical Machine Learning models are to be used or tested).

# the "label_map" parameter is used to convert labels to integers (label 'right' converted to 0, 'left' to 1)
# (set to True if you need the data for use with all language models or with XGboost)
def load_datasets(emb=False, label_map=True):
    dataset = os.listdir("./data")
    dataset.remove("conll04_zeroshot")
    dataset.remove("ADE_corpus")
    dataset.remove("semval-RE")
    dataset.remove(".ipynb_checkpoints")

    train_directions = []
    test_directions = []
    for d in dataset:
        d1 = json.load(open(f"./data/{d}/train.json"))
        d2 = json.load(open(f"./data/{d}/test.json"))
        d1.extend(d2)
        data = d1
        for i in range(len(data)):
            if i % 2 == 0:
                t_d = {'e1': f"{data[i]['relations'][0]['head']['name']}",
                       "relation": f"{data[i]['relations'][0]['type']}",
                       "e2": f"{data[i]['relations'][0]['tail']['name']}",
                       "direction": "right",
                       "dataset": d}
            else:
                t_d = {'e1': f"{data[i]['relations'][0]['tail']['name']}",
                       "relation": f"{data[i]['relations'][0]['type']}",
                       "e2": f"{data[i]['relations'][0]['head']['name']}",
                       "direction": "left",
                       "dataset": d}
            if d in ["wiki_0", "wiki_1", "wiki_2", "wiki_3", "wiki_4"]:
                test_directions.append(t_d)
            else:
                train_directions.append(t_d)

    train_data = pd.DataFrame(train_directions).sample(frac=1, random_state=42).reset_index(drop=True).drop_duplicates(
        ignore_index=True)
    test_data = pd.DataFrame(test_directions).sample(frac=1, random_state=42).reset_index(drop=True).drop_duplicates(
        ignore_index=True)

    if label_map:
        m = {"right": 0, "left": 1}
        train_data.direction = train_data.direction.map(m)
        test_data.direction = test_data.direction.map(m)

    if emb:
        x_train, x_test, y_train, y_test = embedder_data(train_data, test_data)
        return x_train, x_test, y_train, y_test
    else:
        train_data = train_data.rename(columns={'direction': 'labels'})
        test_data = test_data.rename(columns={'direction': 'labels'})

        train_data["text"] = train_data.e1 + " " + train_data.relation + " " + train_data.e2
        test_data["text"] = test_data.e1 + " " + test_data.relation + " " + test_data.e2

        train_data = train_data[["text", "labels"]]
        test_data = test_data[["text", "labels"]]

        train_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
        test_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))
        return train_ds, test_ds
