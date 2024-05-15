from train import train_svm, train_randomforest, train_bert, train_xgboost, llama2_train
from test import ml_model_test, bert_model_test, llama2_model_test
from utils.data_utils import load_datasets


#x_train, x_test, y_train, y_test = load_datasets(emb=True, label_map=False)
#x_xg_train, x_xg_test, y_xg_train, y_xg_test = load_datasets(emb=True, label_map=True)
train_ds, test_ds = load_datasets(emb=False, label_map=True)

if __name__ == '__main__':
    # svm = train_svm(x_train, y_train)
    # print("SVM Performance")
    # ml_model_test(svm, x_test, y_test)
    #
    # del svm
    #
    # rf = train_randomforest(x_train, y_train)
    # print("RandomForest Performance")
    # ml_model_test(rf, x_test, y_test)
    #
    # del rf
    #
    # xgb_class = train_xgboost(x_xg_train, y_xg_train)
    # print("XGBoost Performance")
    # ml_model_test(xgb_class, x_xg_test, y_xg_test, xgb=True)
    #
    # del xgb_class

    bert_base = train_bert(train_ds,test_ds, version_base=True, output_dir="./tuned_models/bert-base-uncased-classifier")
    print("BERT base Performance")
    bert_model_test(bert_base, test_ds, version_base=True)

    del bert_base

    bert_large = train_bert(train_ds,test_ds, version_base=False, output_dir="./tuned_models/bert-large-uncased-classifier", epochs=10)
    print("BERT large Performance")
    bert_model_test(bert_large, test_ds, version_base=False)

    del bert_large

    llama_7b = llama2_train(path_model="../../llama-2-models/Llama-2-7b-hf",
                            output_dir="./tuned_models/llama-7b-classifier",
                            train_ds=train_ds,
                            test_ds=test_ds,
                            epochs=1,
                            batch_size=256)
    print("Llama2-7b Performance")
    llama2_model_test(trainer=llama_7b,
                      test_ds=test_ds,
                      llama_model_path="../../llama-2-models/Llama-2-7b-hf")

    del llama_7b

    llama_13b = llama2_train(path_model="../../llama-2-models/Llama-2-13b-hf",
                            output_dir="./tuned_models/llama-13b-classifier",
                            train_ds=train_ds,
                            test_ds=test_ds,
                            epochs=1,
                            batch_size=256)
    print("Llama2-13b Performance")
    llama2_model_test(trainer=llama_13b,
                      test_ds=test_ds,
                      llama_model_path="../../llama-2-models/Llama-2-13b-hf")

    del llama_13b

    llama_70b = llama2_train(path_model="../../llama-2-models/Llama-2-70b-hf",
                            output_dir="./tuned_models/llama-70b-classifier",
                            train_ds=train_ds,
                            test_ds=test_ds,
                            epochs=1,
                            batch_size=64)
    print("Llama2-70b Performance")
    llama2_model_test(trainer=llama_70b,
                      test_ds=test_ds,
                      llama_model_path="../../llama-2-models/Llama-2-70b-hf")

    del llama_70b



