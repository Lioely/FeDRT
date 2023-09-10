from model.utils import create_model
import pytorch_lightning as pl
from data.data_tools import make_data1, make_data2, DDIDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from metrics import evaluate
import torch.nn.functional as F
import os
import yaml
import time
import torch


def main():
    cross_ver_tim = 5
    batch_size = 256
    # data_info = "../data/dataset2/drug_information_del_noDDIxiaoyu50.csv"
    # data_extraction = "../data/dataset2/df_extraction_cleanxiaoyu50.csv"
    #make_data1("data/dataset1/event.db")
    feature, label, event_num, feature_len = make_data1("data/dataset1/event.db")

    # write yaml_file
    yaml_file = None

    with open('model/task1.yaml', 'r', encoding='utf-8') as f:
        yaml_file = yaml.load(f.read(), Loader=yaml.Loader)
    yaml_file['model']['params']['output_dim'] = event_num
    yaml_file['model']['params']['struct_len'] = feature_len[-1]
    yaml_file['model']['params']['other_len'] = feature_len[0] + feature_len[1] + feature_len[2]
    task = yaml_file['model']['task']
    with open('model/task1.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data=yaml_file, stream=f, allow_unicode=True)

    if os.path.exists("data/dataset1/k_fold.npy"):
        k_fold = np.load("data/dataset1/k_fold.npy",allow_pickle=True)
    else:
        skf = StratifiedKFold(n_splits=cross_ver_tim)
        k_fold = []
        for train_index, test_index in skf.split(feature, label):
            k_fold.append([test_index, test_index])
        np.save("data/dataset1/k_fold.npy", k_fold)
        
    #save result
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])

    print("K-Flod validation:")
    cnt = 1
    start = time.time()
    for train_index, test_index in k_fold:
        print(f"Flod: {cnt}")
        cnt += 1

        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]

        print(f"Making Dataset")
        train_dataset = DDIDataset(x_train, np.array(y_train))
        test_dataset = DDIDataset(x_test, np.array(y_test))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        print(f"Creating Model")
        model = create_model("model/task1.yaml")
        trainer = pl.Trainer(accelerator="gpu", devices="auto", max_epochs=60,
                             default_root_dir=f'lightning_logs/fold{cnt - 1}')
        print(f"Train model, fold:{cnt - 1}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        # test
        model.freeze()

        predictions = trainer.predict(model, test_loader)

        predictions = torch.cat(predictions, 0)

        pre_score = np.zeros((0, event_num), dtype=float)
        pre_score = np.vstack((pre_score, F.softmax(predictions).cpu().numpy()))

        pred_type = np.argmax(pre_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pre_score))

        y_true = np.hstack((y_true, y_test))
        np.save(f"task1_pred.npy", y_pred)
        np.save(f"task1_score.npy", y_score)
        np.save(f"task1_true.npy", y_true)

    end = time.time()
    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, task_type=task)
    np.save(f'task1_result.npy', np.array(result_all))
    np.save(f'task1_result_eve.npy', np.array(result_eve))
    print(f"train_time:{end - start}")


main()
