import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import torch
import sqlite3

feature_list = ["target", "enzyme", "pathway"]


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    feature_len = []
    vector = np.zeros((len(df_drug['name']), 0), dtype=float)  # vector=[]
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        print(f"making data feature {i}")
        tempvec = one_hot_1(i, df_drug)
        vector = np.hstack((vector, tempvec))
        feature_len.append(tempvec.shape[1])
        
    print(f"making data feature smile")
    tempvec = one_hot_2("smile", df_drug)
    vector = np.hstack((vector, tempvec))
    feature_len.append(tempvec.shape[1])

    # Transfrom the drug ID to feature vector
    for i in range(len(df_drug['name'])):
        d_feature[df_drug['name'][i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539

    return new_feature, new_label, event_num, feature_len


def one_hot_1(feature_name, df):
    """
    构建特征矩阵，每一个pathway，enzyme，target是否在drug中出现。出现为0
    """
    all_feature = []
    drug_list = df[feature_name]
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.ones((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 0

    df_feature = np.array(df_feature)
    return df_feature


def one_hot_2(feature_name, df):
    """
    构建特征矩阵，每一个pathway，enzyme，target是否在drug中出现。出现为0
    """
    all_feature = []
    drug_list = df[feature_name]
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    return df_feature


def make_data1(data_path):
    conn = sqlite3.connect(data_path)
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_feature, new_label, event_num, feature_len = prepare(df_drug, feature_list, mechanism, action, drugA,
                                                             drugB)
    return new_feature, new_label, event_num, feature_len


def make_data2(data_info, data_extraction):
    df_drug = pd.read_csv(data_info)
    extraction = pd.read_csv(data_extraction)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_feature, new_label, event_num, feature_len = prepare(df_drug, feature_list, mechanism, action, drugA,
                                                             drugB)
    return new_feature, new_label, event_num, feature_len


def make_data1_task2(data_path):
    conn = sqlite3.connect(data_path)
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_feature, new_label, event_num, feature_len = prepare(df_drug, feature_list, mechanism, action, drugA,
                                                             drugB)
    train_drug,test_drug = data_process(new_feature, new_label,drugA,drugB,event_num,5)
    
    return new_feature, new_label, event_num, feature_len,train_drug,test_drug,drugA ,drugB


def data_process(feature, label,drugA,drugB, event_num,cross_ver_tim):
    # cro val
    temp_drug1 = [[] for i in range(event_num)]
    temp_drug2 = [[] for i in range(event_num)]
    # 每一个event中参与的药
    for i in range(len(label)):
        temp_drug1[label[i]].append(drugA[i])
        temp_drug2[label[i]].append(drugB[i])
    drug_cro_dict = {}
    # 将数据分为cross_ver_tim折
    for i in range(event_num):
        for j in range(len(temp_drug1[i])):
            # 遍历所有的event中参与的所有药物，将其打上K折的记号
            drug_cro_dict[temp_drug1[i][j]] = j % cross_ver_tim
            drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
    # 用字典的意义就是为了让该药物仅在训练集/测试集出现
    train_drug = [[] for i in range(cross_ver_tim)]
    test_drug = [[] for i in range(cross_ver_tim)]
    # 5折均分数据
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            # 根据药物打上的记号来切分数据
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)
    return train_drug,test_drug


class DDIDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)

        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

