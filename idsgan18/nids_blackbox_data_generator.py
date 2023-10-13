import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

debug = False
random.seed(12345)


def preprocess_dataframe(original_dataframe, cat_dict=None):
    original_dataframe["class"] = original_dataframe["class"].astype('category')

    if cat_dict is None:
        cat_dict_r = dict(enumerate(original_dataframe["class"].cat.categories))
        cat_dict_ = dict()

        for key, value in cat_dict_r.items():
            cat_dict_[value] = key
    else:
        cat_dict_ = cat_dict

    original_dataframe["class"] = original_dataframe["class"].map(cat_dict_)
    original_dataframe["class"] = original_dataframe["class"].astype('int64')

    train_np = original_dataframe.values

    trainx, trainy = train_np[:, :-1], train_np[:, -1]
    trainy = np.array(trainy).astype(int)

    return trainx, trainy, cat_dict_


class dataset_train(Dataset):

    def __init__(self, data, cat_dict, original_label_provided=True):
        self.x = data[0]
        self.y = data[1]
        if original_label_provided:
            self.y_original = data[2]
        self.cat_dict = cat_dict

    def set_x(self, new_x):
        self.x = new_x

    def get_x(self):
        return self.x

    def get_cat_dict(self):
        return self.cat_dict

    def set_y(self, new_y):
        self.y = new_y

    def get_y(self):
        return self.y

    def get_original_y(self):
        return self.y_original

    def get_feature_shape(self):
        return self.x.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return torch.from_numpy(np.array(self.x[idx])), torch.LongTensor(np.array(self.y))[idx], \
               torch.LongTensor([idx]).squeeze()

    def get_weight(self):

        trYunique, trYcounts = np.unique(self.y, return_counts=True)

        max_weight = np.max(trYcounts) / np.min(trYcounts)

        max_count = 0
        for i in range(len(trYcounts)):
            if trYunique[i] != -1 and trYcounts[i] > max_count:
                max_count = trYcounts[i]

        labels = list(self.cat_dict.values())
        no_labels = len(labels)
        weights = np.ones(no_labels)
        for i in range(len(trYunique)):
            if trYunique[i] >= 0 and trYcounts[i] > 0:
                weights[int(trYunique[i])] = min(max_weight, max_count / trYcounts[i])

        return weights

    def add_sample(self, sample_X, sample_Y):
        if len(sample_X) == 1:
            self.x = np.concatenate([self.x, np.expand_dims(sample_X, axis=0)], axis=0)
            self.y = np.append(self.y, sample_Y)
        else:
            self.x = np.concatenate([self.x, sample_X], axis=0)
            self.y = np.concatenate([self.y, sample_Y], axis=0)

    def filter(self, given_X, given_Y):
        new_lx = []
        new_ly = []
        for i in range(len(self.x)):
            if given_Y[i] >= 0:
                new_lx.append(given_X[i])
                new_ly.append(given_Y[i])

        self.x = np.array(new_lx)
        self.y = np.array(new_ly)


class dataset_test(Dataset):

    def __init__(self, cat_dict, file_path='dataset/KDDTest+.csv', stratified_test=False):
        test = pd.read_csv(file_path)
        if stratified_test:
            test = stratified_sample_df(test)
        self.x, self.y, self.cat_dict = preprocess_dataframe(test, cat_dict)
        self.feature_size = self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def set_x(self, new_x):
        self.x = new_x

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.LongTensor(np.array(self.y))[idx], \
               torch.LongTensor([idx]).squeeze()


def get_label_masked(original_train_Y, label_ratio):
    masked_train_Y = original_train_Y.copy()

    trYunique, trYcounts = np.unique(original_train_Y, return_counts=True)
    got_once = np.zeros(len(trYunique))

    for i in range(len(original_train_Y)):
        p = np.random.rand()
        if p > label_ratio and got_once[int(original_train_Y[i])] == 1:
            masked_train_Y[i] = -1
        else:
            got_once[int(original_train_Y[i])] = 1

    return masked_train_Y


def stratified_sample_df(df):
    n = df['class'].value_counts().min()
    df_ = df.groupby('class').apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def get_training_data(label_ratio, no_samples=125973, file_path='dataset/KDDTrain+.csv', stratified=False):
    train = pd.read_csv(file_path)
    if stratified:
        train = stratified_sample_df(train)
    elif no_samples == -1:
        train = train.sample(n=125973, random_state=1)
    else:
        # print(train.columns)
        # train = train.groupby('class', group_keys=False).apply(lambda x: x.sample(min(len(x), no_samples)))
        train = train.sample(n=no_samples, random_state=1)

    print("Total Training Size: " + str(train.shape[0]))

    train_X, original_train_Y, cat_dict = preprocess_dataframe(train)

    train_Y = get_label_masked(original_train_Y, label_ratio)

    labeled_data_X = []
    labeled_data_Y = []
    original_labeled_data_Y = []

    unlabeled_data_X = []
    unlabeled_data_Y = []
    original_unlabeled_data_Y = []

    for i in range(len(train_Y)):
        if train_Y[i] != -1:
            labeled_data_X.append(list(train_X[i]))
            labeled_data_Y.append(train_Y[i])
            original_labeled_data_Y.append(original_train_Y[i])
        else:
            unlabeled_data_X.append(list(train_X[i]))
            unlabeled_data_Y.append(train_Y[i])
            original_unlabeled_data_Y.append(original_train_Y[i])

    labeled_data = np.array(labeled_data_X), np.array(labeled_data_Y), np.array(original_labeled_data_Y)
    unlabeled_data = np.array(unlabeled_data_X), np.array(unlabeled_data_Y), np.array(original_unlabeled_data_Y)

    if len(unlabeled_data[1]) != 0 and len(labeled_data[1]) != 0:
        total_data_X = np.append(labeled_data[0], unlabeled_data[0], axis=0)
        total_data_Y = np.append(labeled_data[1], unlabeled_data[1], axis=0)
        original_total_data_Y = np.append(labeled_data[2], unlabeled_data[2], axis=0)

    elif len(unlabeled_data[1]) == 0:
        total_data_X = labeled_data[0]
        total_data_Y = labeled_data[1]
        original_total_data_Y = labeled_data[2]
    else:
        total_data_X = unlabeled_data[0]
        total_data_Y = unlabeled_data[1]
        original_total_data_Y = unlabeled_data[2]

    total_data = total_data_X, total_data_Y, original_total_data_Y

    total_dataset = dataset_train(total_data, cat_dict)
    labeled_dataset = dataset_train(labeled_data, cat_dict)
    unlabeled_dataset = dataset_train(unlabeled_data, cat_dict)

    return total_dataset, labeled_dataset, unlabeled_dataset
