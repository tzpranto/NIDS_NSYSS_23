from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)

NSLKDD_ATTACK_DICT = {
    'DoS': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm'],
    'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
    'Privilege': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm','httptunnel'],
    'Access': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail',
               'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop','worm'],
    'Normal': ['normal']
}

nslkdd_col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

binary = False

NSLKDD_ATTACK_MAP = dict()
for k, v in NSLKDD_ATTACK_DICT.items():
    for att in v:
        if not binary:
            NSLKDD_ATTACK_MAP[att] = k
        else:
            if k == 'Normal':
                NSLKDD_ATTACK_MAP[att] = k
            else:
                NSLKDD_ATTACK_MAP[att] = 'Attack'

cat_dict = dict()

max_weight = 100


def load_nslkdd(train_data=True, test_data_neg=False):
    nRowsRead = None  # specify 'None' if want to read whole file

    df1 = pd.read_csv('Dataset_NSLKDD_2/KDDTrain+_20Percent.txt', delimiter=',', header=None, names=nslkdd_col_names,
                      nrows=nRowsRead)
    df1.dataframeName = 'KDDTrain+_20Percent'

    df2 = pd.read_csv('Dataset_NSLKDD_2/KDDTest+.txt', delimiter=',', header=None, names=nslkdd_col_names)
    df2.dataframeName = 'KDDTest+.txt'

    df3 = pd.read_csv('Dataset_NSLKDD_2/KDDTest-21.txt', delimiter=',', header=None, names=nslkdd_col_names)
    df3.dataframeName = 'KDDTest-21.txt'

    df1.drop(['difficulty_level'], axis=1, inplace=True)
    df2.drop(['difficulty_level'], axis=1, inplace=True)
    df3.drop(['difficulty_level'], axis=1, inplace=True)

    df1.sample(frac=1)

    obj_cols = df1.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    for col in obj_cols:

        if col != 'label':
            onehot_cols_train = pd.get_dummies(df1[col], prefix=col, dtype='float64')
            onehot_cols_test = pd.get_dummies(df2[col], prefix=col, dtype='float64')
            onehot_cols_test2 = pd.get_dummies(df3[col], prefix=col, dtype='float64')

            idx = 0
            for find_col_idx in range(len(list(df1.columns))):
                if list(df1.columns)[find_col_idx] == col:
                    idx = find_col_idx

            itr = 0
            for new_col in list(onehot_cols_train.columns):
                df1.insert(idx + itr + 1, new_col, onehot_cols_train[new_col].values, True)

                if new_col not in list(onehot_cols_test.columns):
                    zero_col = np.zeros(df2.values.shape[0])
                    df2.insert(idx + itr + 1, new_col, zero_col, True)
                else:
                    df2.insert(idx + itr + 1, new_col, onehot_cols_test[new_col].values, True)

                if new_col not in list(onehot_cols_test2.columns):
                    zero_col = np.zeros(df3.values.shape[0])
                    df3.insert(idx + itr + 1, new_col, zero_col, True)
                else:
                    df3.insert(idx + itr + 1, new_col, onehot_cols_test2[new_col].values, True)

                itr += 1

            del df1[col]
            del df2[col]
            del df3[col]

        else:

            df1[col] = df1[col].map(NSLKDD_ATTACK_MAP)
            df2[col] = df2[col].map(NSLKDD_ATTACK_MAP)
            df3[col] = df3[col].map(NSLKDD_ATTACK_MAP)

            df1[col] = df1[col].astype('category')

            cat_dict_r = dict(enumerate(df1[col].cat.categories))

            for key, value in cat_dict_r.items():
                cat_dict[value] = key

            df1 = df1.replace({col: cat_dict})
            df1[col] = df1[col].astype('int64')

            df2[col] = df2[col].astype('category')
            df2 = df2.replace({col: cat_dict})

            for i in range(len(df2[col])):
                if type(df2[col][i]) is str:
                    df2.at[i, col] = len(cat_dict) + 1

            df2[col] = df2[col].astype('int64')

            df3[col] = df3[col].astype('category')
            df3 = df3.replace({col: cat_dict})

            for i in range(len(df3[col])):
                if type(df3[col][i]) is str:
                    df3.at[i, col] = len(cat_dict) + 1

            df3[col] = df3[col].astype('int64')

    # df1 = df1[df1.labels != cat_dict['Normal']]

    normal_label = cat_dict['Normal']

    train_X = df1.values[:, :-1]
    train_Y = df1.values[:, -1]

    test_X = df2.values[:, :-1]
    test_Y = df2.values[:, -1]

    test_Y = np.array(test_Y).astype(np.int64)

    test_X2 = df3.values[:, :-1]
    test_Y2 = df3.values[:, -1]

    test_Y2 = np.array(test_Y2).astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    test_X2 = scaler.transform(test_X2)

    if train_data:
        return train_X, train_Y
    elif not test_data_neg:
        return test_X, test_Y
    else:
        return test_X2, test_Y2


def load_unsw_nb15(train_data=True):
    nRowsRead = None  # specify 'None' if want to read whole file

    df1 = pd.read_csv('Dataset_UNSW_NB15/UNSW_NB15_train.csv', delimiter=',',
                      nrows=nRowsRead)
    df1.dataframeName = 'UNSW_NB15_train.csv'

    df2 = pd.read_csv('Dataset_UNSW_NB15/UNSW_NB15_test.csv', delimiter=',')
    df2.dataframeName = 'UNSW_NB15_test.csv'

    # print(df1['attack_cat'].unique())

    df1.sample(frac=1)
    df1.drop(['id'], axis=1, inplace=True)
    df2.drop(['id'], axis=1, inplace=True)

    if not binary:
        df1.drop(['label'], axis=1, inplace=True)
        df2.drop(['label'], axis=1, inplace=True)
        lbl = 'attack_cat'
    else:
        df1.drop(['attack_map'], axis=1, inplace=True)
        df2.drop(['attack_map'], axis=1, inplace=True)
        lbl = 'label'

    obj_cols = df1.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    for col in obj_cols:

        if col != lbl:
            onehot_cols_train = pd.get_dummies(df1[col], prefix=col, dtype='float64')
            onehot_cols_test = pd.get_dummies(df2[col], prefix=col, dtype='float64')

            idx = 0
            for find_col_idx in range(len(list(df1.columns))):
                if list(df1.columns)[find_col_idx] == col:
                    idx = find_col_idx

            itr = 0
            for new_col in list(onehot_cols_train.columns):
                df1.insert(idx + itr + 1, new_col, onehot_cols_train[new_col].values, True)

                if new_col not in list(onehot_cols_test.columns):
                    zero_col = np.zeros(df2.values.shape[0])
                    df2.insert(idx + itr + 1, new_col, zero_col, True)
                else:
                    df2.insert(idx + itr + 1, new_col, onehot_cols_test[new_col].values, True)

                itr += 1

            del df1[col]
            del df2[col]

        else:

            df1[col] = df1[col].astype('category')

            cat_dict_r = dict(enumerate(df1[col].cat.categories))

            for key, value in cat_dict_r.items():
                cat_dict[value] = key

            df1 = df1.replace({col: cat_dict})
            df1[col] = df1[col].astype('int64')

            df2[col] = df2[col].astype('category')
            df2 = df2.replace({col: cat_dict})

            for i in range(len(df2[col])):
                if type(df2[col][i]) is str:
                    df2.at[i, col] = len(cat_dict) + 1

            df2[col] = df2[col].astype('int64')

    train_X = df1.values[:, :-1]
    train_Y = df1.values[:, -1]

    test_X = df2.values[:, :-1]
    test_Y = df2.values[:, -1]

    test_Y = np.array(test_Y).astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)

    if train_data:
        return train_X, train_Y
    else:
        return test_X, test_Y


def get_training_data(label_ratio):
    train_X, train_Y = load_nslkdd(True)

    trYunique, trYcounts = np.unique(train_Y, return_counts=True)
    got_once = np.zeros(len(trYunique))

    global max_weight
    max_weight = np.max(trYcounts) / np.min(trYcounts)

    for i in range(len(train_Y)):
        p = np.random.rand()
        if p > label_ratio and got_once[int(train_Y[i])] == 1:
            train_Y[i] = -1
        else:
            got_once[int(train_Y[i])] = 1

    labeled_data_X = []
    labeled_data_Y = []
    unlabeled_data_X = []
    unlabeled_data_Y = []
    for i in range(len(train_Y)):
        if train_Y[i] != -1:
            labeled_data_X.append(list(train_X[i]))
            labeled_data_Y.append(train_Y[i])
        else:
            unlabeled_data_X.append(list(train_X[i]))
            unlabeled_data_Y.append(train_Y[i])

    labeled_data = np.array(labeled_data_X), np.array(labeled_data_Y)
    unlabeled_data = np.array(unlabeled_data_X), np.array(unlabeled_data_Y)

    if len(unlabeled_data[1]) != 0 and len(labeled_data[1]) != 0:
        total_data_X = np.append(labeled_data[0], unlabeled_data[0], axis=0)
        total_data_Y = np.append(labeled_data[1], unlabeled_data[1], axis=0)
    elif len(unlabeled_data[1]) == 0:
        total_data_X = labeled_data[0]
        total_data_Y = labeled_data[1]
    else:
        total_data_X = unlabeled_data[0]
        total_data_Y = unlabeled_data[1]

    total_data = total_data_X, total_data_Y

    total_dataset = dataset_train(total_data)
    labeled_dataset = dataset_train(labeled_data)
    unlabeled_dataset = dataset_train(unlabeled_data)

    return total_dataset, labeled_dataset, unlabeled_dataset


class dataset_train(Dataset):

    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]

    def set_x(self, new_x):
        self.x = new_x

    def get_x(self):
        return self.x

    def set_y(self, new_y):
        self.y = new_y

    def get_y(self):
        return self.y

    def get_feature_shape(self):
        return self.x.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.LongTensor(np.array(self.y[idx])), \
               torch.LongTensor([idx]).squeeze()

    def get_weight(self):

        trYunique, trYcounts = np.unique(self.y, return_counts=True)
        max_count = 0
        for i in range(len(trYcounts)):
            if trYunique[i] != -1 and trYcounts[i] > max_count:
                max_count = trYcounts[i]

        labels = list(cat_dict.values())
        no_labels = len(labels)
        weights = np.ones(no_labels)
        for i in range(len(trYunique)):
            if trYunique[i] >= 0 and trYcounts[i] > 0:
                weights[int(trYunique[i])] = min(max_weight, max_count / trYcounts[i])

        return weights

    def add_sample(self, sample_X, sample_Y):
        self.x = np.concatenate([self.x, np.expand_dims(sample_X, axis=0)], axis=0)
        self.y = np.append(self.y, sample_Y)


class dataset_test(Dataset):

    def __init__(self, test_neg=False):
        self.x, self.y = load_nslkdd(False,test_data_neg=test_neg)
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
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
               torch.LongTensor([idx]).squeeze()
