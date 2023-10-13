import pandas as pd
import numpy as np
import pprint
from pickle import dump

from sklearn.preprocessing import StandardScaler

pp = pprint.PrettyPrinter(indent=4)

NSLKDD_COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
                    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty_level"]

FEATURE_CATEGORY = dict()

for i in range(9):
    FEATURE_CATEGORY.setdefault('intrinsic', []).append(NSLKDD_COL_NAMES[i])

for i in range(9, 22):
    FEATURE_CATEGORY.setdefault('content', []).append(NSLKDD_COL_NAMES[i])

for i in range(22, 31):
    FEATURE_CATEGORY.setdefault('time_based', []).append(NSLKDD_COL_NAMES[i])

for i in range(32, 41):
    FEATURE_CATEGORY.setdefault('host_based', []).append(NSLKDD_COL_NAMES[i])


FEATURE_CATEGORY_REV = dict()

for k,v in FEATURE_CATEGORY.items():
    for cat in list(v):
        FEATURE_CATEGORY_REV[cat] = k


FEATURE_TYPE = dict()

for i in [2, 3, 4]:
    FEATURE_TYPE.setdefault('categorical', []).append(NSLKDD_COL_NAMES[i - 1])

for i in [7, 12, 14, 20, 21, 22]:
    FEATURE_TYPE.setdefault('binary', []).append(NSLKDD_COL_NAMES[i - 1])

for i in [8, 9, 15, 43] + list(range(23, 42)):
    FEATURE_TYPE.setdefault('discrete', []).append(NSLKDD_COL_NAMES[i - 1])

for i in [1, 5, 6, 10, 11, 13, 16, 17, 18, 19]:
    FEATURE_TYPE.setdefault('continuous', []).append(NSLKDD_COL_NAMES[i - 1])


def create_batch1(x, y, batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    y = y[a]

    batch_x = [x[batch_size * i: (i + 1) * batch_size, :].tolist() for i in range(len(x) // batch_size)]
    batch_y = [y[batch_size * i: (i + 1) * batch_size].tolist() for i in range(len(x) // batch_size)]
    return batch_x, batch_y


def create_batch2(x, batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    batch_x = [x[batch_size * i: (i + 1) * batch_size, :] for i in range(len(x) // batch_size)]
    return np.array(batch_x).astype(float)


def preprocess4(train, functional_categories=None):
    if functional_categories is None:
        functional_categories = ['intrinsic', 'time_based']
    train["class"] = train["class"].map(lambda x: 1 if x != "Normal" else 0)

    raw_attack = np.array(train[train["class"] == 1])[:, :-1]
    normal = np.array(train[train["class"] == 0])[:, :-1]
    true_label = train["class"]

    del train["class"]

    train_columns = list(train.columns)

    modification_mask = np.ones((len(train.columns)))
    if functional_categories is None:
        functional_categories = ['intrinsic', 'time_based']

    for cat_ in functional_categories:
        cat_cols = list(FEATURE_CATEGORY[cat_])
        for cat_col in cat_cols:
            for idx in range(len(train_columns)):
                if str(train_columns[idx]).startswith(cat_col):
                    modification_mask[idx] = 0

    return train, raw_attack, normal, true_label, modification_mask


# all
def preprocess2(train, test, data_generation=False, attack_map=None):
    train.drop(['difficulty_level'], axis=1, inplace=True)
    test.drop(['difficulty_level'], axis=1, inplace=True)

    train.sample(frac=1)

    obj_cols = train.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    for col in obj_cols:

        if col != 'class':
            onehot_cols_train = pd.get_dummies(train[col], prefix=col, dtype='float64')
            onehot_cols_test = pd.get_dummies(test[col], prefix=col, dtype='float64')

            idx = 0
            for find_col_idx in range(len(list(train.columns))):
                if list(train.columns)[find_col_idx] == col:
                    idx = find_col_idx

            itr = 0
            for new_col in list(onehot_cols_train.columns):
                train.insert(idx + itr + 1, new_col, onehot_cols_train[new_col].values, True)

                if new_col not in list(onehot_cols_test.columns):
                    zero_col = np.zeros(test.values.shape[0])
                    test.insert(idx + itr + 1, new_col, zero_col, True)
                else:
                    test.insert(idx + itr + 1, new_col, onehot_cols_test[new_col].values, True)

                itr += 1

            del train[col]
            del test[col]

    if not data_generation:
        train["class"] = train["class"].map(lambda x: 1 if x != "Normal" else 0)
        test["class"] = test["class"].map(lambda x: 1 if x != "Normal" else 0)
    else:
        train["class"] = train["class"].map(attack_map)
        test["class"] = test["class"].map(attack_map)

    columns = train.columns
    train["num_outbound_cmds"] = train["num_outbound_cmds"].map(lambda x: 0)
    test["num_outbound_cmds"] = test["num_outbound_cmds"].map(lambda x: 0)

    if not data_generation:
        trainx, trainy = np.array(train[train.columns[train.columns != "class"]]), np.array(train["class"])
        testx, testy = np.array(test[train.columns[train.columns != "class"]]), np.array(test["class"])

        scaler = StandardScaler()
        scaler.fit(trainx)

        trainx = scaler.transform(trainx)
        testx = scaler.transform(testx)

        return trainx, trainy, testx, testy
    else:
        trainx, trainy = train[train.columns[train.columns != "class"]], train["class"]
        testx, testy = test[train.columns[train.columns != "class"]], test["class"]

        columns = trainx.columns
        scaler = StandardScaler()
        scaler.fit(trainx)

        trainx = scaler.transform(trainx)
        testx = scaler.transform(testx)

        train_processed = pd.DataFrame(trainx, columns=columns)
        train_processed["class"] = trainy

        test_processed = pd.DataFrame(testx, columns=columns)
        test_processed["class"] = testy

        return train_processed, test_processed


def preprocess3(train, test):
    train["class"] = train["class"].map(lambda x: 1 if x != "Normal" else 0)
    test["class"] = test["class"].map(lambda x: 1 if x != "Normal" else 0)

    train_np = train.values
    test_np = test.values

    trainx, trainy = train_np[:, :-1], train_np[:, -1]
    testx, testy = test_np[:, :-1], test_np[:, -1]

    testy = np.array(testy).astype(int)
    trainy = np.array(trainy).astype(int)

    return trainx, trainy, testx, testy
