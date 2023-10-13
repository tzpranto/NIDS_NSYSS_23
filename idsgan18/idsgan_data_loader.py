import numpy as np
import pandas as pd
import torch
from idsgan_preprocessor import preprocess2

np.random.seed(12345)
torch.manual_seed(12345)

import random

random.seed(12345)

NSLKDD_ATTACK_DICT = {
    'DoS': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm'],
    'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
    'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel'],
    'R2L': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail',
            'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm'],
    'Normal': ['normal']
}

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

max_weight = 100

considered_attack_list = ['DoS']

train = pd.read_csv('dataset/KDDTrain+.txt', delimiter=',', header=None, names=NSLKDD_COL_NAMES)
test = pd.read_csv('dataset/KDDTest+.txt', delimiter=',', header=None, names=NSLKDD_COL_NAMES)

train, test = preprocess2(train, test, data_generation=True, attack_map=NSLKDD_ATTACK_MAP)

blackbox_train = pd.DataFrame(columns=NSLKDD_COL_NAMES)
discriminator_train_normal = pd.DataFrame(columns=NSLKDD_COL_NAMES)
generator_train = pd.DataFrame(columns=NSLKDD_COL_NAMES)

GAN_train = pd.DataFrame(columns=NSLKDD_COL_NAMES)

generator_test_attack = pd.DataFrame(columns=NSLKDD_COL_NAMES)
generator_test_normal = pd.DataFrame(columns=NSLKDD_COL_NAMES)
generator_test_combined = pd.DataFrame(columns=NSLKDD_COL_NAMES)

train_id_dict = dict()

test_id_dict = dict()

for i in range(len(train)):
    train_id_dict.setdefault(train["class"][i], []).append(i)

for i in range(len(test)):
    test_id_dict.setdefault(test["class"][i], []).append(i)

blackbox_train_ids = []
discriminator_train_normal_ids = []
generator_train_ids = []

GAN_train_ids = []

generator_test_attack_ids = []
generator_test_normal_ids = []
generator_test_combined_ids = []



for k, v in train_id_dict.items():
    if k == "Normal":
        np.random.shuffle(v)
        blackbox_train_ids = blackbox_train_ids + v[:(len(v) // 2)]
        discriminator_train_normal_ids = discriminator_train_normal_ids + v[(len(v) // 2):]
    else:
        np.random.shuffle(v)
        if k in considered_attack_list:
            generator_train_ids = generator_train_ids + v[(len(v) // 2):]
        blackbox_train_ids = blackbox_train_ids + v[:(len(v) // 2)]

for k, v in test_id_dict.items():
    if k == "Normal":
        generator_test_normal_ids = generator_test_normal_ids + v
    else:
        if k in considered_attack_list:
            generator_test_attack_ids = generator_test_attack_ids + v

generator_test_combined_ids = generator_test_combined_ids + generator_test_normal_ids
generator_test_combined_ids = generator_test_combined_ids + generator_test_attack_ids

GAN_train_ids = GAN_train_ids + discriminator_train_normal_ids
GAN_train_ids = GAN_train_ids + generator_train_ids

blackbox_train = train.iloc[blackbox_train_ids]
pd.DataFrame.to_csv(blackbox_train, 'dataset/blackbox_train.csv', index=False)

discriminator_train_normal = train.iloc[discriminator_train_normal_ids]
pd.DataFrame.to_csv(discriminator_train_normal, 'dataset/discriminator_train_normal.csv', index=False)

generator_train = train.iloc[generator_train_ids]
pd.DataFrame.to_csv(generator_train, 'dataset/generator_train.csv', index=False)

generator_test_attack = test.iloc[generator_test_attack_ids]
pd.DataFrame.to_csv(generator_test_attack, 'dataset/generator_test_attack.csv', index=False)

generator_test_normal = test.iloc[generator_test_normal_ids]
pd.DataFrame.to_csv(generator_test_normal, 'dataset/generator_test_normal.csv', index=False)

generator_test_combined = test.iloc[generator_test_combined_ids]
pd.DataFrame.to_csv(generator_test_combined, 'dataset/generator_test_combined.csv', index=False)

GAN_train = train.iloc[GAN_train_ids]
pd.DataFrame.to_csv(GAN_train, 'dataset/GAN_train.csv', index=False)

# pd.DataFrame.to_csv(test, 'dataset/KDDTest+.csv', index=False)
# pd.DataFrame.to_csv(train, 'dataset/KDDTrain+.csv', index=False)
