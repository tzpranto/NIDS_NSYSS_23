import os
import pickle
import glob
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, pairwise_distances_argmin_min, pairwise_distances, \
    adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier
from torch.nn import Linear
from torch.utils.data import DataLoader

from nids_blackbox_data_generator import get_training_data, dataset_test, dataset_train

debug = False
method = "full_nids"

label_ratio = 1.0
cluster_centroid_ratio = 50
min_samples_leaf_division = 50

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_dataset, labeled_dataset, unlabeled_dataset = get_training_data(no_samples=125973, label_ratio=label_ratio)
cat_dict = total_dataset.get_cat_dict()
print(cat_dict)
print("Total Dataset Shape: ")
print(total_dataset.get_x().shape)
test_dataset = dataset_test(cat_dict)

ae_epoch = 80
pretrain_epoch = 200
train_epoch = 200

num_data = total_dataset.get_x()
labels = total_dataset.get_y()
original_labels = total_dataset.get_original_y()

total_original_label_counts = dict()
distinct_labels, distinct_label_counts = np.unique(labels, return_counts=True)

for i in range(len(distinct_labels)):
    if distinct_labels[i] != -1:
        total_original_label_counts[distinct_labels[i]] = distinct_label_counts[i]

print(total_original_label_counts)

'''
undercomplete autoencoder for embedding.
'''


class AE(nn.Module):

    def __init__(self, n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, 96)
        self.enc_2 = Linear(96, 64)
        self.z_layer = Linear(64, n_z)

        # decoder
        self.dec_1 = Linear(n_z, 64)
        self.dec_2 = Linear(64, 96)
        self.x_bar_layer = Linear(96, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z


class leaf_dnn(nn.Module):

    def __init__(self, n_input, n_output):
        super(leaf_dnn, self).__init__()
        self.fc1 = nn.Linear(n_input, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, n_output)

    def forward(self, x):
        out_1 = torch.relu(self.fc1(x))
        out_2 = torch.relu(self.fc2(out_1))
        out_3 = self.fc3(out_2)

        return out_3


def get_KM_confusion_matrix(clustering):
    all_cluster_centers = clustering.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(num_data, all_cluster_centers)

    cluster_to_labels_dict = dict()
    for j in range(len(closest)):
        if labels[j] >= 0:
            cluster_to_labels_dict.setdefault(closest[j], []).append(labels[j])

    cluster_to_label_dict = dict()
    for k, v in cluster_to_labels_dict.items():
        un, cnt = np.unique(v, return_counts=True)
        un_idx = np.argmax(cnt)
        cluster_to_label_dict[k] = un[un_idx]

    pd = pairwise_distances(X=test_dataset.get_x(), Y=all_cluster_centers)
    pd_s = np.argsort(pd, axis=1)
    # print(pd_s.shape)

    test_Y_pred = np.zeros(test_dataset.get_x().shape[0])
    for i in range(len(pd_s)):
        for j in range(len(pd_s[i])):
            if pd_s[i][j] in cluster_to_label_dict.keys():
                test_Y_pred[i] = cluster_to_label_dict[pd_s[i][j]]
                break

    test_Y_pred = np.array(test_Y_pred)

    print(confusion_matrix(test_dataset.get_y(), test_Y_pred))
    print(classification_report(test_dataset.get_y(), test_Y_pred))

    return


def tree_work(load_cluster_from_file=False):
    if load_cluster_from_file:
        clustering = pickle.load(
            file=open(os.path.join('clustering', 'nslkdd' + "_" + str(cluster_centroid_ratio) + ".pkl"), 'rb'))
    else:

        print("Beginning K-Means Clustering.")
        print("Cluster Centroid Ratio = 1/{}".format(cluster_centroid_ratio))
        print("Total No. of Centroids = {}".format(int(total_dataset.__len__() / cluster_centroid_ratio)))

        clustering = KMeans(n_clusters=int(total_dataset.__len__() / cluster_centroid_ratio), random_state=0)
        print("Clustering Started.")
        cluster_time_start = time.time()
        clustering.fit(num_data)
        cluster_time_end = time.time()
        print("Clustering ended.")
        print("Total Time: {} Seconds".format(cluster_time_end - cluster_time_start))
        print("\n\n")

    cluster_assignment = clustering.predict(num_data)
    print("ARI Score: {}".format(adjusted_rand_score(cluster_assignment, original_labels)))
    print("\n\n")

    all_clusters = dict()
    for j in range(len(cluster_assignment)):
        all_clusters.setdefault(cluster_assignment[j], []).append(num_data[j])

    cluster_to_label_dict = dict()
    for j in range(len(cluster_assignment)):
        if labels[j] != -1:
            cluster_to_label_dict.setdefault(cluster_assignment[j], []).append(labels[j])

    # print("Clusters:")
    # cont_normal_count = 0
    # for k, v in cluster_to_label_dict.items():
    #     print(k)
    #     trYunique, trYcounts = np.unique(v, return_counts=True)
    #     pr_dict = dict()
    #     for i in range(len(trYunique)):
    #         if trYunique[i] == cat_dict['Normal'] and len(trYcounts) > 1:
    #             cont_normal_count += trYcounts[i]
    #         pr_dict[trYunique[i]] = trYcounts[i]
    #     pprint(pr_dict)
    # print("\n")
    # print(cont_normal_count)

    label_to_cluster_dict = dict()
    for k, v in cluster_to_label_dict.items():
        cl_labels, cl_label_counts = np.unique(np.array(v), return_counts=True)  # labeled member count in each cluster

        total_labeled_counts = np.sum(cl_label_counts)
        max_label = np.argmax(cl_label_counts)

        '''
        If a cluster contains more than 10% of all samples of a label present in the training dataset, it is considered 
        important for this label.
        If the majority label is not the normal label and there are no other labels for which this cluster is important,
        having more than 50% of the labeled members would be enough for soft labeling
        If the majority label is the normal label, then all labeled members must be normal for soft labeling.
        In any other case, we do not soft label.
        '''

        imp_for_label = []
        for label, total_label_count in total_original_label_counts.items():
            for j in range(len(cl_labels)):
                if cl_labels[j] == label and cl_label_counts[j] > 0.1 * total_label_count:
                    imp_for_label.append(label)

        if (cl_label_counts[max_label] / total_labeled_counts) > 0.5:
            selected_label = cl_labels[max_label]
            if len(imp_for_label) == 1:
                if imp_for_label[0] == selected_label:
                    if selected_label != int(cat_dict['Normal']):
                        label = selected_label
                        size = len(v)
                        label_to_cluster_dict.setdefault(label, []).append([k, size])
                    else:
                        if len(cl_labels) == 1:
                            label = selected_label
                            size = len(v)
                            label_to_cluster_dict.setdefault(label, []).append([k, size])
            elif len(imp_for_label) == 0:
                if selected_label != int(cat_dict['Normal']):
                    label = selected_label
                    size = len(v)
                    label_to_cluster_dict.setdefault(label, []).append([k, size])
                else:
                    if len(cl_labels) == 1:
                        label = selected_label
                        size = len(v)
                        label_to_cluster_dict.setdefault(label, []).append([k, size])

    '''
    clusters that belong to a particular label, after soft labeling.
    '''
    print("Cluster to Label Done.")
    soft_label_mapping = dict()
    for k, v in label_to_cluster_dict.items():
        for cluster_index in v:
            soft_label_mapping[cluster_index[0]] = k

    '''
    soft labeling particular unlabeled samples.
    Also add this to labeled dataset.
    '''
    print("Soft Label Mapping of Clusters Done.")

    new_samples = []
    new_samples_labels = []

    orignal_to_predicted_label_mapping = dict()  # original --> [predicted]
    total_correct_soft_labeling = 0
    total_incorrect_soft_labeling = 0
    total_unlabeled_samples = 0

    for j in range(len(labels)):
        if labels[j] == -1:
            total_unlabeled_samples += 1
        if labels[j] == -1 and (int(cluster_assignment[j]) in soft_label_mapping.keys()):
            labels[j] = soft_label_mapping[cluster_assignment[j]]
            orignal_to_predicted_label_mapping.setdefault(original_labels[j], []).append(labels[j])
            new_samples.append(num_data[j])
            new_samples_labels.append(labels[j])

            if labels[j] == original_labels[j]:
                total_correct_soft_labeling += 1
            else:
                total_incorrect_soft_labeling += 1

    if len(new_samples) > 0:
        labeled_dataset.add_sample(np.array(new_samples), np.array(new_samples_labels))

    '''
    checking total labeled and soft labeled members.
    '''
    if label_ratio != 1:
        print("Soft Labeling Done. ")
        print("Labeled Samples Percentage: {}%".format(label_ratio * 100))
        print("Total Samples Soft Labeled: {}".format(total_correct_soft_labeling + total_incorrect_soft_labeling))
        print("Total Correct Soft Labeling: {}".format(total_correct_soft_labeling))
        print("Total Incorrect Soft Labeling: {}".format(total_incorrect_soft_labeling))
        print("Total Soft Labeling Frequency: {}".format((total_correct_soft_labeling + total_incorrect_soft_labeling) /
                                                         total_unlabeled_samples))
        print("Soft Labeling Accuracy: {}".format((total_correct_soft_labeling /
                                                   (
                                                           total_correct_soft_labeling + total_incorrect_soft_labeling)) * 100))

        print("\n\n")

    total_soft_label_counts = dict()
    distinct_slabels, distinct_slabel_counts = np.unique(labels, return_counts=True)
    for j in range(len(distinct_slabels)):
        if distinct_slabels[j] != -1:
            total_soft_label_counts[distinct_slabels[j]] = distinct_slabel_counts[j]

    print("Label Count After Soft Labeling: ")
    print(total_soft_label_counts)
    print("\n\n")
    print("Confusion Matrix With KMeans:")
    get_KM_confusion_matrix(clustering)
    print("\n\n")

    total_dataset.set_y(labels)

    dt_X = labeled_dataset.get_x()
    dt_Y = labeled_dataset.get_y().copy()

    # for i in range(len(dt_Y)):
    #     if dt_Y[i] != cat_dict["Normal"]:
    #         dt_Y[i] = 1
    #     else:
    #         dt_Y[i] = 0

    print("labeled members dimensions:")
    print(dt_X.shape)
    print(dt_Y.shape)

    clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=int(len(dt_X) / min_samples_leaf_division))
    clf.fit(dt_X, dt_Y)

    print("Min. Samples Leaf Ratio: 1/{}".format(min_samples_leaf_division))
    print("Min. Samples Per Leaf: {}".format(int(len(dt_X) / min_samples_leaf_division)))
    print("No. of leaves of decision tree: {}".format(clf.get_n_leaves()))

    '''
    saving decision tree, cluster membership info (this is just to skip the clustering step for our faster use) and soft
    labeling info.
    '''

    file = open(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'tree.pkl'), 'wb')
    pickle.dump(clf, file)
    file.close()

    for k in list(all_clusters.keys()):
        if int(k) not in soft_label_mapping.keys():
            soft_label_mapping[int(k)] = int(cat_dict['Normal'])

    file = open(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'soft_label_mapping.pkl'),
                'wb')
    pickle.dump(soft_label_mapping, file)
    file.close()

    if not load_cluster_from_file:
        file = open(os.path.join('clustering', 'nslkdd' + "_" + str(cluster_centroid_ratio) + ".pkl"), 'wb')
        pickle.dump(clustering, file)
        file.close()

    leaf_dataset_X = dict()
    leaf_dataset_Y = dict()

    '''
    finding out corresponding leaf for each training sample.
    '''

    for j in range(len(num_data)):
        leaf = clf.apply([num_data[j]])[0]
        if labels[j] >= 0:
            leaf_dataset_X.setdefault(leaf, []).append(num_data[j])
            leaf_dataset_Y.setdefault(leaf, []).append(labels[j])

    for k, v in leaf_dataset_X.items():
        leaf_dataset_X[k] = np.array(leaf_dataset_X[k])
        leaf_dataset_Y[k] = np.array(leaf_dataset_Y[k])

    print("Sample Distribution in Leaves:")
    for k, v in leaf_dataset_Y.items():
        print(k)
        trYunique, trYcounts = np.unique(v, return_counts=True)
        pr_dict = dict()
        for i in range(len(trYunique)):
            pr_dict[trYunique[i]] = trYcounts[i]
        pprint(pr_dict)
    print("\n")

    return leaf_dataset_X, leaf_dataset_Y


def train_ae(epochs, load_from_file=False,
             save_path=os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'train_ae')):
    '''
    train autoencoder
    '''

    model = AE(total_dataset.get_feature_shape(), 32)
    model.to(device)

    if load_from_file:
        model.load_state_dict(torch.load(save_path))

    else:
        ae_train_ds = total_dataset
        training_data_length = int(0.7 * ae_train_ds.__len__())
        validation_data_length = ae_train_ds.__len__() - training_data_length
        training_data, validation_data = torch.utils.data.random_split(ae_train_ds,
                                                                       [training_data_length, validation_data_length])

        train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
        validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        min_val_loss = 1000000

        for epoch in range(epochs):
            training_loss = 0.
            validation_loss = 0.
            train_batch_num = 0
            val_batch_num = 0

            model.train()
            for batch_idx, (x, _, idx) in enumerate(train_loader):
                x = x.float()
                x = x.to(device)

                train_batch_num = batch_idx

                optimizer.zero_grad()
                x_bar, z = model(x)
                loss = F.mse_loss(x_bar, x)
                training_loss += loss.item()

                loss.backward()
                optimizer.step()

            training_loss /= (train_batch_num + 1)

            model.eval()

            for batch_idx, (x, _, idx) in enumerate(validation_loader):
                x = x.float()
                x = x.to(device)

                val_batch_num = batch_idx

                x_bar, z = model(x)
                loss = F.mse_loss(x_bar, x)
                validation_loss += loss.item()

            validation_loss /= (val_batch_num + 1)

            if epoch % 1 == 0:
                print(
                    "epoch {} , Training loss={:.4f}, Validation loss={:.4f}".format(epoch, training_loss,
                                                                                     validation_loss))

            if epoch == 0 or min_val_loss > validation_loss:
                min_val_loss = validation_loss
                torch.save(model.state_dict(), save_path)

        print("model saved to {}.".format(save_path))
    return model


'''
pretraining using labeled and soft labeled dataset.
'''


def pretrain_leaf_dnn(save_path, ae_save_path, epochs):
    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(
        torch.load(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'train_ae')))
    ae_model.to(device)

    model = leaf_dnn(32, int(max(labels)) + 1)
    model.to(device)

    weights = torch.FloatTensor(labeled_dataset.get_weight()).to(device)

    train_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)  # soft label must be assigned

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    min_train_loss = 1000000

    for epoch in range(epochs):
        train_loss = 0.0
        train_batch_num = 0
        train_num_correct = 0
        train_num_examples = 0

        model.train()
        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            train_batch_num = batch_idx

            optimizer.zero_grad()
            optimizer_ae.zero_grad()

            x_emb = ae_model(x)[1]
            y_pred = model(x_emb)

            y_t = y_t.clone().detach().to(device)

            loss = torch.nn.CrossEntropyLoss(weight=weights)(y_pred, y_t)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer_ae.step()

            correct = torch.eq(torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1], y_t).view(-1)
            train_num_correct += torch.sum(correct).item()
            train_num_examples += correct.shape[0]

        train_loss /= (train_batch_num + 1)
        train_acc = train_num_correct / train_num_examples

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.4f} T Accuracy={:.4f}".
                  format(epoch, train_loss, train_num_correct / train_num_examples))

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), save_path)
            torch.save(ae_model.state_dict(), ae_save_path)

    print("model saved to {}.".format(save_path))

    return model


'''
The dictionary of X-Y training dataset for each individual leaf.
'''
'''
training dataset of an individual leaf.
'''


def train_leaf_dnn(model, dataset, save_path, epochs):
    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(
        torch.load(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'train_ae')))
    ae_model.to(device)

    weights = torch.FloatTensor(dataset.get_weight()).to(device)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # soft label must be assigned

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    min_train_loss = 1000000
    prev_train_acc = 0
    stop_flag = 1

    for epoch in range(epochs):
        train_loss = 0.0
        train_batch_num = 0
        train_num_correct = 0
        train_num_examples = 0

        model.train()
        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            train_batch_num = batch_idx

            optimizer.zero_grad()

            x_emb = ae_model(x)[1]
            y_pred = model(x_emb)

            y_t = y_t.clone().detach().to(device)

            loss = torch.nn.CrossEntropyLoss(weight=weights)(y_pred, y_t)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1], y_t).view(-1)
            train_num_correct += torch.sum(correct).item()
            train_num_examples += correct.shape[0]

        train_loss /= (train_batch_num + 1)
        train_acc = train_num_correct / train_num_examples

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.4f} T Accuracy={:.4f}".
                  format(epoch, train_loss, train_num_correct / train_num_examples))

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), save_path)

        if min_train_loss < 0.0001:
            break

    print("model saved to {}.".format(save_path))

    return model


'''
training all leaves.
'''


def create_leaf_dnns(leaf_dataset_X, leaf_dataset_Y, remove_priors=True, epochs=150):
    if not os.path.exists(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'leaf_models')):
        os.mkdir(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'leaf_models'))

    if remove_priors:
        filelist = glob.glob(
            os.path.join(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'leaf_models'), "*"))
        for f in filelist:
            os.remove(f)

    for key in leaf_dataset_Y.keys():
        dataset_X = leaf_dataset_X[key]
        dataset_Y = leaf_dataset_Y[key]

        # print(key)
        # print(dataset_X.shape)
        # print(dataset_Y.shape)
        # print("\n")

        # un = np.unique(dataset_Y)
        # if len(un) == 1:
        #     print("Unique leaf "+str(key))
        #     continue

        data = dataset_X, dataset_Y
        dataset = dataset_train(data, cat_dict, original_label_provided=False)

        save_path = os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio), 'leaf_models',
                                 'leaf_' + str(key))

        model = leaf_dnn(32, int(max(labels)) + 1)
        model.load_state_dict(torch.load(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio),
                                                      'pretrain_leaf_dnn')))
        model.to(device)

        train_leaf_dnn(model, dataset, save_path, epochs)


def generate_result(test_X, label_ratio_, test_Y=None, path_to_model='models'):
    clf = pickle.load(
        file=open(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_), 'tree.pkl'), 'rb'))
    soft_label_mapping = pickle.load(
        file=open(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_),
                               'soft_label_mapping.pkl'), 'rb'))

    clustering = pickle.load(
        file=open(os.path.join('clustering', 'nslkdd' + "_" + str(cluster_centroid_ratio) + ".pkl"),
                  'rb'))

    leaf_nodes = clf.apply(test_X)
    cluster_assignment = clustering.predict(test_X)

    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(
        torch.load(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_), 'train_ae')))
    ae_model.to(device)

    pretrained_model = leaf_dnn(32, int(max(labels)) + 1)
    pretrained_model.load_state_dict(
        torch.load(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_),
                                'pretrain_leaf_dnn')))
    pretrained_model.to(device)

    '''
    If model does not exist for a particular leaf, use the default pretrained model.
    '''
    model_dict = dict()
    leaf_model_files = os.listdir(
        os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_), 'leaf_models'))
    for file in leaf_model_files:
        spl = str(file).split("_")
        leaf_model = leaf_dnn(32, int(max(labels)) + 1)
        if not os.path.exists(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_),
                                           'leaf_models', 'leaf_' + spl[1])):
            leaf_model.load_state_dict(
                torch.load(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_),
                                        'pretrain_leaf_dnn')))
        else:
            leaf_model.load_state_dict(
                torch.load(os.path.join(path_to_model, 'models_' + str(method) + "_" + str(label_ratio_),
                                        'leaf_models', 'leaf_' + spl[1])))
        leaf_model.to(device)
        model_dict[int(spl[1])] = leaf_model

    leaf_pred_dict = dict()
    X_emb = ae_model(torch.FloatTensor(test_X).to(device))[1]

    for k, v in model_dict.items():
        leaf_model = v
        Y = torch.softmax(leaf_model(X_emb), dim=-1)
        Y_ = Y.cpu().detach().numpy()
        leaf_pred_dict[k] = Y_

    y_classwise_preds_pretrained_all = torch.softmax(pretrained_model(X_emb), dim=-1)
    y_classwise_preds_pretrained_all = y_classwise_preds_pretrained_all.cpu().detach().numpy()

    test_Y_pred = np.zeros(test_X.shape[0])

    prediction_dictionary = dict()

    for j in range(len(leaf_nodes)):
        y_classwise_preds_leaf = leaf_pred_dict[leaf_nodes[j]][j]
        y_classwise_preds_pretrained = y_classwise_preds_pretrained_all[j]
        if soft_label_mapping[cluster_assignment[j]] != cat_dict['Normal']:
            y_classwise_preds_leaf[int(cat_dict['Normal'])] = 0
            y_classwise_preds_pretrained[int(cat_dict['Normal'])] = 0

        y_prediction_leaf = np.argmax(y_classwise_preds_leaf)
        y_prediction_pretrained = np.argmax(y_classwise_preds_pretrained)

        # test_Y_pred[j] = y_prediction_leaf

        minority_classes = ["U2R", "R2L"]

        predicted = False
        for i in range(len(minority_classes)):
            if y_prediction_pretrained == cat_dict[minority_classes[i]] \
                    or y_prediction_leaf == cat_dict[minority_classes[i]]:
                test_Y_pred[j] = cat_dict[minority_classes[i]]
                predicted = True
                break
        if not predicted:
            test_Y_pred[j] = y_prediction_leaf

    if test_Y is not None:

        test_Y_binary = np.zeros(test_Y.shape)
        test_Y_pred_binary = np.zeros(test_Y_pred.shape)

        for i in range(len(test_Y_pred)):
            if test_Y[i] != cat_dict['Normal']:
                test_Y_binary[i] = 1
            if test_Y_pred[i] != cat_dict['Normal']:
                test_Y_pred_binary[i] = 1

        print(confusion_matrix(test_Y, test_Y_pred))
        print(classification_report(test_Y, test_Y_pred))

        # print(confusion_matrix(test_Y_binary, test_Y_pred_binary))
        # print(classification_report(test_Y_binary, test_Y_pred_binary))

    return test_Y_pred


def train_model():
    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists('clustering'):
        os.mkdir('clustering')

    if not os.path.exists(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio))):
        os.mkdir(os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio)))

    leaf_dataset_X, leaf_dataset_Y = tree_work(load_cluster_from_file=True)
    # train_ae(save_path=os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio),
    #                                 'train_ae'), epochs=ae_epoch, load_from_file=True)
    # pretrain_leaf_dnn(save_path=os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio),
    #                                          'pretrain_leaf_dnn'),
    #                   ae_save_path=os.path.join('models', 'models_' + str(method) + "_" + str(label_ratio),
    #                                             'train_ae'), epochs=pretrain_epoch)
    # create_leaf_dnns(leaf_dataset_X=leaf_dataset_X, leaf_dataset_Y=leaf_dataset_Y, remove_priors=True,
    #                  epochs=train_epoch)


#
# train_model()
# total_dataset, labeled_dataset, unlabeled_dataset = get_training_data(no_samples=125973, label_ratio=1.0)
# generate_result(total_dataset.get_x(), test_Y=total_dataset.get_y(), label_ratio_=label_ratio)
# generate_result(test_dataset.get_x(), test_Y=test_dataset.get_y(), label_ratio_=label_ratio)

# total_dataset_s, labeled_dataset_s, unlabeled_dataset_s = get_training_data(no_samples=175341,
#                                                                             label_ratio=1.0, stratified=True)
#
# test_dataset_s = dataset_test(cat_dict, stratified_test=True)
#
# generate_result(total_dataset_s.get_x(), test_Y=total_dataset_s.get_y(), label_ratio_=label_ratio)
# generate_result(test_dataset_s.get_x(), test_Y=test_dataset_s.get_y(), label_ratio_=label_ratio)
