import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from idsgan_preprocessor import preprocess3, create_batch1
from model.model_class import Blackbox_IDS
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12345)
torch.manual_seed(12345)

import random

random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('dataset/blackbox_train.csv')
test = pd.read_csv('dataset/KDDTest+.csv')

trainx, trainy, testx, testy = preprocess3(train, test)

input_dim = trainx.shape[1]
output_dim = 1
batch_size = 64
tr_N = len(trainx)
te_N = len(testx)
ids_model = Blackbox_IDS(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(ids_model.parameters(), lr=0.001)
loss_f = torch.nn.BCELoss()
max_epoch = 50
train_losses, test_losses = [], []


def train(x, y):
    ids_model.train()
    batch_x, batch_y = create_batch1(x, y, batch_size)
    run_loss = 0
    for x, y in zip(batch_x, batch_y):
        ids_model.zero_grad()
        x_t = torch.FloatTensor(x).to(device)
        y_t = torch.FloatTensor(y).to(device)
        y_t = torch.reshape(y_t, [batch_size])

        out = ids_model(x_t)
        out = torch.reshape(out, [batch_size])

        loss = loss_f(out, y_t)

        run_loss += loss.item()
        loss.backward()
        optimizer.step()
    return run_loss / tr_N


def test(x, y):
    ids_model.eval()
    batch_x, batch_y = create_batch1(x, y, batch_size)
    run_loss = 0

    with torch.no_grad():
        for x, y in zip(batch_x, batch_y):
            x_t = torch.FloatTensor(x).to(device)
            y_t = torch.FloatTensor(y).to(device)
            y_t = torch.reshape(y_t, [batch_size])

            out = ids_model(x_t)
            out = torch.reshape(out, [batch_size])

            loss = loss_f(out, y_t)

            run_loss += loss.item()
    return run_loss / te_N


def main():
    print("IDS start training")
    print("-" * 100)
    for epoch in range(max_epoch):
        train_loss = train(trainx, trainy)
        test_loss = test(testx, testy)

        x_t = torch.FloatTensor(testx).to(device)

        out = ids_model(x_t)
        out = torch.reshape(out, [len(testy)])
        out_np = out.cpu().detach().numpy()

        ids_pred_label = np.array(out_np > 0.5).astype(int)

        correct = np.sum(np.equal(ids_pred_label, testy))
        acc = correct / len(testy)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"{epoch} : {train_loss} \t {test_loss} \t {acc}")

    print("IDS finished training")

    x_t = torch.FloatTensor(testx).to(device)

    out = ids_model(x_t)
    out = torch.reshape(out, [len(testy)])
    out_np = out.cpu().detach().numpy()

    ids_pred_label = np.array(out_np > 0.5).astype(int)

    conf_mat = confusion_matrix(testy, ids_pred_label)
    print(conf_mat)

    torch.save(ids_model.state_dict(), 'model/IDS.pth')
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
