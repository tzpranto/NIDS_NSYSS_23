import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from blackbox_wrapper import BlackBoxWrapper
from idsgan_preprocessor import preprocess4, create_batch2
from model.model_class import Generator, Discriminator

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

functional_categories = ['intrinsic', 'time_based']
#functional_categories = ['intrinsic', 'time_based', 'host_based']
#functional_categories = ['intrinsic', 'content']


data = pd.read_csv("dataset/GAN_train.csv")
train_data, raw_attack, normal, true_label, modification_mask \
    = preprocess4(data, functional_categories=functional_categories)

test = pd.read_csv("dataset/generator_test_combined.csv")
test, test_raw_attack, _, _, modification_mask_test = preprocess4(test, functional_categories)
test_raw_attack = np.array(test_raw_attack).astype(float)

modification_mask_test = np.tile(modification_mask_test, [len(test_raw_attack), 1])
modification_mask_test_t = torch.FloatTensor(modification_mask_test).to(device)

BATCH_SIZE = 256  # Batch size
GEN_ITERS = 20
CRITIC_ITERS = 1  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
MAX_EPOCH = 500  # How many generator iterations to train for
NOISE_SIZE = 27
generator_input_dim = train_data.shape[1] + NOISE_SIZE
generator_output_dim = train_data.shape[1]
discriminator_output_dim = 1
CLAMP = 0.01

# read parameters of IDS

modification_mask = np.tile(modification_mask, [BATCH_SIZE, 1])
modification_mask_t = torch.FloatTensor(modification_mask).to(device)

ids_model = BlackBoxWrapper(generator_output_dim, 1)

# read model


def test_GAN(learned_g):
    # g_param = torch.load('save_model/generator.pth')
    # learned_g.load_state_dict(g_param)
    noise = np.random.uniform(0., 1., (len(test_raw_attack), NOISE_SIZE))
    gen_x = np.concatenate([test_raw_attack, noise], axis=1)

    gen_x_t = torch.FloatTensor(gen_x).to(device)

    adversarial_attack_modification_l = learned_g(gen_x_t, modification_mask_test_t)
    adversarial_attack_l = torch.FloatTensor(test_raw_attack).to(device) + adversarial_attack_modification_l

    out = ids_model(adversarial_attack_l)
    out = torch.reshape(out, [len(adversarial_attack_l)])
    out_np = out.cpu().detach().numpy()

    ids_pred_label = np.array(out_np > 0.5).astype(int)
    corr = np.sum(ids_pred_label)

    adv_det_rate_learned = corr / len(test_raw_attack)

    return adv_det_rate_learned


generator = Generator(generator_input_dim, generator_output_dim).to(device)
discriminator = Discriminator(generator_output_dim, discriminator_output_dim).to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0001)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0001)

batch_attack = create_batch2(raw_attack, BATCH_SIZE)
batch_normal = create_batch2(normal, BATCH_SIZE)

d_losses, g_losses = [], []
ids_model.eval()
generator.train()
discriminator.train()
cnt = -5
print("IDSGAN start training")
print("-" * 100)

gen_attack_batch_idx = 0
dis_normal_batch_idx = 0
adv_dr = 1.00

for epoch in range(MAX_EPOCH):

    #  Train Generator

    gen_loss = 0.
    for g in range(GEN_ITERS):

        for p in generator.parameters():
            p.requires_grad = True

        for p in discriminator.parameters():
            p.requires_grad = False

        ba = batch_attack[gen_attack_batch_idx]
        gen_attack_batch_idx = (gen_attack_batch_idx + 1) % len(batch_attack)

        optimizer_G.zero_grad()

        noise = np.random.uniform(0., 1., (BATCH_SIZE, NOISE_SIZE))
        gen_x = np.concatenate([ba, noise], axis=1)

        gen_x_t = torch.FloatTensor(gen_x).to(device)

        adversarial_attack_modification = generator(gen_x_t, modification_mask_t)
        adversarial_attack = torch.FloatTensor(ba).to(device) + adversarial_attack_modification

        D_pred = discriminator(adversarial_attack)
        g_loss = torch.mean(D_pred)
        # print("G loss")
        # print(g_loss)
        # print("\n")

        g_loss.backward()
        optimizer_G.step()

        for p in generator.parameters():
            p = torch.clamp(p, -0.01, 0.01)

        gen_loss += g_loss.item()

    gen_loss = gen_loss / GEN_ITERS

    for p in discriminator.parameters():
        p.requires_grad = True

    for p in generator.parameters():
        p.requires_grad = False

    dis_loss = 0.
    for c in range(CRITIC_ITERS):
        optimizer_D.zero_grad()

        np.random.shuffle(batch_normal)

        bn = batch_normal[dis_normal_batch_idx]
        ba = batch_attack[dis_normal_batch_idx % len(batch_attack)]
        dis_normal_batch_idx = (dis_normal_batch_idx + 1) % len(batch_normal)

        noise = np.random.uniform(0., 1., (BATCH_SIZE, NOISE_SIZE))
        gen_x = np.concatenate([ba, noise], axis=1)

        gen_x_t = torch.FloatTensor(gen_x).to(device)

        adversarial_attack_modification = generator(gen_x_t, modification_mask_t)
        adversarial_attack_t = torch.FloatTensor(ba).to(device) + adversarial_attack_modification
        adversarial_attack = adversarial_attack_t.cpu().detach().numpy()

        ids_input = np.concatenate([bn, adversarial_attack], axis=0)

        l = list(range(len(ids_input)))
        np.random.shuffle(l)

        ids_input = ids_input[l]

        ids_input = torch.FloatTensor(ids_input).to(device)

        out = ids_model(ids_input)
        out = torch.reshape(out, [len(ids_input)])
        out_np = out.cpu().detach().numpy()

        ids_pred_label = np.array(out_np > 0.5).astype(int)

        pred_normal = ids_input.cpu().detach().numpy()[ids_pred_label == 0]
        pred_attack = ids_input.cpu().detach().numpy()[ids_pred_label == 1]

        # print(len(pred_normal))
        # print(len(pred_attack))

        if len(pred_attack) == 0:
            cnt += 1
            break

        D_normal = discriminator(torch.FloatTensor(pred_normal).to(device))
        D_attack = discriminator(torch.FloatTensor(pred_attack).to(device))
        D_adv = discriminator(adversarial_attack_t)

        loss_normal = torch.mean(D_normal)
        loss_attack = torch.mean(D_attack)
        loss_adv = torch.mean(D_adv)

        # print(loss_normal)
        # print(loss_attack)
        # print(loss_adv)
        # print("\n")

        d_loss = (loss_normal - loss_attack)  # + LAMBDA * gradient_penalty
        dis_loss += d_loss.item()

        d_loss.backward()
        optimizer_D.step()

        for p in discriminator.parameters():
            p = torch.clamp(p, -0.01, 0.01)

    dis_loss = dis_loss / CRITIC_ITERS

    d_losses.append(dis_loss)
    g_losses.append(gen_loss)

    adv_dr_curr = test_GAN(generator)
    if adv_dr_curr < adv_dr or epoch == 0:
        adv_dr = adv_dr_curr
        torch.save(generator.state_dict(), 'save_model/generator.pth')
        torch.save(discriminator.state_dict(), 'save_model/discreminator.pth')

    print("Epoch: {} -- Curr: {}, Min {}".format(epoch, adv_dr_curr, adv_dr))

    if cnt >= 100:
        print("Not exist predicted attack traffic")
        break

print("IDSGAN finish training")
plt.plot(d_losses, label="D_loss")
plt.plot(g_losses, label="G_loss")
plt.legend()
plt.show()