import numpy as np
import pandas as pd
import torch

from blackbox_wrapper import BlackBoxWrapper
from model.model_class import Blackbox_IDS, Generator
from idsgan_preprocessor import preprocess4
import random

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test = pd.read_csv("dataset/generator_test_combined.csv")

functional_categories = ['intrinsic', 'time_based']
# functional_categories = ['intrinsic', 'time_based', 'host_based']
# functional_categories = ['intrinsic', 'content']


test, raw_attack, _, _, modification_mask = preprocess4(test, functional_categories)
raw_attack = np.array(raw_attack).astype(float)

modification_mask = np.tile(modification_mask, [len(raw_attack), 1])
modification_mask_t = torch.FloatTensor(modification_mask).to(device)

NOISE_SIZE = 27
generator_input_dim = test.shape[1] + NOISE_SIZE
generator_output_dim = test.shape[1]
discriminator_output_dim = 1

random_g = Generator(generator_input_dim, generator_output_dim)
learned_g = Generator(generator_input_dim, generator_output_dim).to(device)

ids_model = BlackBoxWrapper(generator_output_dim, 1)

x_t = torch.FloatTensor(raw_attack).to(device)

out = ids_model(x_t)
out = torch.reshape(out, [len(raw_attack)])
out_np = out.cpu().detach().numpy()

ids_pred_label = np.array(out_np > 0.5).astype(int)
corr = np.sum(ids_pred_label)

original_det_rate = corr / len(raw_attack)

print("original detection rate : {}".format(original_det_rate))


def test_GAN():
    g_param = torch.load('save_model/generator.pth')
    learned_g.load_state_dict(g_param)

    noise = np.random.uniform(0., 1., (len(raw_attack), NOISE_SIZE))
    gen_x = np.concatenate([raw_attack, noise], axis=1)

    gen_x_t = torch.FloatTensor(gen_x).to(device)

    adversarial_attack_modification_l = learned_g(gen_x_t, modification_mask_t)
    adversarial_attack_l = torch.FloatTensor(raw_attack).to(device) + adversarial_attack_modification_l

    # print(np.min(adversarial_attack_modification_l.cpu().detach().numpy(), axis=0))
    # print(np.max(adversarial_attack_modification_l.cpu().detach().numpy(), axis=0))

    out = ids_model(adversarial_attack_l)
    out = torch.reshape(out, [len(adversarial_attack_l)])
    out_np = out.cpu().detach().numpy()

    ids_pred_label = np.array(out_np > 0.5).astype(int)
    corr = np.sum(ids_pred_label)

    adv_det_rate_learned = corr / len(raw_attack)

    return adv_det_rate_learned


print("adversarial detection rate: {}".format(test_GAN()))