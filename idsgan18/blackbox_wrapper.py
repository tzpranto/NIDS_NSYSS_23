import torch
from nids_loader import generate_result, label_ratio, cat_dict, device
import numpy as np

class BlackBoxWrapper:
    def __init__(self, input_dim, output_dim, path_to_model='models'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.path_to_model = path_to_model

    def __call__(self, x):
        x = x.cpu().detach().numpy()
        x = x.astype(float)
        y = self.getBinPrediction(x)
        y = torch.FloatTensor(y).to(device)
        return y

    def eval(self):

        return

    def getBinPrediction(self, x):
        pred_Y = generate_result(x, label_ratio_=label_ratio, path_to_model=self.path_to_model)
        bin_pred_y = []
        for y in pred_Y:
            if y == cat_dict["DoS"]:
                bin_pred_y.append(1)
            else:
                bin_pred_y.append(0)
        return np.array(bin_pred_y)
