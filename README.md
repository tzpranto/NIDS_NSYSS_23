# NIDS_NSYSS_23

## Introduction
A Deep Learning Based Semi-Supervised Network Intrusion Detection System Robust to Adversarial Attacks.
* [Paper Link](https://doi.org/10.1145/3629188.3629189)
* [Unofficial PDF Version](https://mshohrabhossain.buet.ac.bd/pub/23-NSysS-NIDS-Camera.pdf)
## Released Experiments
1. Semi-Supervised Learning Model
2. Adversarial Testing
## Requirements 
* Python 3.8
* pandas 1.5.3
* numpy 1.22.4
* pytorch 2.0.1+cu118
* notebook 6.4.8
## Semi-Supervised Learning Model
Dataset is in `Dataset_NSLKDD_2` directory.
Follow the notebook `Helper Notebooks/NIDS_Trainer.ipynb` for training and testing.
## Semi-Supervised Adversarial Testing
Follow the notebooks `Helper Notebooks/NIDS_Testing_IDSGAN.ipynb` and `Helper Notebooks/BlackBoxIDS.ipynb` for testing.
## Citation
If you use this code for publication, please cite the original paper.
```
@inproceedings{nids-nsyss-2023,
  title={A Deep Learning Based Semi-Supervised Network Intrusion Detection System Robust to Adversarial Attacks},
  author={Mukit Rashid, Syed Md. and Toufikuzzaman, Md. and Hossain, Md. Shohrab},
  booktitle={10th International Conference on Networking Systems and Security (NSysS)},
  year={2023},
  address = {Dhaka, Bangladesh},
  volume={},
  number={},
  pages={25--34},
  doi= {https://doi.org/10.1145/3629188.3629189},
  publisher = {ACM}
}
```

## Contact
For help or issues using this codebase, please submit a GitHub issue.

For personal communication related to this codebase, please contact Md. Toufikuzzaman (md.toufikzaman@gmail.com).
