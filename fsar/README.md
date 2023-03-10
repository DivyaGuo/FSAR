# FSAR: Federated Skeleton-based Action Recognition by Revisiting Human Topology

## Installation

```shell
# Install python environment
$ conda create -n fsar python=3.8.2
$ conda activate fsar

# Install PyTorch
$ pip install torch==1.4.0

# Install other python libraries
$ pip install -r requirements.txt
```

## Dataset setup

For all the datasets, be sure to read and follow their license agreements, and cite them accordingly. The datasets we used are as follows:

- [NTU RGB+D 60](https://arxiv.org/pdf/1604.02808.pdf)
- [NTU RGB+D 120](https://arxiv.org/pdf/1905.04757.pdf)
- [PKU MMD (Part I)](https://arxiv.org/pdf/1703.07475.pdf)
- [PKU MMD (Part II)](https://arxiv.org/pdf/1703.07475.pdf)
- [UESTC](https://arxiv.org/pdf/1904.10681.pdf)
- [Kinetics](https://arxiv.org/pdf/1705.06950.pdf)

## Train the model

To train the model under the federated-by-dataset or federated-by-class scenarios:

```bash
cd ./application/fedhar/
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes.

- [EasyFL](https://github.com/EasyFL-AI/EasyFL)

- [ST-GCN](https://github.com/yysijie/st-gcn)

## Licence

This project is licensed under the terms of the MIT license.
