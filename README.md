# FSAR: Federated Skeleton-based Action Recognition with Adaptive Topology Structure and Knowledge Distillation

> Jingwen Guo, Hong Liu, Shitong Sun, Tianyu Guo, Ming Zhang, and Chenyang Si
>
> ICCV2023, in submission

Existing skeleton-based action recognition methods typically follow a centralized learning paradigm, which can pose privacy concerns when exposing human-related videos. Federated Learning (FL) has attracted much attention due to its outstanding advantages in privacy-preserving. However, directly applying FL approaches to skeleton videos suffers from unstable training. In this paper, we investigate and discover that the heterogeneous human topology graph structure is the crucial factor hindering training stability. To address this issue, we pioneer a novel Federated Skeleton-based Action Recognition (FSAR) paradigm, which enables the construction of a globally generalized model without accessing local sensitive data. Specifically, we introduce an Adaptive Topology Structure (ATS), separating generalization and personalization by learning a domain-invariant topology shared across clients and a domain-specific topology decoupled from global model aggregation. Furthermore, we explore Multi-grain Knowledge Distillation (MKD) to mitigate the discrepancy between clients and the server caused by distinct updating patterns through aligning shallow block-wise motion features. Extensive experiments on multiple datasets demonstrate that FSAR outperforms state-of-the-art FL-based methods while inherently protecting privacy for skeleton-based action recognition.



<img src=".\figure\motivation.png" alt="motivation" style="zoom: 33%;" />

The local clients are optimized with our proposed Adaptive Topology Structure (ATS) and Multi-grain Knowledge Distillation (MKD) modules on private data and then perform the client-server collaborative learning iteratively: (i) clients train local models; (ii) clients upload parameters to server; (iii) server aggregates model parameters; (iv) clients download the aggregated models. Moreover, the ATS module extracts the intrinsic structure information of heterogeneous skeleton data, and the MKD module bridges the divergence between the clients and the server.

![architecture](D:\研二下\github\figure\architecture.png)

## Requirements

  ![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)    ![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.4-blue.svg)

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