# AirTrojan
**Abstract**: Federated Learning (FL) is a widely adopted
distributed machine learning technique where clients collab-
oratively train a model without sharing their data. A critical
component of FL is client selection, which involves choosing the
necessary number of clients for each training round. Current
client selection algorithms for wireless FL rely on the conditions
of wireless channels but do not account for vulnerabilities from
attacks on these channels, such as channel state information
(CSI) forgery attacks. In this paper, we introduce AirTrojan,
a novel attack vector that targets client selection in FL. Our
key insight is that since the channel state can be manipulated
by attackers, an attacker can adjust their probability of being
chosen as a participant. AirTrojan enhances the feasibility of
adversarial attacks on FL, which usually assume that malicious
clients are always selected as participants. We demonstrate the
effectiveness of AirTrojan by showing how it can disrupt client
selection and facilitate model poisoning attacks on FL. Our
work highlights that it is urgent to add security components to
client selection processes in wireless FL.

<img width="489" alt="image" src="https://github.com/user-attachments/assets/02a0997b-ce7c-4371-99a6-ce06b0f440f9">


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
  - [General Parameters](#general-parameters)
  - [Backdoor Attack Parameters](#backdoor-attack-parameters)
  - [Selection Strategy Parameters](#selection-strategy-parameters)
  - [Attack Strategy Parameters](#attack-strategy-parameters)
  - [Untargeted Attack Parameters](#untargeted-attack-parameters)
  - [Defense Parameters](#defense-parameters)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Arguments

### General Parameters

```bash
- batch_size: Input batch size for training (default: 32).
- lr: Learning rate (default: 0.00036).
--gamma: Learning rate step gamma (default: 0.998).
--no_cuda: Disables CUDA training (default: False).
--seed: Random seed (default: 1234).
--local_training_epoch: Number of local training epochs (default: 1).
--malicious_local_training_epoch: Number of malicious local training epochs (default: 1).
--num_nets: Number of totally available users (default: 100).
--part_nets_per_round: Number of participating clients per FL round (default: 10).
--fl_round: Total number of FL rounds to conduct (default: 100).
--device: Device to set, can take the value of: cuda or cuda:x (default: cuda:0).
--dataname: Dataset to use during the training process (default: cifar10).
--num_class: Number of classes for dataset (default: 10).
--datadir: The directory of the dataset (default: ./dataset/).
--partition_strategy: Dataset partition strategy, iid (homo) or non-iid (hetero-dir) (default: homo).
--dir_parameter: The parameter of the Dirichlet distribution for data partitioning (default: 0.5).
--model: Model to use during the training process (default: vgg9).
--load_premodel: Whether to load a pre-trained model at the beginning (default: False).
--save_model: Whether to save the intermediate model (default: False).
--client_select: The strategy for the Parameter Server (PS) to select clients (default: fix-1).
