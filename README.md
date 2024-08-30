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

- `--batch_size`: Input batch size for training (default: 32).
- `--lr`: Learning rate (default: 0.00036).
- `--gamma`: Learning rate step gamma (default: 0.998).
- `--no_cuda`: Disables CUDA training (default: False).
- `--seed`: Random seed (default: 1234).
- `--local_training_epoch`: Number of local training epochs (default: 1).
- `--malicious_local_training_epoch`: Number of malicious local training epochs (default: 1).
- `--num_nets`: Number of totally available users (default: 100).
- `--part_nets_per_round`: Number of participating clients per FL round (default: 10).
- `--fl_round`: Total number of FL rounds to conduct (default: 100).
- `--device`: Device to set, can take the value of: `cuda` or `cuda:x` (default: `cuda:0`).
- `--dataname`: Dataset to use during the training process (default: `cifar10`).
- `--num_class`: Number of classes for dataset (default: 10).
- `--datadir`: The directory of the dataset (default: `./dataset/`).
- `--partition_strategy`: Dataset partition strategy, iid (`homo`) or non-iid (`hetero-dir`) (default: `homo`).
- `--dir_parameter`: The parameter of the Dirichlet distribution for data partitioning (default: 0.5).
- `--model`: Model to use during the training process (default: `vgg9`).
- `--load_premodel`: Whether to load a pre-trained model at the beginning (default: False).
- `--save_model`: Whether to save the intermediate model (default: False).
- `--client_select`: The strategy for the Parameter Server (PS) to select clients (default: `fix-1`).

### Backdoor Attack Parameters

- `--malicious_ratio`: The ratio of malicious clients (default: 0.2).
- `--trigger_label`: The NO. of the trigger label (range from 0 to 9, default: 0).
- `--semantic_label`: The NO. of the semantic label (range from 0 to 9, default: 2).
- `--poisoned_portion`: Poisoning portion (range from 0 to 1, default: 0.4).
- `--attack_mode`: Attack method used: `none`, `stealthy`, `pgd` (default: `none`).
- `--pgd_eps`: The epsilon for PGD attack (default: 0.05).
- `--backdoor_type`: Backdoor type used: `none`, `trigger`, `semantic`, `edge-case`, `replacement` (default: `none`).
- `--model_scaling`: Model replacement technology scaling factor (default: 1).

### Selection Strategy Parameters

- `--random`: Use random selection strategy (default: False).
- `--PlainCSI`: Use Plain_CSI selection strategy (default: False).
- `--power`: Total power limit for Plain_CSI strategy.
- `--Bandwidth`: Total bandwidth limit for Plain_CSI strategy.
- `--fix_num`: The number of fixed attackers (default: 1).

### Attack Strategy Parameters

- `--PlainCSI_ATTACK`: Use Plain_CSI ATTACK selection (default: False).
- `--victim`: Victim index in the Estimated selected clients in TDoS attack (default: -1).
- `--conspirator_idx`: Conspirator index in the Estimated selected clients in CPE attack (default: 15).
- `--attacker_index`: Attacker index in the H, must be <= K (default: 16).
- `--FL_TDoS`: Use FL_TDoS selection (default: False).
- `--FL_CPE`: Use FL_CPE selection (default: False).

### Untargeted Attack Parameters

- `--untargeted_type`: Untargeted attack type used: `none`, `krum-attack`, `trimmean-attack` (default: `none`).

### Defense Parameters

- `--defense_method`: Defense method used: `none`, `fedavg`, `krum`, `multi-krum`, `bulyan`, `trimmedmean`, `median`, `fltrust` (default: `none`).

## Examples

Here are some example commands to run the framework:

- **Basic Training**
  
  ```bash
  python main.py --batch_size 64 --lr 0.001 --fl_round 50 --model resnet18
  ```

- **Backdoor Attack**
 ```bash
  python main.py --batch_size 32 --malicious_ratio 0.3 --attack_mode stealthy --backdoor_type trigger --fl_round 100 --defense xxxx  --{client selection strategy}
```

## Examples

Contributions are welcome! Please open an issue or submit a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

