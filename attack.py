import copy
import numpy as np
import random
from scipy.spatial import distance
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from scipy.linalg import qr
import matplotlib.pyplot as plt
import csv





'''
def backdoor(network, train_loader, test_loader, threshold=90, device='gpu', lr=1e-4, batch_size=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    acc = 0.0
    attack_acc = 0.0
    while attack_acc < threshold:
        for _, (feature, target) in enumerate(train_loader, 0):
            if np.random.randint(100) == 0:
                clean_feature = (feature.to(device)).view(-1, 784)
                clean_target = target.type(torch.long).to(device)
                optimizer.zero_grad()
                output = network(clean_feature)
                loss = criterion(output, clean_target)
                loss.backward()
                optimizer.step()
            else:
                attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                attack_target = torch.zeros(batch_size, dtype=torch.long).to(device)
                optimizer.zero_grad()
                output = network(attack_feature)
                loss = criterion(output, attack_target)
                loss.backward()
                optimizer.step()

        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (feature.to(device)).view(-1, 784)
                target = target.type(torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        acc = 100. * correct / len(test_loader.dataset)
        print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

        correct = 0
        # attack success rate
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                target = torch.zeros(batch_size, dtype=torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        attack_acc = 100. * correct / len(test_loader.dataset)
        print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
        print(acc, attack_acc)

'''

def anomaly_loss(network):
    """
    Define an anomaly detection loss based on your chosen detection mechanism.
    For instance, this could be based on the p-norm distance between weight matrices.
    """
    # Placeholder: Replace with your specific implementation
    loss = torch.norm(next(network.parameters()), p=2)
    return loss

# def backdoor(network, train_loader, test_loader, threshold=90, device='cpu', lr=1e-4, batch_size=10, alpha=0.5):

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(network.parameters(), lr=lr)

#     acc = 0.0
#     attack_acc = 0.0
#     while (acc < threshold) or (attack_acc < threshold):
#     # while  attack_acc < threshold:
#         for _, (feature, target) in enumerate(train_loader, 0):
#             if np.random.randint(10) == 0:
#                 clean_feature = (feature.to(device)).view(-1, 784)
#                 clean_target = target.type(torch.long).to(device)
#                 optimizer.zero_grad()
#                 output = network(clean_feature)
#                 classification_loss = criterion(output, clean_target)
#             else:
#                 attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
#                 attack_target = torch.zeros(batch_size, dtype=torch.long).to(device)
#                 optimizer.zero_grad()
#                 output = network(attack_feature)
#                 classification_loss = criterion(output, attack_target)
            
#             # Compute the composite loss with the anomaly term
#             loss = alpha * classification_loss + (1 - alpha) * anomaly_loss(network)
#             loss.backward()
#             optimizer.step()

#         # Evaluation for accuracy
#         correct = 0
#         with torch.no_grad():
#             for feature, target in test_loader:
#                 feature = (feature.to(device)).view(-1, 784)
#                 target = target.type(torch.long).to(device)
#                 output = network(feature)
#                 F.nll_loss(output, target, size_average=False).item()
#                 pred = output.data.max(1, keepdim=True)[1]
#                 correct += pred.eq(target.data.view_as(pred)).sum()
#         acc = 100. * correct / len(test_loader.dataset)
#         print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

#         # Attack success rate evaluation
#         correct = 0
#         with torch.no_grad():
#             for feature, target in test_loader:
#                 feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
#                 target = torch.zeros(batch_size, dtype=torch.long).to(device)
#                 output = network(feature)
#                 F.nll_loss(output, target, size_average=False).item()
#                 pred = output.data.max(1, keepdim=True)[1]
#                 correct += pred.eq(target.data.view_as(pred)).sum()
#         attack_acc = 100. * correct / len(test_loader.dataset)
#         print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
#         print(acc, attack_acc)

#     return

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms.functional as TF

def backdoor(network, train_loader, test_loader, threshold=90, device='cpu', lr=1e-4, batch_size=10, alpha=0.5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    def anomaly_detection_term(model):
        # Implement your anomaly detection method here. 
        # For this example, we're using the L2 norm of the model weights.
        return sum(torch.norm(p) for p in model.parameters())

    def combined_loss(output, target, model, alpha):
        Lclass = criterion(output, target)
        Lano = anomaly_detection_term(model)
        return alpha * Lclass + (1 - alpha) * Lano

    acc = 0.0
    attack_acc = 0.0
    while (acc < threshold) or (attack_acc < threshold):
        for _, (feature, target) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            
            if np.random.randint(2) == 0:
                feature = feature.to(device).view(-1, 784)
                target = target.type(torch.long).to(device)
                output = network(feature)
            else:
                feature = TF.erase(feature, 0, 0, 5, 5, 0).to(device).view(-1, 784)
                target = torch.zeros(batch_size, dtype=torch.long).to(device)
                output = network(feature)
                
            loss = combined_loss(output, target, network, alpha)
            loss.backward()
            optimizer.step()

        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = feature.to(device).view(-1, 784)
                target = target.type(torch.long).to(device)
                output = network(feature)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        attack_acc = 100. * correct / len(test_loader.dataset)
        print('\nAttack_accy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
        print(acc, attack_acc)
        # ... (rest of the code)



def criterion_ano(model):
    """Calculate the L2 norm of all the model's parameters."""
    return sum(torch.norm(p, p=2) for p in model.parameters())

def constrain_and_scale(train_loader, backdoor_loader, Gt, weight_class, early_stop_threshold, Eadv, lradv, step_sched, step_rate, scale_factor, device, **model_config):
    # Initialize the attacker's model and loss
    X = type(Gt)().to(device)  # Create a new instance of the model
    X.load_state_dict(Gt.state_dict())  # Copy the weights from Gt to X

    criterion_class = nn.CrossEntropyLoss()  # Classification criterion
    optimizer = optim.SGD(X.parameters(), lr=lradv)  # Assuming you're using SGD; adjust accordingly if not

    for e in Eadv:
        total_loss_backdoor = 0
        num_batches = 0

        # Calculate classification loss on backdoor data for early stopping
        for features, labels in backdoor_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = X(features)
            loss = criterion_class(outputs, labels)
            total_loss_backdoor += loss.item()
            num_batches += 1
        
        if total_loss_backdoor / num_batches < early_stop_threshold:
            break
        
        # Training on local data
        for features, _ in train_loader:
            # Replace local data with backdoor data
            features_backdoor, labels_backdoor = next(iter(backdoor_loader))
            features, labels = features_backdoor.to(device), labels_backdoor.to(device)

            # Compute combined loss
            outputs = X(features)
            loss_class = criterion_class(outputs, labels)
            loss_ano_value = criterion_ano(X)
            loss = weight_class * loss_class + (1 - weight_class) * loss_ano_value

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Reduce learning rate if needed
        if e in step_sched:
            lradv /= step_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lradv

    # Scale up the model before submission
    elt1 = scale_factor * (X - Gt) + Gt

    return elt1




def backdoor_attack(network, train_loader, backdoor_loader, threshold=90, device='cpu', lr=1e-4, Eadv=10, alpha=0.5, epsilon=0.01, lradv_step_schedule=5, step_rate=2, gamma=1):
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    
    # Model's initial state G_t
    G_t = copy.deepcopy(network.state_dict())

    for e in range(Eadv):
        total_loss_backdoor = 0
        num_batches = 0

        # Calculate classification loss on backdoor data for early stopping
        # for features, _, labels in backdoor_loader:

        for features, _, labels, _ in backdoor_loader:


            features, labels = features.to(device), labels.to(device)
            outputs = network(features)
            loss = criterion_class(outputs, labels)
            total_loss_backdoor += loss.item()
            num_batches += 1
        
        if total_loss_backdoor / num_batches < epsilon:
            print("Early stopping due to convergence.")
            break
        
        # Training on local data
        for features, _ in train_loader:
            # Replace local data with backdoor data
            features_backdoor,_, labels_backdoor,_, = next(iter(backdoor_loader))
            features, labels = features_backdoor.to(device), labels_backdoor.to(device)

            # Compute combined loss
            outputs = network(features)
            loss_class = criterion_class(outputs, labels)
            loss_ano_value = criterion_ano(network)  # Make sure to implement this criterion_ano
            loss = alpha * loss_class + (1 - alpha) * loss_ano_value

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Reduce learning rate if needed
        if e % lradv_step_schedule == 0:
            lr /= step_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Scale up the model before submission
    X = copy.deepcopy(network.state_dict())
    for key in X:
        X[key] = gamma * (X[key] - G_t[key]) + G_t[key]
    network.load_state_dict(X)

    return network




def mal_backdoor(mal_train_loaders, train_loaders, network, criterion, optimizer, 
               params_copy, device, mal_visible, t, dist=True, 
               mal_boost=1, path=None):
    
    # Initial weights copy
    start_weights = params_copy.copy()
    
    # Backdoor attack
    delta_mal = backdoor_attack(mal_train_loaders, network, criterion, optimizer, 
                                params_copy, device)
    
    # Other training processes (benign, regular, etc.) can be added here if needed

    return delta_mal





def attack_xie(local_grads, weight, choices, mal_index):
    attack_vec = []
    for i, pp in enumerate(local_grads[0]):
        tmp = np.zeros_like(pp)
        for ji, j in enumerate(choices):
            if j not in mal_index:
                tmp += local_grads[j][i]
        attack_vec.append((-weight) * tmp / len(choices))
    for i in mal_index:
        local_grads[i] = attack_vec
    return local_grads




def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    return w_attacked


def change_weight(w_attack, w_honest, change_rate=0.5):
    w_result = copy.deepcopy(w_honest)
    device = w_attack[list(w_attack.keys())[0]].device
    for k in w_honest.keys():
        w_h = w_honest[k]
        w_a = w_attack[k]

        assert w_h.shape == w_a.shape

        honest_idx = torch.FloatTensor((np.random.random(w_h.shape) > change_rate).astype(np.float)).to(device)
        attack_idx = torch.ones_like(w_h).to(device) - honest_idx

        weight = honest_idx * w_h + attack_idx * w_a
        w_result[k] = weight

    return w_result





'''
# -*-*--*--*--*--*--*--*--*--*-

***  Define the CSI attack and selection ***

# -*-*--*--*--*--*--*--*--*--*-

'''

def Update_CSI_CPI(H, selected_clients, victim_idx, attacker_id, conspirator_id):
    """
    Replace victim client with a new client with a larger effective channel gain
    and smaller interference component, while maintaining spatial compatibility.

    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - selected_clients: list of indices of clients currently active
    - victim_idx: index of victim client to be replaced
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator client whose CSI should be orthogonal to attacker's CSI
    
    Returns:
    - H_new: updated channel matrix with victim replaced by attacker
    """   
    # Decompose victim's channel into effective channel and interference
    gv = H[victim_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(selected_clients)):
        if selected_clients[j] != victim_idx:
            gj = H[selected_clients[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    ev = gv - ev
    
    # Generate replacement channel for victim_idx-1
    alpha_v = np.random.normal(loc=1.5, scale=0.1)
    omega_v = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients)-1,))
    # new_victim_channel = alpha_v * H[victim_idx-1, :] + omega_v.dot(ev)
    
    # Generate replacement channel for attacker_id
    alpha_a = np.random.normal(loc=1.5, scale=0.1)
    omega_a = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))
    ev_consp = np.zeros_like(H[conspirator_id, :])
    for j in range(len(selected_clients)):
        if selected_clients[j] != conspirator_id:
            gj = H[selected_clients[j], :]
            ev_consp += np.vdot(H[conspirator_id, :], gj) / np.vdot(gj, gj) * gj
    ev_a = gv - ev_consp
    new_attacker_channel = alpha_a * H[victim_idx, :] + omega_a.dot(ev_a)
    # new_attacker_channel = alpha_a * H[victim_idx, :] + omega_a.reshape((9,1)).dot(ev_a)

    # Replace victim with attacker
    H_new = np.copy(H)
    H_new[attacker_id, :] = new_attacker_channel
    
    return H_new





def Update_CSI(H, selected_clients, victim_idx, attacker_id):
    """
    Replace victim client with a new client with a larger effective channel gain
    and smaller interference component, while maintaining spatial compatibility.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - selected_clients: list of indices of clients currently active
    - victim_idx: index of victim client to be replaced
    - attacker_id: index of attacker client to replace victim
    
    Returns:
    - H_new: updated channel matrix with victim replaced by attacker
    """
    
    # Decompose victim's channel into effective channel and interference
    gv = H[victim_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(selected_clients)):
        if selected_clients[j] != victim_idx:
            gj = H[selected_clients[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    ev = gv - ev
    
    # Generate replacement channel with larger effective channel gain and smaller interference
    alpha = np.random.normal(loc=1.5, scale=0.1)
    omega = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))
    new_client_channel = alpha * gv + omega.dot(ev)
    
    # Replace victim with attacker
    H_new = np.copy(H)
    H_new[attacker_id, :] = new_client_channel
    
    return H_new


def find_victim(conspirator_idx, S0, H):
    result =[]
    for idx in reversed(S0):
        victim_idx = idx
        gv = H[victim_idx, :]
        gc = H[conspirator_idx, :]
        
        # Calculate the interference and orthogonal components for the victim and conspirator
        interference_victim = np.abs(np.vdot(gv, gc))
        orthogonal_victim = np.linalg.norm(gv - interference_victim)
        
        interference_conspirator = np.abs(np.vdot(gc, gv))
        orthogonal_conspirator = np.linalg.norm(gc - interference_conspirator)
        
        # Compare the conspirator and victim based on orthogonal and interference components
        if orthogonal_conspirator > orthogonal_victim and interference_conspirator < interference_victim:
            result.append(victim_idx) 

    # If no suitable victim is found, return None
    if len(result) == 0:
        return S0[9]
    else:
        return result[-1]


def calculate_effective_channel_gains(H):
    gains = []
    for idx in range(H.shape[0]):
        g = H[idx, :]
        g_norm = np.linalg.norm(g)
        gains.append(g_norm)
    return gains

# gains = calculate_effective_channel_gains(H)
# conspirator_idx = np.argmax(gains)

def client_rank(client_gain, gains):
    sorted_gains = sorted(gains, reverse=True)
    return sorted_gains.index(client_gain)


def categorize_clients(gains, categories):
    categorized_clients = {i: [] for i in range(categories)}
    gain_min, gain_max = min(gains), max(gains)
    step = (gain_max - gain_min) / categories

    for idx, gain in enumerate(gains):
        category = int((gain - gain_min) // step)
        if category == categories:
            category -= 1
        categorized_clients[category].append(idx)

    return categorized_clients

# categories = 10
# categorized_clients = categorize_clients(gains, categories)

# rank_improvements = []

# for category in range(categories):
#     conspirator_candidates = categorized_clients[category]
#     improvements = []

#     for conspirator_idx in conspirator_candidates:
#         original_rank = client_rank(gains[conspirator_idx], gains)

        # Modify the selection algorithm or other factors to improve the rank of the conspirator
        # For example, you can increase the conspirator's channel gain by a certain factor





def Update_CSI_CPE(H,K,N,conspirator_idx,malicious_idx,alpha,omega):
    """
    The malicious attacker is involved to help the conspirator. 
    # first we get the clients list 
    # and then we observe the clients list,we get some prospective conspirators.we let attacker replace j-1 th user,and based on this new 
    # clients subset,we can get the client selected  based on j-1 users(including attacker)


    ## t-1 round's the client list: [0,1,2..j-1,j,..N] (0<j<=N) victim j is based on the 0~(j-1) CSI
    ## pre-t round: we set CSI of attakcer to replace j-1,then the promsing list of j-1 users: [0,1,2..j-2,attacker]
    ## t round: when we choose the j-th user,the user is based on the new j-1 users list.We can the feature of the new j-th user.
    ## or we can get a list of promsing consipirators.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - K: K  proposed users 
    - j: index of victim client in the predict list(based on plain-text)
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator with high probablity of being selected in next round
    
    Returns:
    - H_new: updated channel matrix of selecting conspirator in next round with high probablity
    """

    H_new = np.copy(H)
    orthogonal_values = {}
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
    S0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)
    # print("S0_ranked:",S0_ranked[:20])
    
    # Find the indices of the (j-1)-th and j-th users in S0_ranked
    
    victim_idx = find_victim(conspirator_idx, S0_ranked, H)
    # print("victim_idx is:",victim_idx)
    j_minus_1_idx = S0_ranked[S0_ranked.index(victim_idx)-1] 
    # print("victim_idx-1 is:",j_minus_1_idx)




    # Generate replacement channel with larger effective channel gain and smaller interference => move to the args list
    # alpha = np.random.normal(loc=1.0, scale=0.1)
    # omega = np.random.normal(loc=0.3, scale=0.1, size=(len(selected_clients),))

    # Calculate the orthogonal part of the conspirator's channel
    gc = H[conspirator_idx, :]
    interference_conspirator = np.zeros_like(gc)
    
    for j in S0_ranked:
        if j != conspirator_idx:
            gj = H[j, :]
            interference_conspirator += np.vdot(gc, gj) / np.vdot(gj, gj) * gj
    orthogonal_conspirator = gc - interference_conspirator
    
    # Calculate the interference part of the victim's channel
    gv = H[victim_idx, :]
    interference_victim = np.zeros_like(gv)

    for j in S0_ranked:
        if j != victim_idx:
            gj = H[j, :]
            interference_victim += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    orthogonal_victim = gv - interference_victim
    influence_channel_victim = alpha * orthogonal_victim + omega*interference_victim



     # Decompose victim-1's channel into effective channel and interference
    gv = H[j_minus_1_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(S0_ranked)):
        if S0_ranked[j] != j_minus_1_idx:
            gj = H[S0_ranked[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    new_client_channel = alpha * gv + omega*ev
    

    # Calculate the distance between the conspirator and the victim
    distance = abs(S0_ranked.index(conspirator_idx) - S0_ranked.index(victim_idx))

    # distance =  abs(conspirator_idx - S0_ranked.index(victim_idx))

    # Create an empty array for the influence channels
    influence_channels = np.zeros_like(H[0, :])

    # Calculate the influence channels for clients between the conspirator and the victim
    for i in range(1, distance):
        idx = S0_ranked[S0_ranked.index(victim_idx) + i]
        gv = H[idx, :]
        interference = np.zeros_like(gv)

        for j in S0_ranked:
            if j != idx:
                gj = H[j, :]
                interference += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
        orthogonal = gv - interference
        influence_channel = alpha * orthogonal + omega*interference
        influence_channels += influence_channel
    
    # Create the new feedback ha for the malicious user  orthogonal_conspirator + interference_victim +new_client_channel + new_client_channel_victim
   
    # we can influence a  a specific client and change its rank(from a higher --> lower rank)  + influence_channels
    ha = influence_channel_victim + new_client_channel  + influence_channels
    # we improve the selection probablity  of a specific client's client
    # ha = new_client_channel + orthogonal_conspirator+ interference_conspirator
    
    # Replace the (j-1)-th user with the malicious user in the channel matrix
    H_new[malicious_idx, :] = ha
    
    # return H_new,victim_idx,j_minus_1_idx
    return H_new








def Update_CSI_CPE_2(H,K,N,conspirator_idx,malicious_idx):
    """
    The malicious attacker is involved to help the conspirator. 
    # first we get the clients list 
    # and then we observe the clients list,we get some prospective conspirators.we let attacker replace j-1 th user,and based on this new 
    # clients subset,we can get the client selected  based on j-1 users(including attacker)


    ## t-1 round's the client list: [0,1,2..j-1,j,..N] (0<j<=N) victim j is based on the 0~(j-1) CSI
    ## pre-t round: we set CSI of attakcer to replace j-1,then the promsing list of j-1 users: [0,1,2..j-2,attacker]
    ## t round: when we choose the j-th user,the user is based on the new j-1 users list.We can the feature of the new j-th user.
    ## or we can get a list of promsing consipirators.
    
    Inputs:
    - H: channel matrix (N_clients x N_antennas)
    - K: K  proposed users 
    - j: index of victim client in the predict list(based on plain-text)
    - attacker_id: index of attacker client to replace victim
    - conspirator_id: index of conspirator with high probablity of being selected in next round
    
    Returns:
    - H_new: updated channel matrix of selecting conspirator in next round with high probablity
    """

    H_new = np.copy(H)
    orthogonal_values = {}
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
    S0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)
    # print("S0_ranked:",S0_ranked[:20])
    
    # Find the indices of the (j-1)-th and j-th users in S0_ranked
    
    victim_idx = find_victim(conspirator_idx, S0_ranked, H)
    # print("victim_idx is:",victim_idx)
    j_minus_1_idx = S0_ranked[S0_ranked.index(victim_idx)-1] 
    # print("victim_idx-1 is:",j_minus_1_idx)




    # Generate replacement channel with larger effective channel gain and smaller interference
    alpha = np.random.normal(loc=1.2, scale=0.1)
    omega = np.random.normal(loc=0.5, scale=0.1, size=(len(selected_clients),))


    # h_attacker = np.zeros_like(H[malicious_idx, :])
    # for client in S0_ranked:
    #     S0_ranked.remove(client)
    #     gv = H[client, :]
    #     ev = np.zeros_like(gv)
    #     for j in range(len(S0_ranked)):
    #         if S0_ranked[j] != client:
    #             gj = H[S0_ranked[j], :]
    #             ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    #     # ev = gv - ev
    #             new_client_channel = alpha * gv + omega.dot(ev)
    #             h_attacker  =  h_attacker + new_client_channel





    # S0_ranked.remove(victim_idx)
    # S0_ranked.remove(j_minus_1_idx)
    # S0_ranked.append(malicious_idx)

    # Calculate the orthogonal part of the conspirator's channel
    gc = H[conspirator_idx, :]
    interference_conspirator = np.zeros_like(gc)
    
    for j in S0_ranked:
        if j != conspirator_idx:
            gj = H[j, :]
            interference_conspirator += np.vdot(gc, gj) / np.vdot(gj, gj) * gj
    orthogonal_conspirator = gc - interference_conspirator
    
    # Calculate the interference part of the victim's channel
    gv = H[victim_idx, :]
    interference_victim = np.zeros_like(gv)

    for j in S0_ranked:
        if j != victim_idx:
            gj = H[j, :]
            interference_victim += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    orthogonal_victim = gv - interference_victim
    influence_channel_victim = alpha * orthogonal_victim + omega.dot(interference_victim)



     # Decompose victim-1's channel into effective channel and interference
    gv = H[j_minus_1_idx, :]
    ev = np.zeros_like(gv)
    for j in range(len(S0_ranked)):
        if S0_ranked[j] != j_minus_1_idx:
            gj = H[S0_ranked[j], :]
            ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    new_client_channel = alpha * gv + omega.dot(ev)
    
    # Replace victim with attacker
    # H_new = np.copy(H)
    # H_new[attacker_id, :] = new_client_channel

    # gv = H[victim_idx, :]
    # ev = np.zeros_like(gv)
    # for j in range(len(S0_ranked)):
    #     if S0_ranked[j] != victim_idx:
    #         gj = H[S0_ranked[j], :]
    #         ev += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # ev = gv - ev
    # new_client_channel_victim = alpha * gv + omega.dot(ev)

        # Calculate the distance between the conspirator and the victim
    distance = abs(S0_ranked.index(conspirator_idx) - S0_ranked.index(victim_idx))

    # Create an empty array for the influence channels
    influence_channels = np.zeros_like(H[0, :])

    # Calculate the influence channels for clients between the conspirator and the victim
    for i in range(1, distance):
        idx = S0_ranked[S0_ranked.index(victim_idx) + i]
        gv = H[idx, :]
        interference = np.zeros_like(gv)

        for j in S0_ranked:
            if j != idx:
                gj = H[j, :]
                interference += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
        orthogonal = gv - interference
        influence_channel = alpha * orthogonal + omega.dot(interference)
        influence_channels += influence_channel

    # Calculate the new feedback ha for the malicious user
    # ha = influence_channels

    # # Replace the (j-1)-th user with the malicious user in the channel matrix
    # H_new[malicious_idx, :] = ha



    # # Calculate the interference part of the victim's channel
    # gv = H[S0_ranked[10], :]
    # interference_10 = np.zeros_like(gv)

    # for j in S0_ranked:
    #     if j != S0_ranked[10]:
    #         gj = H[j, :]
    #         interference_10 += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # orthogonal_10 = gv - interference_10
    # influence_channel_10 = alpha * orthogonal_10 + omega.dot(interference_10)


    #     # Calculate the interference part of the victim's channel
    # gv =  H[S0_ranked[11], :]
    # interference_11 = np.zeros_like(gv)

    # for j in S0_ranked:
    #     if j != S0_ranked[11]:
    #         gj = H[j, :]
    #         interference_11 += np.vdot(gv, gj) / np.vdot(gj, gj) * gj
    # orthogonal_11 = gv - interference_11
    # influence_channel_11 = alpha * orthogonal_11 + omega.dot(interference_11)

    



    
    # Create the new feedback ha for the malicious user  orthogonal_conspirator + interference_victim +new_client_channel + new_client_channel_victim
   
    # we can influence a  a specific client and change its rank(from a higher --> lower rank)
    ha = influence_channel_victim + new_client_channel + influence_channels
     # we improve the selection probablity  of a specific client's client
    # ha = new_client_channel + orthogonal_conspirator+ interference_conspirator
    
    # Replace the (j-1)-th user with the malicious user in the channel matrix
    H_new[malicious_idx, :] = ha
    
    return H_new,victim_idx,j_minus_1_idx











def Plain_Select_CPE(H, K, N):
    selected_clients = []
    noise_power = 1  # Gaussian noise power
    
    # Spatial compatibility quantization algorithm
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    orthogonal_values = {}
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        orthogonal_values[s_hat_n] = g_norms[idx]
         # Sort S_0 based on orthogonal_values
        S_0_ranked = sorted(S_0, key=lambda x: orthogonal_values[x], reverse=True)

        
        # Check for aligned users and select one with higher gain and lower interference
        for j in S_0:
            if j != s_hat_n:
                if np.abs(np.dot(H[s_hat_n], H[j].conj().T)) > 0.99:
                    alpha = np.dot(H[s_hat_n], H[j].conj().T)
                    gi_norms_j = np.linalg.norm(H[j])
                    gi_norms_selected = np.linalg.norm(H[selected_clients])
                    omega = np.sqrt(1 - np.abs(alpha)**2) * gi_norms_j / gi_norms_selected
                    new_client_channel = alpha * H[s_hat_n] + omega * H[j]
                    if np.linalg.norm(new_client_channel) > np.linalg.norm(H[s_hat_n]):
                        s_hat_n = j
                else:
                    continue
        selected_clients.append(s_hat_n)

        # Compute precoding vector and SINR for selected user
        C = np.linalg.pinv(H[selected_clients])
        m = np.ones((len(selected_clients), 1)) / np.sqrt(len(selected_clients))  # Unit power constraint
        interference = 0.0
        for j in S_0:
            if j != s_hat_n:
                interference += np.abs(np.dot(H[s_hat_n], H[j].conj().T)) ** 2 / np.dot(H[j], H[j].conj().T)
        noise = np.random.randn() * np.sqrt(noise_power)
        SINR = np.abs(np.dot(H[s_hat_n], C[:, i]) * m[i]) ** 2 / (interference + noise)

    return selected_clients






def Plain_Select(H, K, N):
    selected_clients = []
    noise_power = 1  # Gaussian noise power
    
    # Spatial compatibility quantization algorithm
    S_ = set(range(K))  # Set of remaining users
    S_0 = set()  # Set of selected users
    
    for i in range(N):
        g_norms = []
        for s_n in S_:
            # Compute component of channel orthogonal to selected users
            g_n = H[s_n]
            for j in S_0:
                g_n -= np.dot(H[s_n], H[j].conj().T) * H[j] / np.dot(H[j], H[j].conj().T)
            g_norm = np.linalg.norm(g_n)
            g_norms.append(g_norm)
        # Select user with largest g_norm and add to selected set
        idx = np.argmax(g_norms)
        s_hat_n = list(S_)[idx]
        S_0.add(s_hat_n)
        S_.remove(s_hat_n)

        
        # Check for aligned users and select one with higher gain and lower interference
        for j in S_0:
            if j != s_hat_n:
                if np.abs(np.dot(H[s_hat_n], H[j].conj().T)) > 0.99:
                    alpha = np.dot(H[s_hat_n], H[j].conj().T)
                    gi_norms_j = np.linalg.norm(H[j])
                    gi_norms_selected = np.linalg.norm(H[selected_clients])
                    omega = np.sqrt(1 - np.abs(alpha)**2) * gi_norms_j / gi_norms_selected
                    new_client_channel = alpha * H[s_hat_n] + omega * H[j]
                    if np.linalg.norm(new_client_channel) > np.linalg.norm(H[s_hat_n]):
                        s_hat_n = j
                else:
                    continue
        selected_clients.append(s_hat_n)

        # Compute precoding vector and SINR for selected user
        C = np.linalg.pinv(H[selected_clients])
        m = np.ones((len(selected_clients), 1)) / np.sqrt(len(selected_clients))  # Unit power constraint
        interference = 0.0
        for j in S_0:
            if j != s_hat_n:
                interference += np.abs(np.dot(H[s_hat_n], H[j].conj().T)) ** 2 / np.dot(H[j], H[j].conj().T)
        noise = np.random.randn() * np.sqrt(noise_power)
        SINR = np.abs(np.dot(H[s_hat_n], C[:, i]) * m[i]) ** 2 / (interference + noise)

    return selected_clients



   

        


# Replace one client per iteration
# replace attack
# victim_idx = np.random.choice(range(num_selected_clients))
# victim_idx = np.random.randint(num_selected_clients)
# print(victim_idx)

# victim_idx = 0
# new_selected_clients = replace(selected_clients,H, victim_idx)
# print("New---- clients:",new_selected_clients)

# The client selection process use case
# Parameters
# N = 10  # Number of base station antennas
# K = 100  # Total number of clients
# P = 100  # Total transmit power
# trials = 1000

'''
# Attack 1
attackers = [15, 25, 35, 45, 55, 65, 75, 85, 95]
attack1_success_counts = {attacker: 0 for attacker in attackers}

print("Attack 1 ----------------------")

for attacker in attackers:
    for _ in range(trials):
        # Generate and normalize channel matrix
        H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
        total_power = np.sum(np.abs(H)**2)
        H = H * np.sqrt(P / total_power)

        selected_clients = Plain_Select(H, K, N)
        H_new = Update_CSI(H, selected_clients, selected_clients[-1], attacker)
        new_selected_clients = Plain_Select(H_new, K, N)

        if attacker in new_selected_clients:
            attack1_success_counts[attacker] += 1

for attacker, count in attack1_success_counts.items():
    print(f"Attacker {attacker} success rate: {count / trials * 100:.2f}%")
'''

'''
# Attack 2
conspirators = [15, 25, 35, 45, 55, 65, 75, 85, 95]
attack2_success_counts = {conspirator: 0 for conspirator in conspirators}
conspirator_index_changes = {conspirator: [] for conspirator in conspirators}

print("Attack 2 ----------------------")

# Parameters
N = 100  # Number of base station antennas
K = 100  # Total number of clients
P = 100  # Total transmit power



for _ in range(trials):
    # Generate and normalize channel matrix
    H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
    total_power = np.sum(np.abs(H)**2)
    H = H * np.sqrt(P / total_power)
    selected_clients = Plain_Select(H, K, N)

    for conspirator in conspirators:
        Before = selected_clients.index(selected_clients[conspirator])
        
        H_new = Update_CSI_CPE(H, K, N, selected_clients[conspirator], 18, 1.2, 0.2)
        new_selected_clients = Plain_Select(H_new, K, N)

        After = new_selected_clients.index(selected_clients[conspirator])

        conspirator_index_changes[conspirator].append(abs(Before - After))
        
        if selected_clients[conspirator] in new_selected_clients[:20]:
            attack2_success_counts[conspirator] += 1

for conspirator, count in attack2_success_counts.items():
    print(f"Conspirator {conspirator} success rate: {count / trials * 100:.2f}%")

for conspirator, changes in conspirator_index_changes.items():
    print(f"Conspirator {conspirator} maximum index change: {max(changes)}")
'''






'''
# Generate channel matrix with random complex entries
H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
# Compute the total transmit power across all antennas
total_power = np.sum(np.abs(H)**2)
# Normalize the channel matrix to satisfy the power constraint
H = H * np.sqrt(P / total_power)


# print("attack 1----------------------")
selected_clients = Plain_Select(H,K,N)
print(selected_clients)
attacker = 18 
H_new  =  Update_CSI(H,selected_clients,selected_clients[-1],attacker)


# The formal client selection process:


new_selected_clients = Plain_Select(H_new,K,N)
print(new_selected_clients)
if attacker in new_selected_clients:
        print("attack success")
else:
        print("attack failed")



print("attack 2----------------------")

# The client selection process use case
# Parameters
N = 100  # Number of base station antennas
K = 100  # Total number of clients
P = 100  # Total transmit power


# H = np.random.randn(K, N) + 1j * np.random.randn(K, N)


# Generate channel matrix with random complex entries
H = np.random.randn(K, N) + 1j * np.random.randn(K, N)
# Compute the total transmit power across all antennas
total_power = np.sum(np.abs(H)**2)
# Normalize the channel matrix to satisfy the power constraint
H = H * np.sqrt(P / total_power)


# def Update_CSI_CPE(H,K,N,attacker_id,conspirator_id):
#Predict the clients based on Plain text
selected_clients = Plain_Select(H,K,N)
print("Original selected_clients is:",selected_clients[:10])


for conspirator in [15,25,35,45,55,65,75,85,95]:
    before.append(conspirator)
    H_new =  Update_CSI_CPE(H,K,N,selected_clients[conspirator],18,1.2,0.2)
    print("The conspirator is:",selected_clients[conspirator])
    new_selected_clients = Plain_Select(H_new,K,N)
    print(new_selected_clients)
    if selected_clients[conspirator] in new_selected_clients[:10]:
        print("attack success")
    else:
        print("attack failed")




    # print(new_selected_clients[:10])
    # print("Before repalce,the index of conspirator is",selected_clients.index(selected_clients[conspirator]))
    # print("After repalce,the index of conspirator is: ",new_selected_clients.index(selected_clients[conspirator]))
    # after.append(new_selected_clients.index(selected_clients[conspirator]))
    # # Write train_acc and test_accs to a CSV file
    # with open('replace_result_3.csv', mode='w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Count', 'Before CPE', 'After CPE','victim_idx','j_minus_1_idx'])
    #     for epoch in range(len(before)):
    #         writer.writerow([epoch+1, before[epoch], after[epoch],victim_idx,j_minus_1_idx])
    

# plt.plot(before, label='Before CPE')
# plt.plot(after, label='After CPE')
# plt.xlabel('Count')
# plt.ylabel('Rank')
# plt.savefig('rank_conspirator_3.pdf')
# plt.legend()
# plt.show()


# we can replace a specific client so that it is never selected 

# we can influence a  a specific client and change its rank(from a higher --> lower rank)

# we improve the selection probablity  of a specific client's client


# we observer the influence of power of attacker on the attack success rate


'''



