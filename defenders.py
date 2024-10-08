import torch
import numpy as np
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hdbscan
import imageio

from sklearn.decomposition import PCA
from collections import Counter
import time



#  "average" "krum"   "trimmedmean"  "median" "clustering" "bulyankrum"  "Fltrust" 

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()



# def vectorize_net(model):
#     """
#     Convert the model's parameters to a single vector.
#     """
#     return torch.nn.utils.parameters_to_vector(model.parameters())

# def load_model_weight(model, vec):
#     """
#     Load the vectorized parameters (vec) into the model.
#     """
#     torch.nn.utils.vector_to_parameters(vec, model.parameters())


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight_diff[index_bias:index_bias + p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorized_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorized_diff).item()
        clipped_weight_diff = vectorized_diff / max(1, weight_diff_norm / self.norm_bound)

        print("The Norm of Weight Difference between received global model and updated client model: {}".format(weight_diff_norm))
        print("The Norm of weight (updated part) after clipping: {}".format(torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None

class RSA(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, global_model, flround, *args, **kwargs):

        for net_index, net in enumerate(client_model):
            whole_aggregator = []
            for p_index, p in enumerate(client_model[0].parameters()):
                params_aggregator = 0.00005 * 0.998 ** flround * torch.sign(list(net.parameters())[p_index].data
                    - list(global_model.parameters())[p_index].data) + list(global_model.parameters())[p_index].data

                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(net.parameters()):
                p.data = whole_aggregator[param_index]

        return None


class Median(Defense):
    """
    Implementing the Median aggregation method.
    """
    
    def __init__(self, *args, **kwargs):
        pass

    # def exec(self, client_models, global_model_pre, device, *args, **kwargs):
    def exec(self, client_models, global_model_pre, device, *args, **kwargs):

        # First, we need to compute the parameter updates from each client
        updates = []
        
        for model in client_models:
            model_update = []
            for param_client, param_global in zip(model.parameters(), global_model_pre.parameters()):
                model_update.append(param_client.data - param_global.data)
            updates.append(model_update)
        
        # Compute the median of the updates
        median_update = []
        for i in range(len(updates[0])):
            stacked_params = torch.stack([update[i] for update in updates], dim=0)
            median_param = torch.median(stacked_params, dim=0).values
            median_update.append(median_param)
        
        # Apply the median update to the global model to get the aggregated model
        aggregated_model = copy.deepcopy(global_model_pre).to(device)
        for param, update in zip(aggregated_model.parameters(), median_update):
            param.data += update
        
        # Return the aggregated model
        return [aggregated_model]





class Bulyan(Defense):
    """
    Implementation of the Bulyan defense mechanism.
    """

    def __init__(self, num_workers, num_adv, k, *args, **kwargs):
        self.num_workers = num_workers
        self.s = num_adv
        self.k = k  # Number of models to exclude in Bulyan

    def exec(self, client_models, global_model_pre, device, *args, **kwargs):
        
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        # Compute distances for the Krum score calculation
        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j) ** 2))
            neighbor_distances.append(distance)

        # Compute Krum scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        
        # Sort based on the scores to get top m models after removing the worst k
        sorted_indices = np.argsort(scores)
        top_m_indices = sorted_indices[self.k:self.k+self.num_workers-2*self.s]

        aggregated_updates = [vectorize_net(client_models[i]).detach().cpu().numpy() for i in top_m_indices]

        # Compute Bulyan aggregation (median of top m selected models)
        # aggregated_model_vector = np.median(aggregated_updates, axis=0)

        # # Convert aggregated model vector back to PyTorch model
        # aggregated_model = copy.deepcopy(client_models[0])
        # load_model_weight(aggregated_model, torch.from_numpy(aggregated_model_vector.astype(np.float32)).to(device))
        
        # return [aggregated_model]


        # Compute Bulyan aggregation (median of top m selected models)
        aggregated_model_vector = np.median(aggregated_updates, axis=0)

        # Convert aggregated model vector back to PyTorch model
        aggregated_model = copy.deepcopy(client_models[0])
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_model_vector.astype(np.float32)).to(device))

        return [aggregated_model]




class FedAvg(Defense):
    """
    Implementation of the FedAvg (Federated Averaging) aggregation mechanism.
    """

    def __init__(self, *args, **kwargs):
        # Initialization can be extended if necessary
        pass

    def exec(self, client_models, global_model_pre, device, *args, **kwargs):
        
        # Vectorize client models to numpy arrays
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        # Compute the FedAvg aggregation (mean of all client models)
        aggregated_model_vector = np.mean(vectorize_nets, axis=0)

        # Convert aggregated model vector back to PyTorch model
        aggregated_model = copy.deepcopy(client_models[0])
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_model_vector.astype(np.float32)).to(device))
        
        return [aggregated_model]





# aggregated_model_vector = np.array([np.median(aggregated_updates, axis=0)])




class TrimmedMean(Defense):
    def __init__(self, k, *args, **kwargs):
        """
        k: Number of highest and lowest updates to trim.
        """
        self.k = k

    def exec(self, client_models, global_model, device, *args, **kwargs):
        # Collect all updates from client models as vectors
        all_updates = [vectorize_net(client) - vectorize_net(global_model) for client in client_models]

        # Sort updates
        sorted_updates = sorted(all_updates, key=lambda x: torch.norm(x))

        # Trim the k lowest and k highest updates
        trimmed_updates = sorted_updates[self.k:-self.k]

        # Compute the average of the trimmed updates
        avg_update = torch.stack(trimmed_updates).mean(dim=0)

        # Apply the average update to the global model to obtain the aggregated model
        aggregated_model = copy.deepcopy(global_model)
        load_model_weight(aggregated_model, vectorize_net(global_model) + avg_update.to(device))

        return [aggregated_model]










class FLTrust(Defense):
    """
    Implementing the FLTrust defense.
    """

    def __init__(self, threshold=0.5, *args, **kwargs):
        """
        :param threshold: Cosine similarity threshold. Updates with a similarity below this are discarded.
        """
        self.threshold = threshold

    def exec(self, client_models, global_model_pre, device, *args, **kwargs):
        
        # Convert the global_model_pre to a vector
        global_model_vector = vectorize_net(global_model_pre).detach().cpu().numpy()

        valid_models = []

        for cm in client_models:
            client_vector = vectorize_net(cm).detach().cpu().numpy()

            # Compute the cosine similarity between the client update and the global model
            cosine_similarity = F.cosine_similarity(
                torch.tensor(client_vector).to(device),
                torch.tensor(global_model_vector).to(device),
                dim=0
            )

            # Check against the threshold
            if cosine_similarity > self.threshold:
                valid_models.append(cm)

        # If no models are valid, just return the global model
        if not valid_models:
            return [global_model_pre]

        # Aggregate the updates of valid models by taking the mean for each parameter across valid models.
        aggregated_updates = []
        for i, _ in enumerate(valid_models[0].parameters()):
            stacked_param = torch.stack([list(model.parameters())[i].data for model in valid_models])
            mean_param = torch.mean(stacked_param, dim=0)
            aggregated_updates.append(mean_param)

        aggregated_model = copy.deepcopy(global_model_pre)
        for param, update in zip(aggregated_model.parameters(), aggregated_updates):
            param.data = update.to(device)

        return [aggregated_model]










class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """

    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, global_model_pre, num_dps, g_user_indices, device, *args, **kwargs):

        ######################################################################## separate model to get updated part
        whole_aggregator = []
        client_models_copy = copy.deepcopy(client_models)
        for i in range(len(client_models_copy)):
            for p_index, p in enumerate(client_models_copy[i].parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                params_aggregator = params_aggregator + (list(client_models_copy[i].parameters())[p_index].data -
                                                         list(global_model_pre.parameters())[p_index].data)
                # params_aggregator = torch.sign(params_aggregator)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(client_models_copy[i].parameters()):
                p.data = whole_aggregator[param_index]

            whole_aggregator = []

        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j) ** 2))
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            print("===starr===:", i_star)
            print("===scoree===:", scores)
            print("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)),
                                                                                           g_user_indices[scores.index(
                                                                                               min(scores))]))
            aggregated_model = client_models[i]
            neo_net_list = [aggregated_model]
            print("Norm of Aggregated Model: {}".format(
                torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, i_star
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score + 2)[:nb_in_score + 2]

            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]

            print("===scores===", scores)
            print("Num data points: {}".format(num_dps))
            print("Num selected data points: {}".format(selected_num_dps))
            print("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
                                                                    [g_user_indices[ti] for ti in topk_ind]))

            aggregated_model=[]
            for i in range(len(topk_ind)):
                aggregated_model.append(client_models[topk_ind[i]])

            neo_net_list = aggregated_model
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, topk_ind

class XMAM(Defense):

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, x_ray_loader, global_model_pre, g_user_indices, device, malicious_ratio, untargeted_type, *args, **kwargs):

        # for data, target in x_ray_loader[1]:
        if malicious_ratio == 0:
            for data, target in x_ray_loader:
                x_ray = data[0:1]
                break
        else:
            if untargeted_type == 'none':
                for data, target in x_ray_loader[1]:
                    x_ray = data[0:1]
                    break
            else:
                for data, target in x_ray_loader:
                    x_ray = data[0:1]
                    break


        x_ray = x_ray.to(device)
        x_ray = torch.ones_like(x_ray)

        client_num = len(client_models)

        ######################################################################## separate model to get updated part
        whole_aggregator = []
        client_models_copy = copy.deepcopy(client_models)
        for i in range(len(client_models_copy)):
            for p_index, p in enumerate(client_models_copy[i].parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                params_aggregator = params_aggregator + 1*(list(client_models_copy[i].parameters())[p_index].data -
                                                         list(global_model_pre.parameters())[p_index].data)
                # params_aggregator = torch.sign(params_aggregator)
                whole_aggregator.append(params_aggregator)
                # print("{}--{}:{}".format(i, p_index, p))

            for param_index, p in enumerate(client_models_copy[i].parameters()):
                p.data = whole_aggregator[param_index]

            whole_aggregator = []

        client_SLPDs = []
        Temperature = 1

        for net_index, net in enumerate(client_models_copy):
            SLPD_now = net(x_ray)
            SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
            SLPD_now = SLPD_now.detach().cpu().numpy()
            SLPD_now = SLPD_now[0]
            client_SLPDs.append(SLPD_now)

        client_SLPDs = np.array(client_SLPDs)


        ################################################## the first screening: for abnormal SLPD value like nan etc.
        client_models_nonan = []
        client_SLPDs_nonan = []
        jjj = 0
        for i in range(client_num):
            for j in range(len(client_SLPDs[i])):
                jjj = j
                if np.isnan(client_SLPDs[i][j]):
                    print("********delete client {}'s model for nan********".format(i))
                    break

            if jjj == len(client_SLPDs[i])-1:
                client_models_nonan.append(client_models[i])
                client_SLPDs_nonan.append(client_SLPDs[i])

        client_num_remain = len(client_models_nonan)


        ######################################################################### the second screening: cluster SLPDs
        pca = PCA(n_components=3)
        X_new = pca.fit_transform(client_SLPDs_nonan)
        # X_new = pca.fit_transform(net_vec)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(X_new)
        majority = Counter(cluster_labels)
        majority = majority.most_common()[0][0]

        client_models_remain = []
        g_user_indices_remain = []
        for i in range(client_num_remain):
            if cluster_labels[i] == majority:
                client_models_remain.append(client_models_nonan[i])
                g_user_indices_remain.append(i)
        print("======models selected=====:", g_user_indices_remain)
        return client_models_remain, g_user_indices_remain

class RFA(Defense):
    """
    we implement the robust aggregator at:
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation:
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, maxiter=4, eps=1e-5, ftol=1e-6, device=torch.device("cuda"), *args, **kwargs):
        """
        Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        net_freq = [0.1 for i in range(len(client_model))]
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_model]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        print('Starting Weiszfeld algorithm')
        print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            print("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        # print("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_model[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        # tot_weights = np.sum(weights)
        # weighted_updates = [np.zeros_like(v) for v in points[0]]
        # for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        # return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])





