import copy
import yaml
import numpy as np
import os
import torch
import torch.distributions as tdist
import torch.nn.functional as F


def load_yaml_to_dict(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def dp(model, mean=0.0, std=0.01, device='cuda'):

    dp_model = copy.deepcopy(model)

    for key in dp_model.state_dict().keys():
        if dp_model.state_dict()[key].dtype == torch.int64:
            continue
        else:
            value = torch.std(dp_model.state_dict()[key]).detach().cpu()
            if value.item() == 'nan':
                temp = dp_model.state_dict()[key]
                dp_model.state_dict()[key].data.copy_(temp)
            else:
                nn = tdist.Normal(torch.tensor([mean]), std * value)
                noise = nn.sample(dp_model.state_dict()[key].size()).squeeze()
                noise = noise.to(device)
                temp = dp_model.state_dict()[key] + noise
                dp_model.state_dict()[key].data.copy_(temp)
    return dp_model


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_k1 = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_k1


def norm(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data-mean)/std


def norm_tensor(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)
    normalized_data = (data - mean) / std
    normalized_data = normalized_data.reshape(-1, data.size()[1]*data.size()[2])
    return normalized_data, mean, std


# process data
def id2data_2d_array(abs_path, data_list, shape, used_label, normalize=True):
    data = np.zeros((0, shape[0], shape[1]))
    label = np.zeros((0, 1))

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        original_array = np.expand_dims(original_array, axis=0)
        data = np.concatenate((data, original_array), axis=0)
        ###################################################
        # -- map raw label into the consistent labels
        ###################################################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = used_label.index(raw_label)
        mapped_label = np.array([[mapped_label]], dtype=int)
        label = np.concatenate((label, mapped_label), axis=0)
        count = count + 1

    if normalize == True:
        data = norm(data)
    data = data.reshape(-1, shape[0], shape[1])
    return data, label


def id2data(abs_path, data_list, shape, used_label, normalize=True):

    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = used_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    data = data.reshape(-1, shape[0], shape[1])
    if normalize == True:
        data, _, _ = norm_tensor(data)
    return data, label