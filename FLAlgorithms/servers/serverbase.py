import torch
import numpy as np

import copy


class Server:
    def __init__(self, dataset, subject_split_file_path, train_subject_path, rounds, clients, clients_h, d, window_size,
                 bs, l_epochs, classes, z_batch_size, models, lr_model, generators, discriminators, generator_global,
                 lr_gen, lr_dis, device):

        self.dataset = dataset
        self.rounds = rounds
        self.num_clients = clients
        self.num_clients_h = clients_h
        self.window_size = window_size
        self.d = d
        self.classes = classes
        self.local_epochs = l_epochs
        self.bs = bs
        self.generator = generator_global
        self.z_batch_size = z_batch_size
        self.device = device
        self.clients = []
        self.local_models = []
        self.local_models = [copy.deepcopy(models[i]).to(device) for i in range(clients)]
        self.model = copy.deepcopy(models[0]).to(device)

    def send_parameters2personalized(self):
        for i, client in enumerate(self.clients):
            client.set_parameters2personal(self.local_models[i])

    def add_parameters(self, client, ratio):
        for server_param, user_param in zip(self.model.parameters(), client.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.clients is not None and len(self.clients) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for client in self.clients:
            total_train += client.train_samples

        num = len(self.clients)
        with torch.no_grad():
            for keys in self.model.state_dict().keys():
                temp = torch.zeros_like(self.model.state_dict()[keys]).to(self.device)
                for user in self.clients:
                    temp += (1 / num) * user.model.state_dict()[keys]
                self.model.state_dict()[keys].data.copy_(temp)

    def select_clients(self, clients_h, ratio):
        if ratio == 1.0:
            print("All users are selected")
            return self.clients
        selected_clients = np.random.choice(self.clients[clients_h:], size=int(len(clients_h[4:]) * ratio), replace=False)
        return self.clients[:clients_h] + selected_clients

    def test(self):
        '''tests self.latest_model on given clients
        '''
        accs, f1s, kappas = [], [], []
        for c in self.clients:
            acc, f1, kappa = c.test()
            accs.append(acc)
            f1s.append(f1)
            kappas.append(kappa)
        acc_tensor = torch.tensor(accs)
        f1_tensor = torch.tensor(f1s)
        kappa_tensor = torch.tensor(kappas)
        mean_acc = torch.mean(acc_tensor)
        mean_f1 = torch.mean(f1_tensor)
        mean_kappa = torch.mean(kappa_tensor)
        return mean_acc, mean_f1, mean_kappa

