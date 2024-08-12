from tqdm import tqdm
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.clients.clientH2Fed import ClientH2Fed
from FLAlgorithms.utils import *


class H2Fed(Server):
    def __init__(self, dataset, subject_split_file_path, train_subject_path, rounds, clients, clients_h, d, window_size,
                 bs, l_epochs, classes, z_batch_size, models, lr_reduced, lr_model, generators, discriminators,
                 generator_global, lr_gen, lr_dis, device, ratio, g_iter, g_epochs, alpha, N, ema, multiplier):
        super().__init__(dataset, subject_split_file_path, train_subject_path, rounds, clients, clients_h, d,
                         window_size, bs, l_epochs, classes, z_batch_size,
                         models, lr_model, generators, discriminators, generator_global, lr_gen, lr_dis, device)

        self.ratio = ratio
        self.N = N
        self.using_EMA = ema
        self.multiplier = multiplier
        self.lr_reduced = lr_reduced
        self.kd_loss = kdloss
        self.Correlation = torch.ones((self.num_clients, self.num_clients))
        train_subject_path = train_subject_path
        subject_split_list = load_yaml_to_dict(subject_split_file_path)
        train_subject_list = subject_split_list['train']

        train_label_list = list()
        train_unlabel_list = list()
        test_list = list()

        data_path = []

        # Each list consists of sub-lists for each client. Each sublist stores the filenames of the data
        for i in range(clients):
            id = train_subject_list[i]

            abs_path = os.path.join(train_subject_path, id)
            data_path.append(abs_path)

            train_data_list_name = 'sample_id_train.txt'
            train_data_list_path = os.path.join(train_subject_path, id, train_data_list_name)
            with open(train_data_list_path, 'r') as f:
                train_label_list_l = f.readlines()
            f.close()
            train_label_list.append(train_label_list_l)

            train_data_unlabel_list_name = 'sample_id_train_unlabel.txt'
            train_data_unlabel_list_path = os.path.join(train_subject_path, id, train_data_unlabel_list_name)
            with open(train_data_unlabel_list_path, 'r') as f:
                train_unlabel_list_l = f.readlines()
            f.close()
            train_unlabel_list.append(train_unlabel_list_l)

            test_data_list_name = 'sample_id_test.txt'
            test_data_list_path = os.path.join(train_subject_path, id, test_data_list_name)
            with open(test_data_list_path, 'r') as f:
                test_list_l = f.readlines()
            f.close()
            test_list.append(test_list_l)

        label_used = []
        if dataset == "PAMAP2":
            label_used = [0, 1, 2, 3, 10, 11, 12, 13]
        elif dataset == "UCI-HAR":
            label_used = [0, 1, 2, 3, 4, 5]
        elif dataset == "USC-HAD":
            label_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif dataset == "WISDM":
            label_used = [0, 1, 2, 3, 4]
        elif dataset == "EHR":
            label_used = [0, 1, 2]
        elif dataset == "HARBox":
            label_used = [0, 1, 2, 3, 4]

        print('clients initializing...')
        for i in tqdm(range(clients), total=clients):
            id = 'f_{0:05d}'.format(i)
            local_data_path = data_path[i]
            train_data_name_list = train_label_list[i]
            train_data_unlabel_name_list = train_unlabel_list[i]
            test_data_name_list = test_list[i]
            if i < clients_h:
                flag = True
                client = ClientH2Fed(dataset, id, clients, clients_h, local_data_path, train_data_name_list,
                                     train_data_unlabel_name_list, test_data_name_list, window_size, d, label_used,
                                     models[i], l_epochs, bs, classes, lr_model, device, flag, generators[i],
                                     discriminators[i], lr_gen, lr_dis, z_batch_size, alpha, g_iter, g_epochs)

            else:
                flag = False
                client = ClientH2Fed(dataset, id, clients, clients_h, local_data_path, train_data_name_list,
                                     train_data_unlabel_name_list, test_data_name_list, window_size, d, label_used,
                                     models[i], l_epochs, bs, classes, lr_model, device, flag, [], [],
                                     lr_gen=0.0, lr_dis=0.0, z_batch_size=0, alpha=0.0, g_iter=0, g_epochs=0)

            self.clients.append(client)
        print("Finished creating H2Fed server.")

    def train(self):
        ACC, F1, KAPPA = [], [], []
        for round in range(self.rounds):
            print("-------------Round number: ", round, " -------------")
            if self.lr_reduced == True:
                if round > 0 and round % 5 == 0:
                    for client in self.clients:
                        client.adjust_LR()
            if round == 0:
                stats = self.evaluate()
                ACC.append(stats[0])
                F1.append(stats[1])
                KAPPA.append(stats[2])

            self.send_parameters2personalized()

            local_optimizers, generators = [], []
            for i, client in enumerate(self.clients):
                model, lr, generator = client.train(round, i)
                self.local_models[i] = copy.deepcopy(model).to(self.device)
                optimizer = torch.optim.Adam(self.local_models[i].parameters(), lr=lr)
                local_optimizers.append(optimizer)
                if i < self.num_clients_h:
                    generators.append(generator)

            stats = self.evaluate()
            ACC.append(stats[0])
            F1.append(stats[1])
            KAPPA.append(stats[2])

            GEN_data = torch.zeros((0, self.window_size * self.d)).to(self.device)
            GEN_labels = list()
            [GEN_labels.append(torch.zeros((0, self.classes)).to(self.device)) for _ in range(self.num_clients)]
            similarity = torch.zeros((self.num_clients, self.num_clients)).to(self.device)
            for client in range(self.num_clients_h):
                for _ in range(self.N):
                    z = torch.randn(self.z_batch_size, 100).to(self.device)
                    self.generator.load_state_dict(generators[client].state_dict())
                    gen_data = self.generator(z)
                    GEN_data = torch.cat((GEN_data, gen_data.detach()), dim=0)
                    current_labels = list()
                    [current_labels.append(torch.zeros((self.z_batch_size, self.classes)).to(self.device)) for _ in range(self.num_clients)]
                    with torch.no_grad():
                        for client_ in range(self.num_clients):
                            reshaped_gen_data = gen_data.reshape(gen_data.size()[0], 1, self.window_size, self.d)
                            _, local_pred = self.local_models[client_](reshaped_gen_data)
                            GEN_labels[client_] = torch.cat((GEN_labels[client_], local_pred.detach()), dim=0)
                            current_labels[client_] = local_pred
                    for client_1 in range(self.num_clients):
                        for client_2 in range(self.num_clients):
                            a = current_labels[client_1]
                            b = current_labels[client_2]
                            sim = self.kd_loss(a, b)
                            similarity[client_1:(client_1 + 1), client_2:(client_2 + 1)] += sim

            mean = torch.mean(similarity, dim=1, keepdim=True)
            indices = torch.nonzero(similarity < mean)
            correlation = torch.zeros((self.num_clients, self.num_clients))
            correlation[indices[:, 0], indices[:, 1]] = 1
            start = int(0.1*self.rounds)
            if round >= start:
                if self.using_EMA == True:
                    self.Correlation = (1 - self.multiplier)*self.Correlation + self.multiplier*correlation
                else:
                    self.Correlation = correlation

            num = GEN_data.size()[0]
            n_steps = int(np.ceil(num / self.z_batch_size))

            if round < (self.rounds-1):
                for client in range(self.num_clients):
                    self.local_models[client].train()
                    epochs = int(np.ceil(0.25*self.local_epochs))
                    for local_epoch in range(epochs):
                        shuffle_index = torch.randperm(num)
                        GEN_data = GEN_data[shuffle_index, :]
                        for client_1 in range(self.num_clients):
                            GEN_labels[client_1] = GEN_labels[client_1][shuffle_index, :]

                        for n in range(n_steps):
                            mod = n % n_steps
                            start = mod * self.bs
                            end = min((mod + 1) * self.bs, num)
                            batch_data = GEN_data[start:end, :]
                            batch_data = batch_data.reshape(batch_data.size()[0], 1, self.window_size, self.d)

                            batch_labels = list()
                            [batch_labels.append(GEN_labels[client_1][start:end, :].to(self.device)) for client_1 in range(self.num_clients)]

                            local_optimizers[client].zero_grad()
                            _, prediction = self.local_models[client](batch_data)
                            kl_loss = torch.zeros((1,)).to(self.device)
                            for s_clients in range(self.num_clients):
                                sum = torch.sum(self.Correlation[client, :]).item()
                                weight = self.Correlation[client, s_clients].item() / sum
                                kl_loss += weight * self.kd_loss(prediction, batch_labels[s_clients])

                            kl_loss.backward()
                            local_optimizers[client].step()

        self.save_metric_info(ACC, F1, KAPPA)

    def evaluate(self):
        stats = self.test()  # acc, f1, kappa
        print('ACC: {}, F1:{}, KAPPA:{}'.format(stats[0], stats[1], stats[2]))
        return stats

    def save_metric_info(self, Accs, F1s, Kappas):
        root_path = 'logs'
        file_name = '{}_average_info.txt'.format(self.dataset)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        path = os.path.join(root_path, file_name)

        if os.path.exists(path):
            os.remove(path)

        communication_rounds = len(Accs)
        ffg = open(path, 'a')
        for i in range(communication_rounds):
            print('CURRENT ROUND: {}, ACC: {}, F1:{}, KAPPA:{}'.format(i, Accs[i].item(), F1s[i].item(), Kappas[i].item()), file=ffg)






