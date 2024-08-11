import numpy as np
import torch
import torch.nn as nn

from FLAlgorithms.utils import *
from FLAlgorithms.augmentation import *

from FLAlgorithms.clients.clientbase import Client


class ClientH2Fed(Client):
    def __init__(self, dataset, id, clients, clients_h, local_data_path, train_data_name_list, train_data_unlabel_name_list,
                 test_data_name_list, window_size, d, label_used, model, l_epochs, bs, classes, lr_model, device, flag,
                 generator, discriminator, lr_gen, lr_dis, z_batch_size, alpha, g_iter, g_epochs):
        super().__init__(dataset, id, clients, clients_h, local_data_path, train_data_name_list, train_data_unlabel_name_list,
                         test_data_name_list, window_size, d, label_used, model, l_epochs, bs, classes, lr_model, device,
                         flag, generator, discriminator, lr_gen, lr_dis, z_batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if flag == True:
            self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.9999))
            self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_dis, betas=(0.5, 0.9999))
        self.rand = RandAugment(2)
        self.alpha = alpha
        self.g_iter = g_iter
        self.gan_f = self.g_iter
        self.g_epochs = g_epochs

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def adjust_LR(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, current_round, client_index):

        ############################################
        ######### training for all clients #########
        ############################################

        train_data, train_label = id2data_2d_array(self.data_path, self.train_data_name_list,
                                                   [self.window_size, self.d], self.label_used)

        train_data_unlabel, _ = id2data_2d_array(self.data_path, self.train_data_unlabel_name_list,
                                                 [self.window_size, self.d], self.label_used)

        train_data_augment = self.rand(train_data_unlabel)

        num = train_data.shape[0]
        num_unlabel = train_data_unlabel.shape[0]

        train_data = torch.from_numpy(train_data).to(self.device, dtype=torch.float32).reshape(num, 1, self.window_size, self.d)
        train_label = torch.from_numpy(train_label).to(self.device, dtype=torch.int64)

        train_data_unlabel = torch.from_numpy(train_data_unlabel).to(self.device, dtype=torch.float32).reshape(num_unlabel, 1, self.window_size, self.d)
        train_data_augment = torch.from_numpy(train_data_augment).to(self.device, dtype=torch.float32).reshape(num_unlabel, 1, self.window_size, self.d)

        n_steps = int(np.ceil(num / self.bs))
        n_steps_unlabel = int(np.ceil(num_unlabel / self.bs))
        if n_steps_unlabel > 1:
            n_steps_unlabel = n_steps_unlabel - 1

        for local_epoch in range(self.local_epochs):
            self.model.train()
            shuffle_index = torch.randperm(num)
            train_data = train_data[shuffle_index, :, :, :]
            train_label = train_label[shuffle_index, :]

            shuffle_index_unlabel = torch.randperm(num_unlabel)
            train_data_unlabel = train_data_unlabel[shuffle_index_unlabel, :, :, :]
            train_data_augment = train_data_augment[shuffle_index_unlabel, :, :, :]

            for n in range(n_steps_unlabel):

                self.optimizer.zero_grad()
                mod_1 = n % n_steps
                start_1 = mod_1 * self.bs
                end_1 = min((mod_1 + 1) * self.bs, num)
                batch_data = train_data[start_1:end_1, :, :, :]
                batch_label = train_label[start_1:end_1, :].reshape(-1, )

                _, prediction = self.model(batch_data)
                cross_e_loss = self.ce_loss(prediction, batch_label)

                mod = n % n_steps_unlabel
                start = mod * self.bs
                end = min((mod + 1) * self.bs, num_unlabel)
                batch_data_unlabel = train_data_unlabel[start:end, :, :, :]
                batch_data_augment = train_data_augment[start:end, :, :, :]
                batch_cat = torch.cat((batch_data_unlabel, batch_data_augment), dim=0)
                num_unsup = batch_data_unlabel.size()[0]
                embeddings, output = self.model(batch_cat)
                embeddings_raw = embeddings[:num_unsup, :]
                embeddings_aug = embeddings[num_unsup:, :]
                loss_unsup = self.mse_loss(embeddings_raw, embeddings_aug)

                loss = cross_e_loss + loss_unsup
                loss.backward()
                self.optimizer.step()

        ################################################################
        ######### training for clients with high computational power ###
        ################################################################
        generator_dp = []
        if self.flag:
            self.generator.train()
            self.discriminator.train()
            self.model.eval()
            train_data_unlabel, _ = id2data_2d_array(self.data_path, self.train_data_unlabel_name_list,
                                                     [self.window_size, self.d], self.label_used)
            num = train_data_unlabel.shape[0]
            train_data_unlabel = torch.from_numpy(train_data_unlabel).to(self.device, dtype=torch.float32).reshape(num, -1)
            n_steps_unlabel = int(np.ceil(num / self.z_batch_size))

            current_gap = self.gan_f
            if n_steps_unlabel > 1:
                iterations = int(np.ceil(self.g_epochs * self.local_epochs * (n_steps_unlabel-1) / current_gap))
            else:
                iterations = int(np.ceil(self.g_epochs * self.local_epochs * n_steps_unlabel / current_gap)) - 1
            iterations_gen = iterations * current_gap

            count, count_dis, D_LOSS, G_LOSS = 0, 0, 0, 0
            for local_epoch in range(self.g_epochs * self.local_epochs):
                shuffle_index = torch.randperm(num)
                train_data_unlabel = train_data_unlabel[shuffle_index, :]
                for n in range(n_steps_unlabel):
                    mod = n % n_steps_unlabel
                    start = mod * self.z_batch_size
                    end = min((mod + 1) * self.z_batch_size, num)
                    batch_data = train_data_unlabel[start:end, :]
                    batch_data = batch_data.to(self.device, dtype=torch.float32)

                    ones = torch.ones((batch_data.size()[0], 1)).to(self.device)
                    zeros = torch.zeros((batch_data.size()[0], 1)).to(self.device)

                    z = torch.randn(batch_data.size()[0], 100).to(self.device)
                    gen_data = self.generator(z)

                    flattened_batch_data = batch_data.reshape(batch_data.size()[0], -1).to(self.device)
                    d_real = self.discriminator(flattened_batch_data)
                    d_fake = self.discriminator(gen_data.detach())
                    d_loss_1 = self.mse_loss(d_real, ones)
                    d_loss_2 = self.mse_loss(d_fake, zeros)
                    d_loss = (d_loss_1 + d_loss_2) * 0.5

                    if count_dis < iterations and count % self.g_iter == 0:
                        self.optimizer_dis.zero_grad()
                        d_loss.backward()
                        self.optimizer_dis.step()
                        # print('d_loss: {:4f}'.format(d_loss.item()))
                        count_dis += 1
                        D_LOSS = d_loss.item()

                    d_fake_1 = self.discriminator(gen_data)
                    reshaped_gen_data = gen_data.reshape(gen_data.size()[0], 1, self.window_size, self.d)
                    _, s_prob = self.model(reshaped_gen_data)
                    prob = F.softmax(s_prob, dim=1).mean(dim=0)
                    loss_information_entropy = (prob * torch.log10(prob)).sum()
                    task_loss = self.mse_loss(d_fake_1, ones)
                    gen_loss = task_loss + self.alpha * loss_information_entropy

                    if count < iterations_gen:
                        self.optimizer_gen.zero_grad()
                        gen_loss.backward()
                        self.optimizer_gen.step()
                        # print('g_mse_loss: {:4f}'.format(task_loss.item()))
                        count += 1
                        G_LOSS = task_loss.item()

            if G_LOSS >= 1.5 * D_LOSS:
                self.gan_f += 5
            else:
                self.gan_f = self.g_iter

            generator_dp = dp(self.generator)

        lr = 0.001
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']

        # evaluate
        acc, f1, kappa = self.test()
        print('CURRENT ROUND: {}, CLIENT IINDEX: {}, ACC: {}, F1:{}, KAPPA:{}'.format(current_round, client_index, acc, f1, kappa))

        return self.model, lr, generator_dp

