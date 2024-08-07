from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa
from FLAlgorithms.utils import *


class Client:

    def __init__(self, dataset, id, clients, clients_h, local_data_path, train_data_name_list, train_data_unlabel_name_list,
                 test_data_name_list, window_size, d, label_used, model, l_epochs, bs, classes, lr_model, device,
                 flag, generator, discriminator, lr_gen, lr_dis, z_batch_size):

        self.dataset = dataset
        self.id = id
        self.num_all_clients = clients
        self.num_clients_h = clients_h
        self.data_path = local_data_path
        self.train_data_name_list = train_data_name_list
        self.train_samples = len(self.train_data_name_list)
        self.train_data_unlabel_name_list = train_data_unlabel_name_list
        self.test_data_name_list = test_data_name_list

        self.window_size = window_size
        self.d = d
        self.label_used = label_used

        self.model = model
        self.lr = lr_model
        self.local_epochs = l_epochs
        self.bs = bs
        self.classes = classes

        self.device = device

        self.f1_score = MulticlassF1Score(num_classes=self.classes).to(device)
        self.cohen_kappa = MulticlassCohenKappa(num_classes=self.classes).to(device)

        self.flag = flag
        self.generator = generator
        self.discriminator = discriminator
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.z_batch_size = z_batch_size

    def set_parameters2personal(self, model):
        with torch.no_grad():
            for keys in model.state_dict().keys():
                self.model.state_dict()[keys].data.copy_(model.state_dict()[keys])

    def get_parameters(self):  # of personal model
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):  # clone 'param' to 'clone_param'
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_data, test_label = id2data(self.data_path, self.test_data_name_list, [self.window_size, self.d], self.label_used)
            num = len(self.test_data_name_list)
            test_data = test_data.reshape(num, 1, self.window_size, self.d)
            test_data = test_data.to(self.device)

            test_label = test_label.to(self.device).reshape(-1, )
            _, output = self.model(test_data)
            pred = output.data.max(1)[1]
            correct = pred.eq(test_label).sum().item()
            acc = correct / num
            f1 = self.f1_score(output, test_label)
            kappa = self.cohen_kappa(output, test_label)
        self.model.train()
        return acc, f1, kappa

    def save_model(self):
        model_path = os.path.join("models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "client_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models")
        self.model = torch.load(os.path.join(model_path, "client_" + self.id + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
