#!/usr/bin/env python
import argparse
import copy
from FLAlgorithms.servers.serverH2Fed import H2Fed
from FLAlgorithms.models.model_PAMAP2 import (ResNet as pamap2_ResNet, CNN as pamap2_CNN,
                                              CNN_tiny as pamap2_CNN_tiny, ActivityLSTM as pamap2_lstm)
from FLAlgorithms.models.generator_PAMPA2 import Generator as pamap2_generator, Discriminator as pamap2_discriminator
from FLAlgorithms.models.model_UCIHAR import (ResNet as uci_ResNet, ResNet_tiny as uci_ResNet_tiny,
                                              CNN as uci_CNN, CNN_tiny as uci_CNN_tiny, ActivityLSTM as uci_lstm)
from FLAlgorithms.models.generator_UCIHAR import Generator as uci_generator, Discriminator as uci_discriminator
from FLAlgorithms.models.model_WISDM import (ResNet as wisdm_ResNet, ResNet_reduce_filters as wisdm_ResNet_rf,
                                             ResNet_reduced_layers as wisdm_ResNet_rl, CNN as wisdm_CNN,
                                             CNN_reduce_filters as wisdm_CNN_rf,
                                             CNN_reduce_layers as wisdm_CNN_rl, ActivityGRU as wisdm_gru)
from FLAlgorithms.models.generator_WISDM import Generator as wisdm_generator, Discriminator as wisdm_discriminator
from FLAlgorithms.models.model_USCHAD import (ResNet as uschad_ResNet, CNN as uschad_CNN,
                                              CNN_tiny as uschad_CNN_tiny, ActivityGRU as uschad_gru)
from FLAlgorithms.models.generator_USCHAD import Generator as uschad_generator, Discriminator as uschad_discriminator
from FLAlgorithms.models.model_EHR import *
from FLAlgorithms.models.generator_EHR import Generator as ehr_generator, Discriminator as ehr_discriminator
from FLAlgorithms.models.model_HARBox import *
from FLAlgorithms.models.generator_HARBox import Generator as box_generator, Discriminator as box_discriminator


def main(dataset, subject_split_file_path, train_subject_path, rounds, clients, clients_h, d, window_size, bs, l_epochs,
         classes, ratio, z_latent_dim, z_batch_size, fc_units, g_iter, g_epochs, alpha, N, ema, multiplier, LR_reduced,
         lr_model, lr_gen, lr_dis, device):
    model_path = []
    if dataset == "PAMAP2":
        models = list()
        model_1 = pamap2_ResNet(input_channel=1, num_classes=classes).to(device)
        model_2 = pamap2_CNN(input_channel=1, num_classes=classes).to(device)
        model_3 = pamap2_CNN_tiny(input_channel=1, num_classes=classes).to(device)
        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients_h)]
        [models.append(copy.deepcopy(model_2).to(device)) for _ in range(3)]
        [models.append(copy.deepcopy(model_3).to(device)) for _ in range(clients-clients_h - 3)]
        # define the gans for clients with high computational power
        generator_base = pamap2_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = pamap2_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]

        trainable_params1 = sum(p.numel() for p in model_1.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params2 = sum(p.numel() for p in model_2.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params3 = sum(p.numel() for p in model_3.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params4 = sum(p.numel() for p in generator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params5 = sum(p.numel() for p in discriminator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        print("MODEL PARAMS: {}, {}, {}, {}, {}".format(
            trainable_params1, trainable_params2, trainable_params3, trainable_params4, trainable_params5))

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    elif dataset == "UCI-HAR":
        models = list()
        model_1 = uci_ResNet(input_channel=1, num_classes=classes).to(device)
        model_2 = uci_ResNet_tiny(input_channel=1, num_classes=classes).to(device)
        model_3 = uci_CNN(input_channel=1, num_classes=classes).to(device)
        model_4 = uci_CNN_tiny(input_channel=1, num_classes=classes).to(device)

        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients_h)]
        [models.append(copy.deepcopy(model_2).to(device)) for _ in range(10)]
        [models.append(copy.deepcopy(model_3).to(device)) for _ in range(9)]
        [models.append(copy.deepcopy(model_4).to(device)) for _ in range(clients - clients_h - 19)]

        generator_base = uci_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = uci_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]
        trainable_params1 = sum(p.numel() for p in model_1.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params2 = sum(p.numel() for p in model_2.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params3 = sum(p.numel() for p in model_3.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params4 = sum(p.numel() for p in model_4.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params5 = sum(p.numel() for p in generator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params6 = sum(p.numel() for p in discriminator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        print("MODEL PARAMS: {}, {}, {}, {}, {}, {}".format(
            trainable_params1, trainable_params2, trainable_params3, trainable_params4, trainable_params5,
            trainable_params6))

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    elif dataset == "WISDM":
        models = list()
        model_1 = wisdm_ResNet(input_channel=1, num_classes=classes).to(device)
        model_2 = wisdm_ResNet_rf(input_channel=1, num_classes=classes).to(device)
        model_3 = wisdm_CNN(input_channel=1, num_classes=classes).to(device)
        model_4 = wisdm_ResNet_rl(input_channel=1, num_classes=classes).to(device)
        model_5 = wisdm_CNN_rl(input_channel=1, num_classes=classes).to(device)
        model_6 = wisdm_CNN_rf(input_channel=1, num_classes=classes).to(device)

        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients_h)]
        [models.append(copy.deepcopy(model_2).to(device)) for _ in range(8)]
        [models.append(copy.deepcopy(model_3).to(device)) for _ in range(9)]
        [models.append(copy.deepcopy(model_4).to(device)) for _ in range(9)]
        [models.append(copy.deepcopy(model_5).to(device)) for _ in range(8)]
        [models.append(copy.deepcopy(model_6).to(device)) for _ in range(clients - clients_h - 34)]

        generator_base = wisdm_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = wisdm_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]
        trainable_params1 = sum(p.numel() for p in model_1.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params2 = sum(p.numel() for p in model_2.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params3 = sum(p.numel() for p in model_3.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params4 = sum(p.numel() for p in model_4.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params5 = sum(p.numel() for p in model_5.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params6 = sum(p.numel() for p in model_6.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params7 = sum(p.numel() for p in generator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params8 = sum(p.numel() for p in discriminator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        print("MODEL PARAMS: {}, {}, {}, {}, {}, {}, {}, {}".format(
            trainable_params1, trainable_params2, trainable_params3, trainable_params4, trainable_params5,
            trainable_params6, trainable_params7, trainable_params8))

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    elif dataset == "USC-HAD":
        # define models
        models = list()
        model_1 = uschad_ResNet(input_channel=1, num_classes=classes).to(device)
        model_2 = uschad_CNN(input_channel=1, num_classes=classes).to(device)
        model_3 = uschad_CNN_tiny(input_channel=1, num_classes=classes).to(device)
        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients_h)]
        [models.append(copy.deepcopy(model_2).to(device)) for _ in range(6)]
        [models.append(copy.deepcopy(model_3).to(device)) for _ in range(clients-clients_h - 6)]
        # define the gans for clients with high computational power
        generator_base = uschad_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = uschad_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]

        trainable_params1 = sum(p.numel() for p in model_1.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params2 = sum(p.numel() for p in model_2.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params3 = sum(p.numel() for p in model_3.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params4 = sum(p.numel() for p in generator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        trainable_params5 = sum(p.numel() for p in discriminator_base.parameters(recurse=True) if p.requires_grad) / 1000000
        print("MODEL PARAMS: {}, {}, {}, {}, {}".format(
            trainable_params1, trainable_params2, trainable_params3, trainable_params4, trainable_params5))

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    elif dataset == "EHR":
        models = list()
        model_1 = EHR_FCN(dim=classes).to(device)
        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients)]
        generator_base = ehr_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = ehr_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    elif dataset == "HARBox":
        models = list()
        model_1 = HARBox_1DCNN().to(device)
        model_2 = HARBox_1DCNNtiny().to(device)
        [models.append(copy.deepcopy(model_1).to(device)) for _ in range(clients_h)]
        [models.append(copy.deepcopy(model_2).to(device)) for _ in range(clients - clients_h)]

        generator_base = box_generator(z_size=z_latent_dim, input_feat=window_size * d, fc_units=fc_units).to(device)
        generators = [copy.deepcopy(generator_base).to(device) for _ in range(clients_h)]
        generator_global = copy.deepcopy(generator_base).to(device)
        discriminator_base = box_discriminator(hidden_dim=window_size * d, output_dim=1).to(device)
        discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(clients_h)]

        server = H2Fed(dataset=dataset, subject_split_file_path=subject_split_file_path,
                       train_subject_path=train_subject_path,  rounds=rounds, clients=clients, clients_h=clients_h, d=d,
                       window_size=window_size, bs=bs, l_epochs=l_epochs, classes=classes, z_batch_size=z_batch_size,
                       models=models, lr_reduced=LR_reduced, lr_model=lr_model,  generators=generators,
                       discriminators=discriminators, generator_global=generator_global, lr_gen=lr_gen, lr_dis=lr_dis,
                       device=device, ratio=ratio, g_iter=g_iter, g_epochs=g_epochs, alpha=alpha, N=N, ema=ema,
                       multiplier=multiplier)
        server.train()

    else:
        raise NotImplementedError()

    return model_path


def run():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="PAMAP2",
                        choices=["PAMAP2", "UCI-HAR", "WISDM", "USC-HAD", "EHR", "HARBox"])
    parser.add_argument('--subject_split_file_path', type=str, help='subject ids')
    parser.add_argument('--train_subject_path', type=str, help='subject data')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--clients', type=int, default=7)
    parser.add_argument('--clients_h', type=int, default=2)
    parser.add_argument('--d', type=int, default=36)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--l_epochs', type=int, default=10)
    parser.add_argument('--classes', type=int, default=8)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--z_latent_dim', type=int, default=100)
    parser.add_argument('--z_batch_size', type=int, default=128)
    parser.add_argument('--fc_units', type=int, default=512)
    parser.add_argument('--g_iter', type=int, default=15)
    parser.add_argument('--g_epochs', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--N', type=int, default=30)
    parser.add_argument('--EMA', type=bool, default=False)
    parser.add_argument('--multiplier', type=float, default=0.7)
    parser.add_argument('--LR_reduced', type=bool, default=True)
    parser.add_argument('-lr_model', type=float, default=0.001)
    parser.add_argument('--lr_gen', type=float, default=0.0005)
    parser.add_argument('--lr_dis', type=float, default=0.0005)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Summary of training process:")
    print("FL framework       : H2Fed")
    print("Learning rate      : {}".format(args.lr_model))
    print("Number of communication rounds       : {}".format(args.rounds))
    print("Number of local epochs       : {}".format(args.l_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Device        : {}".format(device))
    print("=" * 80)

    return main(
        dataset=args.dataset,
        subject_split_file_path=args.subject_split_file_path,
        train_subject_path=args.train_subject_path,
        rounds=args.rounds,
        clients=args.clients,
        clients_h=args.clients_h,
        d=args.d,
        window_size=args.window_size,
        bs=args.bs,
        l_epochs=args.l_epochs,
        classes=args.classes,
        ratio=args.ratio,
        z_latent_dim=args.z_latent_dim,
        z_batch_size=args.z_batch_size,
        fc_units=args.fc_units,
        g_iter=args.g_iter,
        g_epochs=args.g_epochs,
        alpha=args.alpha,
        N=args.N,
        ema=args.EMA,
        multiplier=args.multiplier,
        lr_model=args.lr_model,
        LR_reduced=args.LR_reduced,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        device=device
    )


if __name__ == "__main__":
    run()
