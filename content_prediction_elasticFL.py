import copy
import time
import numpy as np
from tqdm import tqdm
import torch

from options import args_parser
from local_update import LocalUpdate
from dataset_processing import average_weights

def content_prediction(n_veh, netE, netP, netD, data_set, users_group_train):
    # args & 输出实验参数
    args = args_parser()
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # Set the model to train and send it to device.
    netE.to(device)
    netP.to(device)
    netD.to(device)

    # Encoder Weight Distance
    e_e_alpha_w1 = []
    e_e_alpha_w2 = []
    e_e_alpha_w3 = []
    e_e_alpha_w4 = []
    e_e_alpha_w_avg = []

    # Decoder Weight Distance
    e_p_alpha_w1 = []
    e_p_alpha_w2 = []
    e_p_alpha_w3 = []
    e_p_alpha_w4 = []
    e_p_alpha_w5 = []
    e_p_alpha_w_avg = []

    # Discriminator Weight Distance
    e_d_alpha_w1 = []
    e_d_alpha_w2 = []
    e_d_alpha_w3 = []
    e_d_alpha_w_avg = []

    # all epoch weights
    w_e_all_epochs = dict([(k, []) for k in range(args.epochs)])
    w_p_all_epochs = dict([(k, []) for k in range(args.epochs)])
    w_d_all_epochs = dict([(k, []) for k in range(args.epochs)])

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Elastic Federated Leaning Global Training Round : {epoch+ 1} |\n')
        # copy weights
        netE_w = netE.state_dict()
        netP_w = netP.state_dict()
        netD_w = netD.state_dict()

        # Encoder Weight Distance
        v_e_alpha_w1 = 0
        v_e_alpha_w2 = 0
        v_e_alpha_w3 = 0
        v_e_alpha_w4 = 0

        # Decoder Weight Distance
        v_p_alpha_w1 = 0
        v_p_alpha_w2 = 0
        v_p_alpha_w3 = 0
        v_p_alpha_w4 = 0
        v_p_alpha_w5 = 0

        # Discriminator Weight Distance
        v_d_alpha_w1 = 0
        v_d_alpha_w2 = 0
        v_d_alpha_w3 = 0

        local_e_weights = []
        local_p_weights = []
        local_d_weights = []

        for idx in range(n_veh):

            print(f'\n | Elastic Federated Leaning Vehicle: {idx + 1} |\n')
            local_model = LocalUpdate(args=args, dataset=data_set,
                                      idxs=users_group_train[idx])
            encoder_local, decoder_local, discriminator_local, local_epoch = local_model.update_weights(
                encoder=copy.deepcopy(netE), decoder=copy.deepcopy(netP), discriminator=copy.deepcopy(netD), clo=int(max(data_set[1, :])), lr=args.lr)

            # encoder weight 4 layer
            encoder_local_w = encoder_local.state_dict()
            encoder_local.load_state_dict(encoder_local_w)
            #decoder weight 5 layer
            decoder_local_w = decoder_local.state_dict()
            decoder_local.load_state_dict(decoder_local_w)
            # discriminator weight 3 layer
            discriminator_local_w = discriminator_local.state_dict()
            discriminator_local.load_state_dict(discriminator_local_w)

            w_e_all_epochs[epoch].append(encoder_local.state_dict()['map_L1.weight'])
            w_p_all_epochs[epoch].append(decoder_local.state_dict()['map_L1.weight'])
            w_d_all_epochs[epoch].append(discriminator_local.state_dict()['map_L1.weight'])

            local_e_weights.append(copy.deepcopy(encoder_local.state_dict()))
            local_p_weights.append(copy.deepcopy(decoder_local.state_dict()))
            local_d_weights.append(copy.deepcopy(discriminator_local.state_dict()))

        netE_w = average_weights(local_e_weights)
        netE.load_state_dict(netE_w)

        netP_w = average_weights(local_p_weights)
        netP.load_state_dict(netP_w)

        netD_w = average_weights(local_d_weights)
        netD.load_state_dict(netD_w)

        v_e_alpha_w_avg = 1 / 4 * (v_e_alpha_w1 + v_e_alpha_w2 + v_e_alpha_w3 + v_e_alpha_w4)
        v_p_alpha_w_avg = 1 / 5 * (v_p_alpha_w1 + v_p_alpha_w2 + v_p_alpha_w3 + v_p_alpha_w4 + v_p_alpha_w5)
        v_d_alpha_w_avg = 1 / 3 * (v_d_alpha_w1 + v_d_alpha_w2 + v_d_alpha_w3)

        e_e_alpha_w1.append(v_e_alpha_w1)
        e_e_alpha_w2.append(v_e_alpha_w2)
        e_e_alpha_w3.append(v_e_alpha_w3)
        e_e_alpha_w4.append(v_e_alpha_w4)
        e_e_alpha_w_avg.append(v_e_alpha_w_avg)

        e_p_alpha_w1.append(v_p_alpha_w1)
        e_p_alpha_w2.append(v_p_alpha_w2)
        e_p_alpha_w3.append(v_p_alpha_w3)
        e_p_alpha_w4.append(v_p_alpha_w4)
        e_p_alpha_w5.append(v_p_alpha_w5)
        e_p_alpha_w_avg.append(v_p_alpha_w_avg)

        e_d_alpha_w1.append(v_d_alpha_w1)
        e_d_alpha_w3.append(v_d_alpha_w3)
        e_d_alpha_w2.append(v_d_alpha_w2)
        e_d_alpha_w_avg.append(v_d_alpha_w_avg)

    return netE, netP, netD, w_e_all_epochs, w_p_all_epochs, w_d_all_epochs


