import numpy as np
import torch
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data_set import convert
import torch.optim as optim
import random

class MovieDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
        这里假设，输入的数据集为sample 的矩阵形式
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # self.user_movie = convert(self.dataset[self.idxs], max(self.dataset[:, 1]))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        ratings = self.dataset[self.idxs[item], 1:-3]
        return torch.tensor(ratings)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):

        self.args = args
        # 将输入的dataset转换为user_movie
        self.dataset = convert(dataset[idxs], int(max(dataset[:, 1])))
        self.idxs = np.arange(0, len(self.dataset))
        self.trainloader = DataLoader(MovieDataset(self.dataset, self.idxs),
                                      batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to MSE Loss function
        self.criterion = nn.MSELoss().to(self.device)


    def update_weights(self, encoder, decoder, discriminator, clo, lr):

        """
        训练本地模型，得到模型参数和训练loss
        :param model:
        :param client_idx: 客户0~9
        :param global_round: 全局回合数
        :return: model.state_dict() 模型参数
        :return: sum(epoch_loss) / len(epoch_loss) 本地训练损失
        """
        # define the optimization criterion / loss function
        reconstruction_criterion_categorical = nn.BCELoss(reduction='mean')
        reconstruction_criterion_numeric = nn.MSELoss(reduction='mean')

        # define encoder and decoder optimization strategy
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

        # init the discriminator losses
        discriminator_criterion = nn.BCELoss()

        # define generator and discriminator optimization strategy
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

        # define the number of gaussians
        tau = 5

        # define radius of each gaussian
        radius = 0.8

        # define the sigma of each gaussian
        sigma = 0.01

        # define the dimensionality of each gaussian
        dim = 2

        # determine x and y coordinates of the target mixture of gaussians
        x_centroid = (radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
        y_centroid = (radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2

        # determine each gaussians mean (centroid) and standard deviation
        mu_gauss = np.vstack([x_centroid, y_centroid]).T

        # determine the number of samples to be created per gaussian
        samples_per_gaussian = 100000

        # iterate over the number of distinct gaussians
        for i, mu in enumerate(mu_gauss):
            # case: first gaussian
            if i == 0:
                # randomly sample from gaussion distribution
                z_continous_samples_all = np.random.normal(mu, sigma, size=(samples_per_gaussian, dim))

            # case: non-first gaussian
            else:

                # randomly sample from gaussian distribution
                z_continous_samples = np.random.normal(mu, sigma, size=(samples_per_gaussian, dim))

                # collect and stack new samples
                z_continous_samples_all = np.vstack([z_continous_samples_all, z_continous_samples])

        # init collection of training losses
        epoch_reconstruction_losses = []
        epoch_discriminator_losses = []
        epoch_generator_losses = []

        epoch=1
        iter=1

        # 本地训练 训练回合数设置为local_ep
        while iter>0:

            # init mini batch counter
            mini_batch_count = 0

            # init epoch training losses
            batch_reconstruction_losses = 0.0
            batch_discriminator_losses = 0.0
            batch_generator_losses = 0.0

            # determine if GPU training is enabled
            encoder.to(self.device)
            decoder.to(self.device)
            discriminator.to(self.device)

            # set networks in training mode (apply dropout when needed)
            encoder.train()
            decoder.train()
            discriminator.train()

            # iterate over epoch mini batches
            for mini_batch_count, ratings in enumerate(self.trainloader):
                ratings = ratings.to(self.device)
                # increase mini batch counter
                mini_batch_count += 1
                # reset the networks gradients
                encoder.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                # =================== reconstruction phase =====================

                # run autoencoder encoding - decoding
                z_sample = encoder(ratings)
                mini_batch_reconstruction = decoder(z_sample)

                # split input date to numerical and categorical part
                batch_cat = ratings[:, :clo]
                batch_num = ratings[:, clo:]

                # split reconstruction to numerical and categorical part
                rec_batch_cat = mini_batch_reconstruction[:, :clo]
                rec_batch_num = mini_batch_reconstruction[:, clo:]

                # backward pass + gradients update
                rec_error_cat = reconstruction_criterion_categorical(input=rec_batch_cat,
                                                                     target=batch_cat)  # one-hot attr error
                rec_error_num = reconstruction_criterion_numeric(input=rec_batch_num,
                                                                 target=batch_num)  # numeric attr error=ratings)
                # combine both reconstruction errors
                reconstruction_loss = rec_error_cat + rec_error_num

                # run backward pass - determine gradients
                reconstruction_loss.backward()

                # collect batch reconstruction loss
                batch_reconstruction_losses += reconstruction_loss.item()

                # update network parameter - decoder and encoder
                decoder_optimizer.step()
                encoder_optimizer.step()

                # =================== regularization phase =====================
                # =================== discriminator training ===================

                # set discriminator in evaluation mode
                discriminator.eval()

                # generate target latent space data
                z_target_batch = z_continous_samples_all[
                                 random.sample(range(0, z_continous_samples_all.shape[0]), self.args.local_bs), :]

                # convert to torch tensor
                z_target_batch = torch.FloatTensor(z_target_batch)
                # determine mini batch sample generated by the encoder -> fake gaussian sample
                z_fake_gauss = encoder(ratings)
                # determine discriminator classification of both samples
                d_fake_gauss = discriminator(z_fake_gauss)  # fake created gaussian
                d_real_gauss = discriminator(z_target_batch)  # real sampled gaussian

                # determine discriminator classification target variables
                d_real_gauss_target = torch.ones(d_real_gauss.shape) # real -> 1
                d_fake_gauss_target = torch.zeros(d_fake_gauss.shape)  # fake -> 0

                # determine if GPU training is enabled
                d_real_gauss_target = d_real_gauss_target.to(self.device)
                d_fake_gauss_target = d_fake_gauss_target.to(self.device)

                # determine individual discrimination losses
                discriminator_loss_real = discriminator_criterion(target=d_real_gauss_target,
                                                                  input=d_real_gauss)  # real loss
                discriminator_loss_fake = discriminator_criterion(target=d_fake_gauss_target,
                                                                  input=d_fake_gauss)  # fake loss

                # add real loss and fake loss
                discriminator_loss = discriminator_loss_fake + discriminator_loss_real

                # run backward through the discriminator network
                discriminator_loss.backward()

                # collect discriminator loss
                batch_discriminator_losses += discriminator_loss.item()

                # update network the discriminator network parameters
                discriminator_optimizer.step()

                # reset the networks gradients
                encoder.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                # =================== regularization phase =====================
                # =================== generator training =======================

                # set encoder / generator in training mode
                encoder.train()

                # reset the encoder / generator networks gradients
                encoder.zero_grad()

                # determine fake gaussian sample generated by the encoder / generator
                z_fake_gauss = encoder(ratings)

                # determine discriminator classification of fake gaussian sample
                d_fake_gauss = discriminator(z_fake_gauss)

                # determine discriminator classification target variables
                d_fake_gauss_target = torch.ones(d_fake_gauss.shape)  # fake -> 1

                # determine discrimination loss of fake gaussian sample
                generator_loss = discriminator_criterion(target=d_fake_gauss_target, input=d_fake_gauss)

                # collect generator loss
                batch_generator_losses += generator_loss.item()

                # run backward pass - determine gradients
                generator_loss.backward()

                # update network paramaters - encoder / generatorc
                encoder_optimizer.step()

                # reset the networks gradients
                encoder.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

            # collect epoch training losses - reconstruction loss
            epoch_reconstruction_loss = batch_reconstruction_losses / mini_batch_count
            epoch_reconstruction_losses.extend([epoch_reconstruction_loss])

            # collect epoch training losses - discriminator loss
            epoch_discriminator_loss = batch_discriminator_losses / mini_batch_count
            epoch_discriminator_losses.extend([epoch_discriminator_loss])

            # collect epoch training losses - generator loss
            epoch_generator_loss = batch_generator_losses / mini_batch_count
            epoch_generator_losses.extend([epoch_generator_loss])

            # print epoch reconstruction loss
            print('epoch: {:04}, reconstruction loss: {:.4f}'.format(iter, epoch_reconstruction_loss))
            print('epoch: {:04}, discriminator loss: {:.4f}'.format(iter, epoch_discriminator_loss))
            print('epoch: {:04}, generator loss: {:.4f}'.format(iter,epoch_generator_loss))

            if epoch_reconstruction_losses[-1]>=0.5:
                epoch = epoch + 1
                iter+=1

            if epoch_reconstruction_losses[-1]<0.5 or epoch>10:
                iter=-1

            # =================== save model snapshots to disk ============================
            # # save trained encoder model file to disk
            # encoder_model_name = "ep_{}_encoder_model.pth".format(epoch)
            # torch.save(encoder.state_dict(), os.path.join("./model", encoder_model_name))
            #
            # # save trained decoder model file to disk
            # decoder_model_name = "ep_{}_decoder_model.pth".format((epoch))
            # torch.save(decoder.state_dict(), os.path.join("./model", decoder_model_name))
            #
            # # save trained discriminator model file to disk
            # discriminator_model_name = "ep_{}_discriminator_model.pth".format((epoch))
            # torch.save(discriminator.state_dict(), os.path.join("./model", discriminator_model_name))

        return encoder, decoder, discriminator, epoch



def cache_hit_ratio(test_dataset, cache_items):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100
    return CACHE_HIT_RATIO

def cache_hit(test_dataset, cache_items, cache_items_n):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]

    CACHE_HIT_NUM_N = 0
    for item in cache_items_n:
        if item not in cache_items:
            CACHE_HIT_NUM_N += count[item]

    return len(requset_items), CACHE_HIT_NUM, CACHE_HIT_NUM_N

def cache_hit1(test_dataset, cache_items):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]

    return len(requset_items), CACHE_HIT_NUM

def cache_hit3(test_dataset, cache_items, cache_items_n1, cache_items_n2):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]

    CACHE_HIT_NUM_N = 0
    CACHE_HIT_NUM_DELETE = 0
    for item in cache_items_n1:
        if item not in cache_items:
            CACHE_HIT_NUM_N += count[item]
    for item in cache_items_n2:
        if item not in cache_items:
            CACHE_HIT_NUM_N += count[item]
    for item in cache_items_n2:
        if item in cache_items_n1:
            CACHE_HIT_NUM_DELETE += count[item]
    CACHE_HIT_NUM_N -= CACHE_HIT_NUM_DELETE

    return len(requset_items), CACHE_HIT_NUM, CACHE_HIT_NUM_N