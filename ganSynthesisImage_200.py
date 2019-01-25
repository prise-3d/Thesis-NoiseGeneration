#!/usr/bin/env python
import random
import argparse
import cv2
import os, sys, getopt

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from PIL import Image

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 400
DISCR_FILTERS = 200
GENER_FILTERS = 200
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 200
input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

BACKUP_MODEL_NAME = "synthesis_{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 20
MAX_ITERATION = 100000

data_folder = 'synthesis_images/generated_blocks'

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=DISCR_FILTERS * 16,
                      kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 16),
            nn.ReLU(),

            nn.Conv2d(in_channels=DISCR_FILTERS * 16, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 5,
                               kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 5, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 3,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 3),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 3, out_channels=GENER_FILTERS * 2,
                               kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

# here we have to generate our batches from final or noisy synthesis images
def iterate_batches(batch_size=BATCH_SIZE):

    batch = []
    images = os.listdir(data_folder)
    nb_images = len(images)

    while True:
        i = random.randint(0, nb_images - 1)

        img = Image.open(os.path.join(data_folder, images[i]))
        img_arr = np.asarray(img)

        new_obs = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)

        batch.append(new_obs)

        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()


if __name__ == "__main__":

    save_model = False
    load_model = False
    p_cuda = False

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda computation")
    #args = parser.parse_args()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hflc", ["help=", "folder=", "load=", "cuda="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python ganSynthesisImage_200.py --folder folder_name_to_save --cuda 1')
        print('python ganSynthesisImage_200.py --load model_name_to_load ')
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            print('python ganSynthesisImage_200.py --folder folder_name_to_save --cuda 1')
            print('python ganSynthesisImage_200.py --load folder_name_to_load ')
            sys.exit()
        elif o in ("-f", "--folder"):
            p_model_folder = a
            save_model = True
        elif o in ("-l", "--load"):
            p_load = a
            load_model = True
        elif o in ("-c", "--cuda"):
            p_cuda = int(a)
        else:
            assert False, "unhandled option"

    if save_model and load_model:
        raise Exception("Cannot save and load model. One argurment in only required.")
    if not save_model and not load_model:
        print('python ganSynthesisImage_200.py --folder folder_name_to_save --cuda 1')
        print('python ganSynthesisImage_200.py --load folder_name_to_load ')
        print("Need at least one argurment.")
        sys.exit(2)

    device = torch.device("cuda" if p_cuda else "cpu")
    #envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]


    # prepare folder names to save models
    if save_model:

        models_folder_path = os.path.join(BACKUP_FOLDER, p_model_folder)
        dis_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('disc'))
        gen_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('gen'))

    if load_model:

        models_folder_path = os.path.join(BACKUP_FOLDER, p_load)
        dis_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('disc'))
        gen_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('gen'))

    # Construct model
    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)
    print(net_discr)
    print(net_gener)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)


    # load models checkpoint if exists
    if load_model:
        gen_checkpoint = torch.load(gen_model_path)

        net_gener.load_state_dict(gen_checkpoint['model_state_dict'])
        gen_optimizer.load_state_dict(gen_checkpoint['optimizer_state_dict'])
        gen_losses = gen_checkpoint['gen_losses']
        iteration = gen_checkpoint['iteration'] # retrieve only from the gen net the iteration number

        dis_checkpoint = torch.load(dis_model_path)

        net_discr.load_state_dict(dis_checkpoint['model_state_dict'])
        dis_optimizer.load_state_dict(dis_checkpoint['optimizer_state_dict'])
        dis_losses = dis_checkpoint['dis_losses']

        iter_no = iteration

    for batch_v in iterate_batches():

        # generate extra fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)

        # There we get data
        batch_v = batch_v.to(device)

        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        print("Iteration : ", iter_no)

        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []

        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(gen_output_v.data[:IMAGE_SIZE], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(batch_v.data[:IMAGE_SIZE], normalize=True), iter_no)

        if iter_no % BACKUP_EVERY_ITER == 0:
            if not os.path.exists(models_folder_path):
                os.makedirs(models_folder_path)

            torch.save({
                        'iteration': iter_no,
                        'model_state_dict': net_gener.state_dict(),
                        'optimizer_state_dict': gen_optimizer.state_dict(),
                        'gen_losses': gen_losses
                    }, gen_model_path)

            torch.save({
                        'iteration': iter_no,
                        'model_state_dict': net_discr.state_dict(),
                        'optimizer_state_dict': dis_optimizer.state_dict(),
                        'dis_losses': dis_losses
                    }, dis_model_path)





