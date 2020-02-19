# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
import argparse
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from utils import draw_loss
from PIL import Image

im_size = 128


def loss_function(recon_x, x, mu, logvar):
    # recon_loss = torch.mean((recon_x - x)**2)
    l1loss = nn.L1Loss()
    recon_loss = l1loss(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return recon_loss, KLD * 0.1


def process_batch(batch):
    # data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]
    data = [ np.array(Image.fromarray(x).resize([im_size, im_size])).transpose((2, 0, 1)) for x in batch]
   

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def main(args):
    if args.dataset == 'face':
        dataset_name = './data/'
    else:
        dataset_name = os.path.join('data', args.dataset+'_')
    output_root = os.path.join('checkpoints',args.exp_name)
    result_rec_pth = os.path.join(output_root, 'results_rec')
    result_gen_pth = os.path.join(output_root, 'results_gen')
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(result_rec_pth, exist_ok=True)
    os.makedirs(result_gen_pth, exist_ok=True)


    batch_size = args.batch_size
    z_size = 512
    vae = VAE(zsize=z_size, layer_count=5)
    # vae = ResEncoder(ndf=16, latent_variable_size=512)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)
    fold = 5
    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch = args.num_epoch

    sample1 = torch.randn(batch_size, z_size).view(-1, z_size, 1, 1)

    rec_loss_draw = []
    kl_loss_draw = []
    x_idx = []
    for epoch in range(train_epoch):
        vae.train()

        with open(dataset_name + 'data_fold_%d.pkl' % (epoch % fold), 'rb') as pkl:
            data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=False)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0 and (epoch + 1) < 50:
            vae_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")


        i = 0
        
        for x in batches:
            vae.train()
            vae.zero_grad()
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################


            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))

                rec_loss_draw.append(rec_loss)
                kl_loss_draw.append(kl_loss)
                x_idx.append(float('%.2f' % (epoch + (i/(len(data_train)/batch_size)))))
                print(x_idx)
                
                rec_loss = 0
                kl_loss = 0
                
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               result_rec_pth + '/sample_' + str(epoch) + "_" + str(i) + '.png')
                    
                    x_gen = vae.decode(sample1)
                    resultsample = x_gen * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               result_gen_pth + '/sample_' + str(epoch) + "_" + str(i) + '.png')

        del batches
        del data_train
        # draw_loss(output_root, rec_loss_draw, kl_loss_draw, x_idx)

    print("Training finish!... save training results")
    torch.save(vae.state_dict(), os.path.join(output_root, "VAEmodel.pkl"))

def get_arguments():

    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--exp_name', help="experiment name", default="test")
    parser.add_argument('--dataset', help="dataset type. [face | wikiart]", default="face")
    parser.add_argument('--num_epoch', type=int, help="total number of epoch for training", default=40)
    parser.add_argument('--batch_size', type=int, help="batch size for processing", default=128)
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_arguments()
    main(args)
