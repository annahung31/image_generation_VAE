import torch
import torch.optim as optim
import multiprocessing
import time
from utils import center_crop, preprocess
import numpy as np
from net import *
import ipdb
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image



# generate n=num images using the model
def generate(model, z_size, num, device):
    model.eval()
    z = torch.randn(num, z_size).to(device)
    with torch.no_grad():
        x_gen = model.decode(z)
        resultsample = x_gen * 0.5 + 0.5
        resultsample = resultsample.cpu()
        return resultsample

# returns pytorch tensor z
def get_z(im, model, device):
    model.eval()
    im = torch.unsqueeze(im, dim=0).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(im)
        z = model.reparameterize(mu, logvar)

    return z


def linear_interpolate(im1, im2, interp_num, model, device):
    model.eval()
    z1 = get_z(im1, model, device)
    z2 = get_z(im2, model, device)

    factors = np.linspace(1, 0, num=interp_num)
    result = []

    with torch.no_grad():

        for f in factors:
            z = (f * z1 + (1 - f) * z2).to(device)
            x_gen = model.decode(z)
            resultsample = x_gen * 0.5 + 0.5
            im = torch.squeeze(resultsample.cpu())
            result.append(im)

    print(len(result))
    return result

def load_model(model_fn, model):
    #model.load_state_dict(torch.load(args.vqvae_path)['vqvae_state_dict'],strict=False)
    S = torch.load(model_fn, map_location='cpu')
    model.load_state_dict(S, strict=False)

    model.eval()
    for p in model.parameters():
            p.requires_grad = False
    return model

if __name__ == '__main__':
    USE_CUDA = False
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device:', device)
    

    im_size = 128
    z_size = 512
    MODEL_PATH = 'checkpoints/wikiart_0/wikiart_0VAEmodel.pkl'
    MODEL = 'wikiart'
    OUTPUT_PATH = 'outputs/'
    num = 5
    
    vae = VAE(zsize=z_size, layer_count=5)
    vae = load_model(MODEL_PATH, vae)
    #load model
    vae.to(device)
    
    '''
    generate images using model
    '''
    samples = generate(vae, z_size, 5, device)
    save_image(samples, OUTPUT_PATH + MODEL + '.png', padding=0, nrow=10)


    '''
    linear interpolate
    '''
    
    imgs = [
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/Pop_Art/00006.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/Pop_Art/00001.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/Pop_Art/00002.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/Pop_Art/00003.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/Pop_Art/00008.jpg'
    ]
    '''
    imgs = [
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba/023423.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba/023448.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba/032454.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba/013429.jpg',
        '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba/012321.jpg',
    ]
    '''
    # dataset_root = '/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/face/images/img_align_celeba'
    # man_sunglasses_ids = ['172624.jpg', '164754.jpg', '089604.jpg', '024726.jpg']
    # man_ids = ['056224.jpg', '118398.jpg', '168342.jpg']
    # woman_smiles_ids = ['168124.jpg', '176294.jpg', '169359.jpg']
    # woman_ids = ['034343.jpg', '066393.jpg']

    imgA = preprocess(imgs[0], im_size, device)
    imgB = preprocess(imgs[1], im_size, device)
    imgC = preprocess(imgs[2], im_size, device)
    imgD = preprocess(imgs[3], im_size, device)
    imgE = preprocess(imgs[4], im_size, device)

    interp_num = 30
    inter1 = linear_interpolate(imgA, imgB, interp_num, vae, device)
    inter2 = linear_interpolate(imgB, imgC, interp_num, vae, device)
    inter3 = linear_interpolate(imgC, imgD, interp_num, vae, device)
    inter4 = linear_interpolate(imgD, imgE, interp_num, vae, device)

    for i, item in enumerate(inter1+inter2+inter3+inter4):
        save_image(item , OUTPUT_PATH + 'art_interpolate-dfc' + str(i) +'.png', padding=0, nrow=1)