from scipy import misc
import imageio
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.)) + 15
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w]) 


def preprocess(img_path, im_size, device):
    image = center_crop(imageio.imread(img_path))
    data = misc.imresize(image, [im_size, im_size]).transpose((2, 0, 1))
    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).to(device)/ 127.5 - 1.
    x = x.view(3, im_size, im_size)
    return x





def draw_loss(output_root, rec_loss_draw, kl_loss_draw, x_idx):
    mpl.style.use('seaborn')
    fig_path = os.path.join(output_root, 'loss.jpg')
    opts = {
        'title': output_root.split('/')[-1] + ' loss',
        'ylabel' : 'loss',
        'xlabel' : 'epoch',
        'legend' :['rec', 'KLD']
    }
    plt.plot(x_idx, rec_loss_draw)
    plt.plot(x_idx, kl_loss_draw)
    plt.title(opts['title'])
    plt.ylabel(opts['ylabel'])
    plt.xlabel(opts['xlabel'])
    plt.legend(opts['legend'])
    plt.savefig(fig_path)
    plt.close()