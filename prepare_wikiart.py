import os
import random
import pickle
import zipfile
import ipdb
import numpy as np
import glob
from scipy import misc
import imageio
from tqdm import tqdm
import PIL
from dlutils import download
from utils import center_crop
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def verify(filelist):
    good_list = []
    for filename in tqdm(filelist):
        if filename[-4:] == '.jpg':
            try:
                img = Image.open(filename) # open the image file
                img.verify() # verify that it is, in fact an image
                good_list.append(filename)
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
    return good_list





dataset_name = 'data/wikiart'
dataset = glob.glob('/volume/annahung-project/image_generation/draw-the-music/annadraw/dataset/wikiart/*/*')
print('dataset:', len(dataset))

names = verify(dataset)

folds = 5

random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)
count_per_fold = count // folds

i = 0
im = 0
for imgfile in tqdm(names):
    image = center_crop(imageio.imread(imgfile))
    images[imgfile] = image
    im += 1

    if im == count_per_fold:
        output = open(dataset_name +'_data_fold_%d.pkl' % i, 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
