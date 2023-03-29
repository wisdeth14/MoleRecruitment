import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import clip
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import csv
import sys
import os
import argparse
import math
import scipy
from PIL import Image

from datasets.dataset_config import dataset_config
from datasets.data_loader import *

parser = argparse.ArgumentParser(description='Mole Search')
parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    print('WARNING: [CUDA unavailable] Using CPU instead!')
    device = 'cpu'

print("begin")

if 'cifar100' in args.datasets:
    train_set = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True)
    test_set = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    total_set = deepcopy(train_set)
    total_set.targets = train_set.targets + test_set.targets
    total_set.data = np.concatenate((train_set.data, test_set.data))
    classes = total_set.classes
if 'imagenet_subset' in args.datasets:
    classes = ['kit_fox', 'English_setter', 'Siberian_husky', 'Australian_terrier', 'English_springer', 'grey_whale', 'lesser_panda', 'Egyptian_cat', 'ibex', 'Persian_cat', 'cougar', 'gazelle', 'porcupine', 'sea_lion', 'malamute', 'badger', 'Great_Dane', 'Walker_hound', 'Welsh_springer_spaniel', 'whippet', 'Scottish_deerhound', 'killer_whale', 'mink', 'African_elephant', 'Weimaraner', 'soft-coated_wheaten_terrier', 'Dandie_Dinmont', 'red_wolf', 'Old_English_sheepdog', 'jaguar', 'otterhound', 'bloodhound', 'Airedale', 'hyena', 'meerkat', 'giant_schnauzer', 'titi', 'three-toed_sloth', 'sorrel', 'black-footed_ferret', 'dalmatian', 'black-and-tan_coonhound', 'papillon', 'skunk', 'Staffordshire_bullterrier', 'Mexican_hairless', 'Bouvier_des_Flandres', 'weasel', 'miniature_poodle', 'Cardigan', 'malinois', 'bighorn', 'fox_squirrel', 'colobus', 'tiger_cat', 'Lhasa', 'impala', 'coyote', 'Yorkshire_terrier', 'Newfoundland', 'brown_bear', 'red_fox', 'Norwegian_elkhound', 'Rottweiler', 'hartebeest', 'Saluki', 'grey_fox', 'schipperke', 'Pekinese', 'Brabancon_griffon', 'West_Highland_white_terrier', 'Sealyham_terrier', 'guenon', 'mongoose', 'indri', 'tiger', 'Irish_wolfhound', 'wild_boar', 'EntleBucher', 'zebra', 'ram', 'French_bulldog', 'orangutan', 'basenji', 'leopard', 'Bernese_mountain_dog', 'Maltese_dog', 'Norfolk_terrier', 'toy_terrier', 'vizsla', 'cairn', 'squirrel_monkey', 'groenendael', 'clumber', 'Siamese_cat', 'chimpanzee', 'komondor', 'Afghan_hound', 'Japanese_spaniel', 'proboscis_monkey']
    path = '/scratch/ewisdom/ImageNet/train.txt'
    with open(path, 'r') as f:
        content = f.read().splitlines()
    f.close()

model, preprocess = clip.load('ViT-B/32', device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

if 'imagenet_subset' in args.datasets:
    counter = 0
    table = []
    while counter < len(content):
        images, labels = [], []
        for line in content[counter:counter+1000]:
            images.append(Image.open(line.split(' ')[0]).convert('RGB'))
            labels.append(int(line.split(' ')[1])) #imagenet is labeled 1-100
            counter += 1
        total_set = list(zip(images, labels))

        for i in range(len(total_set)):
            image, class_id = total_set[i]
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            row = torch.Tensor.tolist(similarity[0]) + [class_id] + [i+counter-1000]
            table.append(row)
        print("{} data samples processed".format(counter))
    table_np = np.array(table)
else:
    table = []
    for i in range(len(total_set)):
        if i % 1000 == 0:
            print("{} data samples processed".format(i))
        image, class_id = total_set[i]
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        row = torch.Tensor.tolist(similarity[0]) + [class_id] + [i]
        table.append(row)
    table_np = np.array(table)

matrix = []
for i in range(len(classes)):
    index = (table_np[:, len(classes)] == i)
    matrix.append(table_np[index])
matrix_np = np.array(matrix, dtype=object)

if 'cifar100' in args.datasets:
    np.save('probabilitymatrix_cifar-100', matrix_np)
if 'imagenet_subset' in args.datasets:
    np.save('probabilitymatrix_imagenet_subset', matrix_np)

