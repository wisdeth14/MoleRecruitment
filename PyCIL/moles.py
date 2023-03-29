import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def moleSearch(device, total_set):

    model, preprocess = clip.load('ViT-B/32', device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in total_set.classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    table = []
    for i in range(len(total_set)):
        if (i % 1000 == 0):
            print("{} data samples processed".format(i))
        image, class_id = total_set[i]  #total_set[i][0], total_set[i][1]  # total_set[3637]
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        row = torch.Tensor.tolist(similarity[0]) + [class_id] + [i]
        table.append(row)
    table_np = np.array(table)

    matrix = []
    for i in range(len(total_set.classes)):
        index = (table_np[:, len(total_set.classes)] == i)
        matrix.append(table_np[index])
    matrix_np = np.array(matrix, dtype=object)
    return matrix_np

def moleRecruitment(matrix, N):
    moles = []
    print(len(matrix))
    for i in range(len(matrix)):
        partialpoison = []
        for j in range(matrix.shape[-1] - 2):
            part = np.flip(matrix[i][matrix[i][:, j].argsort()], axis=0)
            partialpoison.append(part[:N])
        moles.append(partialpoison)
    moles_np = np.array(moles)
    return moles_np

def moleRecruitment_imagenet(matrix):
    moles = []
    for i in range(len(matrix)):
        partialpoison = []
        #for j in range(matrix.shape[-1] - 2): #can't use shape here! (since ndarray doesn't have equal length rows)
        for j in range(matrix[i].shape[-1] - 2):
            part = np.flip(matrix[i][matrix[i][:, j].argsort()], axis=0)
            partialpoison.append(part[:len(matrix[i])]) #use len of i instead right?
        moles.append(partialpoison)
    moles_np = np.array(moles, dtype=object)
    return moles_np

def selectMultiCombo(moles, prev_attacked, rho, past_classes, curr_classes, all_classes):

    #rho = 0.01 #optimal threshold per ablation study
    p = 99 #optimal percentile per ablation study
    percentiles, percentiles_inv = [], []
    for i in past_classes: #attacked
        for j in curr_classes: #confounding
            spread = moles[all_classes.index(j)][all_classes.index(i)][:, all_classes.index(i)]
            pc = np.percentile(spread, p)
            percentiles.append([pc, i, j])
            # percentiles.append([np.percentile(spread, p), i, j])
            # percentiles_inv.append([np.percentile(spread_inv, p), j, i])
    percentiles = sorted(percentiles, key=lambda x: float(x[0]), reverse=True)
    # percentiles_inv = sorted(percentiles_inv, key=lambda x: float(x[0]), reverse=True)
    # print(percentiles)
    # print(percentiles_inv)

    combos = []
    for i in percentiles:
        #don't want to use same confounding twice
        #what about same attacked? might be worth checking
        if i[0] >= rho and not any(i[1] in x for x in combos) and not any(i[2] in x for x in combos):
            combos.append(i)
            prev_attacked.append(i[1])

    return combos, prev_attacked


def moleSetMulti(moles, combos, table, curr_classes, all_classes, id, n_experiences, batch_size):

    m, b = 0.765, 0.101  # correlation derived from ablation study
    samples = []
    for i in combos:
        mean_target = (m * i[0]) + b
        # mean_target = (0.765 * i[0]) + 0.101 #f_99 y = 0.765x + 0.101
        # mean_target = (0.840 * 0.313) + 0.349 #f_97 y = 0.840x + 0.313
        # mean_target = (1.122 * i[0]) + 0.349 #f_95 y = 1.122x + 0.349
        mean_current = 1
        n = 1
        while mean_current > mean_target:
            mean_current = np.mean(moles[all_classes.index(i[2])][all_classes.index(i[1])][:n, all_classes.index(i[1])])
            n += 1
        samples.append(n)
        print(i, n, mean_target)
    normalize = math.floor(batch_size / len(curr_classes)) #ensures entire mole set will be roughly divisible by batch_size
    print(normalize)
    mean_sample = round(np.mean(samples) / normalize) * normalize #or have no rho threshold and have weighted average based off percentile
    if mean_sample < normalize:
        mean_sample = normalize
    print(mean_sample)

    index_moles = []
    index_random = []
    unused = []
    indices = []

    for i in combos:
        index_moles.append(moles[all_classes.index(i[2])][all_classes.index(i[1])][:mean_sample, -1].astype(int))
    for i in curr_classes:
        if i not in np.array(combos)[:, 2]:
                unused.append(i)
    for i in unused:
        index_random.append(np.random.choice(table[all_classes.index(i)][:, -1], size=mean_sample, replace=False).astype(int))
    index_random = list(np.array(index_random).flatten())
    index_moles = list(np.array(index_moles).flatten())

    return index_moles + index_random

    # #PROB GET RID OF THIS LAST PART (and just return index_moles + index_random)
    # for i in range(n_experiences):
    #     if i == id:
    #         indices.append(index_moles + index_random)
    #     else:
    #         indices.append([])
    #
    # return indices