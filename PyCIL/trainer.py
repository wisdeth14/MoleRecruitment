import sys
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import inspect
import csv
from moles import *


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    classes = None
    if args["dataset"] == 'cifar100':
        classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    if args["dataset"] == 'imagenet100':
        classes = ['kit_fox', 'English_setter', 'Siberian_husky', 'Australian_terrier', 'English_springer',
                   'grey_whale', 'lesser_panda', 'Egyptian_cat', 'ibex', 'Persian_cat', 'cougar', 'gazelle',
                   'porcupine', 'sea_lion', 'malamute', 'badger', 'Great_Dane', 'Walker_hound',
                   'Welsh_springer_spaniel', 'whippet', 'Scottish_deerhound', 'killer_whale', 'mink',
                   'African_elephant', 'Weimaraner', 'soft-coated_wheaten_terrier', 'Dandie_Dinmont', 'red_wolf',
                   'Old_English_sheepdog', 'jaguar', 'otterhound', 'bloodhound', 'Airedale', 'hyena', 'meerkat',
                   'giant_schnauzer', 'titi', 'three-toed_sloth', 'sorrel', 'black-footed_ferret', 'dalmatian',
                   'black-and-tan_coonhound', 'papillon', 'skunk', 'Staffordshire_bullterrier', 'Mexican_hairless',
                   'Bouvier_des_Flandres', 'weasel', 'miniature_poodle', 'Cardigan', 'malinois', 'bighorn',
                   'fox_squirrel', 'colobus', 'tiger_cat', 'Lhasa', 'impala', 'coyote', 'Yorkshire_terrier',
                   'Newfoundland', 'brown_bear', 'red_fox', 'Norwegian_elkhound', 'Rottweiler', 'hartebeest', 'Saluki',
                   'grey_fox', 'schipperke', 'Pekinese', 'Brabancon_griffon', 'West_Highland_white_terrier',
                   'Sealyham_terrier', 'guenon', 'mongoose', 'indri', 'tiger', 'Irish_wolfhound', 'wild_boar',
                   'EntleBucher', 'zebra', 'ram', 'French_bulldog', 'orangutan', 'basenji', 'leopard',
                   'Bernese_mountain_dog', 'Maltese_dog', 'Norfolk_terrier', 'toy_terrier', 'vizsla', 'cairn',
                   'squirrel_monkey', 'groenendael', 'clumber', 'Siamese_cat', 'chimpanzee', 'komondor', 'Afghan_hound',
                   'Japanese_spaniel', 'proboscis_monkey']

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    if args["moles"]:
        if args["dataset"] == 'cifar100':
            table = np.load('./probabilitymatrix_cifar-100.npy', allow_pickle=True)
            moles = moleRecruitment(table, table.shape[1])
        if args["dataset"] == 'imagenet100':
            table = np.load('./probabilitymatrix_imagenet_subset.npy', allow_pickle=True)
            moles = moleRecruitment_imagenet(table)
    id_attacked, attacked_confounding = {}, {}

    seen, prev_attacked = [], []
    file = '/{}_moles{}_seed{}_rho{}_BACKUP.txt'.format(args["dataset"], args["moles"], args["seed"], args["rho"])
    if args["test"]:
        f = open('mylogs/test.txt', 'w', newline='')
    else:
        f = open('mylogs/' + str(args["approach"]) + file, 'w', newline='')
    writer = csv.writer(f)

    for task in range(data_manager.nb_tasks):
        writer.writerow(['*' * 108])
        writer.writerow(['Task {:2d}'.format(task)])
        writer.writerow(['*' * 108])
        print('*' * 108)
        print('Task {:2d}'.format(task))
        print('*' * 108)

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        if task == 0:
            adder = args["init_cls"]
        else:
            adder = args['increment']
        task_classes = [classes[c] for c in data_manager._class_order[len(seen):len(seen) + adder]]
        mole_indices = None
        if task and args["moles"]:
            combos, prev_attacked = selectMultiCombo(moles, prev_attacked, args["rho"], seen, task_classes, classes)
            print(combos)
            if (combos):
                mole_indices = moleSetMulti(moles, combos, table, task_classes, classes, task, data_manager.nb_tasks, args["batch_size"])
            #mole_indices = None #DELETE
            for percentile, attacked, confounding in combos:
                # id_attacked[self.id_order[self.class_order.index(attacked)]] = attacked
                if attacked in attacked_confounding.keys():
                    attacked_confounding[attacked].append(confounding)
                else:
                    attacked_confounding[attacked] = [confounding]
        model.incremental_train(data_manager, mole_indices) #two steps of training here: nominal then moles

        for c in task_classes:
            seen.append(c)

        cnn_accy, nme_accy = model.eval_task()

        attacked_stats, confounding_list = [], []
        for i, r in enumerate(model.confusion_matrix[:len(seen)]): #:10 * (task + 1)
            writer.writerow(['{0:15s} - {1:.1f}|'.format(classes[data_manager._class_order[i]], r[i] / np.sum(r) * 100)])
            print('{0:15s} - {1:.1f}|'.format(classes[data_manager._class_order[i]], r[i] / np.sum(r) * 100))
            if classes[data_manager._class_order[i]] in attacked_confounding.keys():
                attacked_stats.append('{0:15s} - {1:.1f}'.format(classes[data_manager._class_order[i]], r[i] / np.sum(r) * 100))
                confounding_list.append(attacked_confounding[classes[data_manager._class_order[i]]])
        if task and args["moles"]:
            writer.writerow(["ATTACKED CLASSES"])
            print("ATTACKED CLASSES")
            for i, j in zip(attacked_stats, confounding_list):
                writer.writerow([str(i) + '     confounding = ' + str(j) + '|'])
                print(i, '     confounding = ', j, "|")
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

    f.close()


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            #device = torch.device("cuda:{}".format(device))
            device = torch.device("cuda")

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
