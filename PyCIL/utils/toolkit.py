import os
import numpy as np
import torch

mapping = {}
mapping['n02119789'] = 1
mapping['n02100735'] = 2
mapping['n02110185'] = 3
mapping['n02096294'] = 4
mapping['n02102040'] = 5
mapping['n02066245'] = 6
mapping['n02509815'] = 7
mapping['n02124075'] = 8
mapping['n02417914'] = 9
mapping['n02123394'] = 10
mapping['n02125311'] = 11
mapping['n02423022'] = 12
mapping['n02346627'] = 13
mapping['n02077923'] = 14
mapping['n02110063'] = 15
mapping['n02447366'] = 16
mapping['n02109047'] = 17
mapping['n02089867'] = 18
mapping['n02102177'] = 19
mapping['n02091134'] = 20
mapping['n02092002'] = 21
mapping['n02071294'] = 22
mapping['n02442845'] = 23
mapping['n02504458'] = 24
mapping['n02092339'] = 25
mapping['n02098105'] = 26
mapping['n02096437'] = 27
mapping['n02114712'] = 28
mapping['n02105641'] = 29
mapping['n02128925'] = 30
mapping['n02091635'] = 31
mapping['n02088466'] = 32
mapping['n02096051'] = 33
mapping['n02117135'] = 34
mapping['n02138441'] = 35
mapping['n02097130'] = 36
mapping['n02493509'] = 37
mapping['n02457408'] = 38
mapping['n02389026'] = 39
mapping['n02443484'] = 40
mapping['n02110341'] = 41
mapping['n02089078'] = 42
mapping['n02086910'] = 43
mapping['n02445715'] = 44
mapping['n02093256'] = 45
mapping['n02113978'] = 46
mapping['n02106382'] = 47
mapping['n02441942'] = 48
mapping['n02113712'] = 49
mapping['n02113186'] = 50
mapping['n02105162'] = 51
mapping['n02415577'] = 52
mapping['n02356798'] = 53
mapping['n02488702'] = 54
mapping['n02123159'] = 55
mapping['n02098413'] = 56
mapping['n02422699'] = 57
mapping['n02114855'] = 58
mapping['n02094433'] = 59
mapping['n02111277'] = 60
mapping['n02132136'] = 61
mapping['n02119022'] = 62
mapping['n02091467'] = 63
mapping['n02106550'] = 64
mapping['n02422106'] = 65
mapping['n02091831'] = 66
mapping['n02120505'] = 67
mapping['n02104365'] = 68
mapping['n02086079'] = 69
mapping['n02112706'] = 70
mapping['n02098286'] = 71
mapping['n02095889'] = 72
mapping['n02484975'] = 73
mapping['n02137549'] = 74
mapping['n02500267'] = 75
mapping['n02129604'] = 76
mapping['n02090721'] = 77
mapping['n02396427'] = 78
mapping['n02108000'] = 79
mapping['n02391049'] = 80
mapping['n02412080'] = 81
mapping['n02108915'] = 82
mapping['n02480495'] = 83
mapping['n02110806'] = 84
mapping['n02128385'] = 85
mapping['n02107683'] = 86
mapping['n02085936'] = 87
mapping['n02094114'] = 88
mapping['n02087046'] = 89
mapping['n02100583'] = 90
mapping['n02096177'] = 91
mapping['n02494079'] = 92
mapping['n02105056'] = 93
mapping['n02101556'] = 94
mapping['n02123597'] = 95
mapping['n02481823'] = 96
mapping['n02105505'] = 97
mapping['n02088094'] = 98
mapping['n02085782'] = 99
mapping['n02489166'] = 100


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc

def find_line_with_string(filename, string):
    with open(filename, 'r') as file:
        for line in file:
            if string in line:
                return line

def split_images_labels(imgs, key):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    if key == "train":
        for item in imgs:
            images.append(item[0])
            labels.append(mapping[item[0].split('/')[-2]] - 1)
            # labels.append(item[1]-1)
    if key == "test":
        for item in imgs:
            line = find_line_with_string("./mytest.txt", item[0].split('/')[-1])
            if not line:
                continue
            images.append(item[0])
            labels.append(int(line.split(' ')[-1]) - 1)

    return np.array(images), np.array(labels)
