import os
import subprocess
from glob import glob
import numpy

# sourcepath = '../docs'
# sourcefile = '/ILSVRC2012_validation_ground_truth.txt'
#think i need to import folder:class json

destpath = '/scratch/ewisdom/ImageNet/train'
destfile = '/scratch/ewisdom/ImageNet/train.txt'
folderfile = '/scratch/ewisdom/ImageNet/folders.txt'
mapping = '/scratch/ewisdom/ImageNet/mapping.txt'

mapDict = {}
with open(mapping, 'r') as f:
    maps = f.read().splitlines()
f.close()
for m in maps:
    mapDict[m.split(' ')[0]] = m.split(' ')[1]

ls = subprocess.Popen(['ls','-l',destpath], stdout=subprocess.PIPE)
folders = ls.stdout.read().splitlines()
print(len(folders))
quit()
with open(destfile, 'w') as f:
    for folder in folders:
        folder = str(folder).split(' ')[-1][:-1]
        if folder[0] == 'n':
            if int(mapDict[folder]) <= 100:
                ls = subprocess.Popen(['ls', '-l', destpath + '/' + folder], stdout=subprocess.PIPE)
                images = ls.stdout.read().splitlines()
                for i in images:
                    i = str(i).split(' ')[-1][:-1]
                    if i[0] == 'n':
                        f.write('{}/{} {}\n'.format(destpath + '/' + folder, i, mapDict[folder]))
f.close()