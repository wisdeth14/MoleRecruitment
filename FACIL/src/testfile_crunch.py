#this is setup to operate locally
import numpy

sourcepath = '../docs'
sourcefile = '/ILSVRC2012_validation_ground_truth.txt'

destpath = '/scratch/ewisdom/ImageNet/val'
destfile = '../docs/test.txt'

with open(sourcepath + sourcefile, 'r') as f:
    content = f.read().splitlines()
f.close()

file_no = 1
with open(destfile, 'w') as f:
    while file_no <= 50000:
        if int(content[file_no-1]) <= 100:
            f.write('{}/ILSVRC2012_val_{}.JPEG {}\n'.format(destpath, str(file_no).zfill(8), content[file_no-1]))
        file_no += 1
f.close()