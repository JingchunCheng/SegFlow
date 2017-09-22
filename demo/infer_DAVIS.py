import os,sys
sys.path.append("../caffe/")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")
sys.path.insert(0,"../fcn_python/")


import caffe
import surgery

import numpy as np
from PIL import Image
import scipy.io

from scipy.misc import imresize

import os
from scipy import io

import shutil


def load_image(im_name):
      # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(im_name)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    print >> sys.stderr, 'loading {}'.format(im_name)
    return in_

cls_name = sys.argv[1]

davis_dir = '../data/DAVIS/'
split_f  = '{}/ImageSets/480p/train.txt'.format(davis_dir)

caffe_model  = '../model/SegFlow.caffemodel'
deploy_proto = '../model/deploy.prototxt'
file_out     = '../results/DAVIS/'
device_id    = 0


# init
caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(deploy_proto , caffe_model, caffe.TEST)

file_path = '{}/JPEGImages/480p/{}'.format(davis_dir, cls_name)
images    = os.listdir(file_path)


for idx in range(len(images)):
    if idx == len(images) - 1:
      im_name_1 = '{}/{}'.format(file_path,images[idx])
      im_name_2 = im_name_1
    else:
      im_name_1 = '{}/{}'.format(file_path,images[idx])
      im_name_2 = '{}/{}'.format(file_path,images[idx+1])


    ss = images[idx].split('.jpg')
    ss = ss[0]
    flow_name   = '{}/{}/{}.mat'.format(file_out, cls_name, ss)
    seg_name    = '{}/{}/{}.jpg'.format(file_out, cls_name, ss)

    if os.path.exists(file_out) == False:
        os.mkdir(file_out)

    if os.path.exists('{}/{}'.format(file_out, cls_name)) == False:
        os.mkdir('{}/{}'.format(file_out, cls_name))

    img1 = load_image(im_name_1)
    img2 = load_image(im_name_2)

    net.blobs['data'].reshape(1,  *img1.shape)
    net.blobs['data2'].reshape(1, *img2.shape)
    net.blobs['data'].data[...]  = img1
    net.blobs['data2'].data[...] = img2


    net.forward()

    print(im_name_2)
    out1 = net.blobs['score'].data[0].argmax(axis=0)*255
    out1 = np.array(out1, dtype=np.float32)
    res_img = Image.fromarray(out1)
    res_img.convert('L').save(seg_name)

    out2 = net.blobs['score_flow'].data
    io.savemat(flow_name, {'flo': out2})


print('done')
