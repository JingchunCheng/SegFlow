import os,sys
sys.path.append("../caffe/")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")
sys.path.insert(0, "../python_layers/")
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


davis_dir = '../data/DAVIS/'
split_f  = '{}/ImageSets/480p/train.txt'.format(davis_dir)

caffe_model  = '../models/SegFlow.caffemodel'
deploy_proto = '../prototxts/deploy.prototxt'
file_out     = '../results/Res_SegFlow/'
device_id    = 0


# init
caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(deploy_proto , caffe_model, caffe.TEST)

indices = open(split_f, 'r').read().splitlines()
print >> sys.stderr, 'Total Number of Images: {}'.format(len(indices))


for idx in range(len(indices)):
    clip1 = indices[idx].split(' ')[0].split('/')[-2]
    clip2 = indices[idx+1].split(' ')[0].split('/')[-2]



    # load image + label image pair
    im_name_1 = '{}/{}'.format(davis_dir, indices[idx].split(' ')[0])
    im_name_2 = '{}/{}'.format(davis_dir, indices[idx+1].split(' ')[0])

    if clip1 != clip2 : 
        im_name_2 = im_name_1
      
    img_name = indices[idx].split(' ')[1]
    ss = img_name.split('/')
    ss = ss[len(ss)-1]
    ss = ss[0:len(ss)-4]
    flow_name   = '{}/{}/{}.mat'.format(file_out, clip1, ss) 
    seg_name    = '{}/{}/{}.jpg'.format(file_out, clip1, ss) 

    if os.path.exists(file_out) == False:
        os.mkdir(file_out)

    if os.path.exists('{}/{}'.format(file_out, clip1)) == False:
        os.mkdir('{}/{}'.format(file_out, clip1))

    img1 = load_image(im_name_1)
    img2 = load_image(im_name_2)

    net.blobs['data'].reshape(1,  *img1.shape) 
    net.blobs['data2'].reshape(1, *img2.shape)
    net.blobs['data'].data[...] = img1
    net.blobs['data2'].data[...] = img2
    

    net.forward()

    print(im_name_2)
    out1 = net.blobs['score'].data[0].argmax(axis=0)
    out1 = np.array(out1, dtype=np.float32)
    res_img = Image.fromarray(out1)
    res_img.convert('L').save(seg_name)

    out2 = net.blobs['score_flow'].data
    io.savemat(flow_name, {'flo': out2})



print('done')

