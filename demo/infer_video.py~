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

import cv2

def img_transform(im):
    im  = imresize(im, size = (480,854), interp="bilinear")
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    return in_

video_name = sys.argv[1]

caffe_model  = '../model/SegFlow.caffemodel'
deploy_proto = '../model/deploy.prototxt'
file_out     = '../results/' + video_name[8:-4] + '/'

device_id    = 1  

# init network
caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(deploy_proto , caffe_model, caffe.TEST)


if os.path.exists(file_out) == False:
   os.mkdir(file_out)


vid = cv2.VideoCapture(video_name)
flag, frame = vid.read()
print(flag)
num = 0
while flag:
    num  = num + 1
    img1 = img_transform(frame)
    flag, frame = vid.read()
    if flag == False:
       img2 = img1
    else:
       img2 = img_transform(frame)

    flow_name   = '{}/{:0>5d}.mat'.format(file_out, num)
    seg_name    = '{}/{:0>5d}.jpg'.format(file_out, num)

    net.blobs['data'].reshape(1,  *img1.shape) 
    net.blobs['data2'].reshape(1, *img2.shape)
    net.blobs['data'].data[...]  = img1
    net.blobs['data2'].data[...] = img2
    
    net.forward()

    print ('Processing frame {:0>5d}'.format(num))

    out1 = net.blobs['score'].data[0].argmax(axis=0)
    out1 = np.array(out1, dtype=np.float32)*255
    res_img = Image.fromarray(out1)
    res_img.convert('L').save(seg_name)

    out2 = net.blobs['score_flow'].data
    io.savemat(flow_name, {'flo': out2})    



print('done')

