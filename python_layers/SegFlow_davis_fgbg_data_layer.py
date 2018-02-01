import caffe

import numpy as np
from PIL import Image

import cv2
from scipy.misc import imresize
from scipy.misc import imrotate

import sys

import random

class SegFlowDAVISFgBgDataLayer(caffe.Layer):
    """
    Load (frm1, frm2, label_segmentation, weight_fgbg) pairs from DAVIS2016 one at a time.
    label = [0,1] for (bg, fg)
    frm2  = frm1 at the end of each video
    """

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.davis_dir = params['davis_dir']
        self.split     = params['split']
        self.mean      = np.array(params['mean'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.augment    = params.get('with_augmentation', True)
        self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 480
        self.W         = 854


        # flow tops: data1, data2, label, weight 
        if len(top) != 4:
            raise Exception("Need to define four tops: data1, data2, label, weight")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/480p/{}.txt'.format(self.davis_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = -1 # we pick idx in reshape

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]
           self.max_rotate      = self.aug_params[2]
           self.max_transW      = self.aug_params[3]
           self.max_transH      = self.aug_params[4]
           self.flip            = (self.aug_params[5]>0)


        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        while True:
            # pick next input
            if self.random:
                self.idx = random.randint(0, len(self.indices)-1)
            else:
                self.idx += 1
                if self.idx == len(self.indices):
                    self.idx = 0
           

            if self.idx == (len(self.indices) - 1):
               continue
        

            idx1 = self.idx
            idx2 = idx1 + 1
            if idx2 == (len(self.indices) - 1):
               idx2 = idx1

            clip1 = self.indices[idx1].split(' ')[0].split('/')[-2]            
            clip2 = self.indices[idx2].split(' ')[0].split('/')[-2]

            if clip1 != clip2:
               idx2 = idx1
            
            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.img1   = self.load_image(self.indices[idx1].split(' ')[0])
               self.img2   = self.load_image(self.indices[idx2].split(' ')[0])
               self.label  = self.load_label(self.indices[idx1].split(' ')[1])
               self.img1   = imresize(self.img1,    size=(self.H, self.W), interp="bilinear")
               self.img2   = imresize(self.img2,    size=(self.H, self.W), interp="bilinear")
               self.label  = imresize(self.label,   size=(self.H, self.W), interp="nearest")
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  (random.random()*2-1) * self.max_rotate
               trans_w     =  np.int( (random.random()*2-1) * self.max_transW * self.W )
               trans_h     =  np.int( (random.random()*2-1) * self.max_transH * self.H )
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.img1   = self.load_image_transform(self.indices[idx1].split(' ')[0], scale, rotation, trans_h, trans_w, flip)
               self.img2   = self.load_image_transform(self.indices[idx2].split(' ')[0], scale, rotation, trans_h, trans_w, flip)
               self.label  = self.load_label_transform(self.indices[idx1].split(' ')[1], scale, rotation, trans_h, trans_w, flip)


 #           if self.scale != 1:
 #              self.img1   = imresize(self.img1,    size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="bilinear")
 #              self.img2   = imresize(self.img2,    size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="bilinear")
 #              self.label  = imresize(self.label,   size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="nearest")

            self.weight = self.calculate_weight(self.label)

            self.img1 = self.img1.transpose((2,0,1))
            self.img2 = self.img2.transpose((2,0,1))
            break            

        # reshape tops to fit (leading 2 is for batch dimension)
        top[0].reshape(1, *self.img1.shape)
        top[1].reshape(1, *self.img2.shape)
        top[2].reshape(1, *self.label.shape)
        top[3].reshape(1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img1
        top[1].data[...] = self.img2
        top[2].data[...] = self.label
        top[3].data[...] = self.weight

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print >> sys.stderr, 'loading Original {}'.format(idx)
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        im  = im.resize((self.W, self.H))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        im  = im.resize((self.W, self.H), Image.NEAREST)
        label = np.array(im, dtype=np.uint8)
        label = np.uint8((label>0))
        
        return label


    def calculate_weight(self, label):
       weight    = np.zeros_like(label, dtype = np.float32) 
       num_class = np.max(label) + 1
       for class_id in range(num_class):
           pos        = np.where(label == class_id)
           weight_idx = np.float32(1 - len(pos[0])*1.0/weight.size)
           print >> sys.stderr, 'Class {}, weight = {}'.format(class_id, weight_idx)
           for idx  in range (len(pos[0])):
               weight[pos[0][idx], pos[1][idx]] = weight_idx
       
       return weight


    def load_image_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
       img_W = np.int( self.W*(1.0 + scale) )
       img_H = np.int( self.H*(1.0 + scale) ) 

       print >> sys.stderr, 'loading {}'.format(idx)
       print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

       im    = Image.open('{}/{}'.format(self.davis_dir, idx))
       im    = im.resize((img_W,img_H))
       im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
       im    = im.rotate(rotation)
       if flip:
          im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
       if scale>0:
          box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
          im  = im.crop(box)
       else:
          im  = im.resize((self.W, self.H))
       

       in_ = np.array(im, dtype=np.float32)
       in_ = in_[:,:,::-1]
       in_ -= self.mean  

       return in_


    def load_label_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) )
        

        im    = Image.open('{}/{}'.format(self.davis_dir, idx))
        im    = im.resize((img_W,img_H), Image.NEAREST)
        im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
           im = im.transpose(Image.FLIP_LEFT_RIGHT)

        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H), Image.NEAREST)

        label = np.array(im, dtype=np.uint8)
        label = np.uint8((label>0))

        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        return label




