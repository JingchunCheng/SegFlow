import os,sys
sys.path.append("../caffe")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")
sys.path.insert(0, "../fcn_python/")
sys.path.insert(0, "../python_layers/")

import caffe
import surgery

import numpy as np


solver_proto = sys.argv[1]
gpu_id       = np.int(sys.argv[2])


caffe.set_mode_gpu()
caffe.set_device(gpu_id)

solver = caffe.SGDSolver(solver_proto)

weights_seg = '../model/ResNet-101-model.caffemodel'
weights_flow = '../model/flownets_conv1rename.caffemodel'
solver.net.copy_from(weights_seg)
solver.net.copy_from(weights_flow)


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for _ in range(20):
    solver.step(10000)
