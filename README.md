SegFlow
=========================================
Introduction
-----------------------------------------
This paper proposes an end-to-end trainable network, SegFlow, for simultaneously predicting pixel-wise object segmentation and optical flow in videos. The proposed SegFlow has two branches where useful information of object segmentation and optical flow is propagated bidirectionally in a unified framework. The segmentation branch is based on a fully convolutional network, which has been proved effective in image segmentation task, and the optical flow branch takes advantage of the FlowNet model. The unified framework is trained iteratively offline to learn a generic notion, and fine tuned online for specific objects. Extensive experiments on both the video object segmentation and optical flow datasets demonstrate that introducing optical flow improves the performance of segmentation and vice versa, against the state-of-the-art algorithms.

[SegFlow]() will be published at ICCV2017.

Video Results of SegFlow
-------------------------------------------
[Optical Flow Comparison](https://www.youtube.com/watch?v=pyYbqeBteq4&feature=youtu.be)

[Comparison with Unsupervised Method](https://www.youtube.com/watch?v=MzWSGgPMTlo&feature=youtu.be)

[Comparison with Semi-supervised Method](https://www.youtube.com/watch?v=FN_ePVSDMvo&feature=youtu.be)


Citing SegFlow
-------------------------------------------
If you find SegFlow useful in your research, please consider citing:
`

`

Requirements
-------------------------------------------
Requirements for `caffe` and `pycaffe`.


Installation
-----------------------------------------------------
1. Download code and offline trained SegFlow model.

`cd $SegFlow_ROOT`

`mkdir models`

`cd models`

`wget https://www.dropbox.com/s/9ega4py8uzvya72/SegFlow.caffemodel`

2. Download DAVIS 2017 dataset and put it in `$SegFlow_ROOT/data`.

3. Install caffe and pycaffe.

`cd $SegFlow_ROOT/caffe`

`make all -j8`

`make pycaffe`

4. Run testing model.

`cd $SegFlow_ROOT/demo`

`python infer.py`

5. Train your own model.

*See [paper]() for details of model training.



Acknowledgement
--------------------------------------------------
SegFlow uses the following open source code:
* FlowNetS
* ResNet-101


