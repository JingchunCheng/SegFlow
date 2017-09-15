# SegFlow: Joint Learning for Video Object Segmentation and Optical Flow

![Alt Text](http://vllab1.ucmerced.edu/~ytsai/ICCV17/iccv17_segflow.png) 

Project webpage: https://sites.google.com/site/yihsuantsai/research/iccv17-segflow <br />
Contact: Jingchun Cheng (chengjingchun at gmail dot com)

## Paper
SegFlow: Joint Learning for Video Object Segmentation and Optical Flow <br />
Jingchun Cheng, Yi-Hsuan Tsai, Shengjin Wang and Ming-Hsuan Yang <br />
IEEE International Conference on Computer Vision (ICCV), 2017.

This is the authors' demo code described in the above paper. Please cite our paper if you find it useful for your research.

```
@inproceedings{Cheng_ICCV_2017,
  author = {J. Cheng and Y.-H. Tsai and S. Wang and M.-H. Yang},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  title = {SegFlow: Joint Learning for Video Object Segmentation and Optical Flow},
  year = {2017}
}
```

## Our Results of SegFlow
-------------------------------------------
[Optical Flow Comparison](https://www.youtube.com/watch?v=pyYbqeBteq4&feature=youtu.be)

[Comparison with Unsupervised Method](https://www.youtube.com/watch?v=MzWSGgPMTlo&feature=youtu.be)

[Comparison with Semi-supervised Method](https://www.youtube.com/watch?v=FN_ePVSDMvo&feature=youtu.be)

## Requirements
* Install `caffe` and `pycaffe` contained in this project. <br />
`cd caffe` <br />
`make all -j8` <br />
`make pycaffe`

* Download the [DAVIS 2016 dataset](http://davischallenge.org/code.html) and put it in the **data** folder.

* Download our pre-trained caffe model [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/SegFlow.caffemodel) and put it in the **model** folder.

## Testing <br />
`cd demo` <br />
`python infer.py` <br />

This code provides an example of parent net (Ours_OL) for SegFlow.

## Download Our Segmentation Results on DAVIS 2016

* SegFlow without online training step (Ours_OL) [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours_OL.zip)
* SegFlow without optical flow branch (Ours_FLO) [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours_FLO.zip)
* Final SegFlow results [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours.zip)
