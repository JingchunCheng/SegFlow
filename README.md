# SegFlow: Joint Learning for Video Object Segmentation and Optical Flow

![Alt Text](http://vllab1.ucmerced.edu/~ytsai/ICCV17/iccv17_segflow_git.png) 

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

## SegFlow Results
[Segmentation Comparisons with Unsupervised Method](https://www.youtube.com/watch?v=MzWSGgPMTlo&feature=youtu.be)

[Segmentation Comparisons with Semi-supervised Method](https://www.youtube.com/watch?v=FN_ePVSDMvo&feature=youtu.be)

[Optical Flow Comparisons](https://www.youtube.com/watch?v=pyYbqeBteq4&feature=youtu.be)

## Requirements
* Install `caffe` and `pycaffe` contained in this project. <br />
`cd caffe` <br />
`make all -j8` <br />
`make pycaffe`

* Download the [DAVIS 2016 dataset](http://davischallenge.org/code.html) and put it in the **data** folder.

* Download our pre-trained caffe model [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/SegFlow.caffemodel) and put it in the **model** folder.

## Demo on DAVIS 2016 <br />
`cd demo` <br />
`python infer.py VIDEO_NAME` <br />
For example, run `python infer.py lions`

This code provides a demo for the parent net (Ours_OL) in SegFlow. The output contains both the segmentation and optical flow results.

## Test on your own Videos <br />
`cd demo` <br />
`python infer_video.py VIDEO_FILE` <br />
For example, run `python infer.py video_example.mp4`

## Download Our Segmentation Results on DAVIS 2016
* SegFlow without online training step (Ours_OL) [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours_OL.zip)
* SegFlow without optical flow branch (Ours_FLO) [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours_FLO.zip)
* Final SegFlow results [here](http://vllab1.ucmerced.edu/~ytsai/ICCV17/Ours.zip)

## Note
The model and code are available for non-commercial research purposes only.
* 09/2017: demo code released
