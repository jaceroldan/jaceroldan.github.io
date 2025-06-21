---
layout: post
title:  "A YOLO Segmentation Model for Grocery Items!"
---

# A YOLO Segmentation Model for Grocery Items!

An experiment to train a model to detect Filipino grocery items using YOLO.

| Run | BoxP  | R     | mAP   | Remarks                                                                                                                             |
|-----|-------|-------|-------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1   | 0.910 | 0.800 | 0.879 | Default YOLO-V8 nano, 10 epochs                                                                                                     |
| 2   | 0.946 | 0.901 | 0.922 | Run 1 + Default YOLO-V11 nano                                                                                                       |
| 3   | 0.939 | 0.885 | 0.923 | Run 1 + mosaic=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.373, translate=0.45, scale=0.5, shear=0.3, flipud=0.01, fliplr=0.5 |                                                                      |
| 4   | 0.947 | 0.921 | 0.941 | Run3 with YOLO-V8 medium                                                                                                            |
| 5   | 0.955 | 0.912 | 0.940 | Run3 with YOLO-V11 medium                                                                                                           |
| 6   | 0.951 | 0.919 | 0.940 | Run3 with YOLO-V11 X                                                                                                                |
| 7   | 0.951 | 0.918 | 0.938 | Run6 with CosLR 0.01 to 0.001                                                                                                       |
| 8   | 0.964 | 0.933 | 0.948 | Run6 with 20 epochs                                                                                                                 |
| 9   | 0.964 | 0.958 | 0.974 | Run6 with 30 epochs                                                                                                                 |

# Training runs

Training runs and weights can be found in this [Google Drive](https://drive.google.com/drive/folders/1_awr49-evoKf2umHpMZX-DZkjJaTCUfU?usp=sharing) (Request for access from owner). 

# Sample inference

![YOLO-Examples](/assets/yolo-assets-examples.png)

For a sample video, you may look at this [Google Drive link](https://drive.google.com/file/d/11ZkKFuzX7ROK4D8aibv1qCen53INYlT0/view?usp=sharing).
