---
layout: post
title:  "A YOLO Segmentation Model for Grocery Items!"
---

# A YOLO Segmentation Model for Grocery Items!

An experiment to train a model to detect Filipino grocery items using YOLO. YOLO v11 model (X-size) is trained in this experiment for object detection and segmentation tasks for common Filipino grocery items under a variety of model configurations and data augmentation techniques. An inference application is deployed via tunneling on a Gradio interface.

# Prompt

We were given a task during our AI 212 course to 

# Methodology and Rationale of Proposed Solution

## Part 1

This was composed of Run 1 to 14 configurations.

## Part 2
* Used last model parameters and hyperparameters. 
(mosaic=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.373, translate=0.45, scale=0.5, shear=0.3, flipud=0.01, fliplr=0.5)
* Trained for 50 more epochs (previously trained for only 50). 
* Varied HSV (Hue, Saturation, and Brightness) parameters



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
| 9   | 0.964 | 0.933 | 0.948 | Run6 with 30 epochs                                                                                                                 |
| 10   | 0.964 | 0.958 | 0.974 | Run9 with Mixup variation on 30 epochs                                                                                             |
| 11   | 0.964 | 0.958 | 0.974 | Run9 with Mixup variation on 50 epochs                                                                                             |
| 12   | 0.964 | 0.958 | 0.974 | Made a dataset improvement and reran using Trial 11 settings on 30 epochs                                                          |
| 13   | 0.964 | 0.958 | 0.974 | Trial 12 but on 50 epochs                                                                                                          |
| 14   | 0.964 | 0.958 | 0.974 | Made further dataset improvments and reran using Trial 11 on 30 epochs                                                             |
| 15   | 0.823 | 0.958 | 0.974 | Trial 14 with HSV_h 0.25                                                                                                           |
| 16   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_h 0.35                                                                                                           |
| 17   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_h 0.45                                                                                                           |
| 18   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_s 0.6                                                                                                            |
| 19   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_s 0.5                                                                                                            |
| 20   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_s 0.4                                                                                                            |
| 21   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_v 0.5                                                                                                            |
| 22   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_v 0.6                                                                                                            |
| 23   | 0.964 | 0.958 | 0.974 | Trial 14 with HSV_v 0.7                                                                                                            |

# Training runs

Training runs and weights can be found in this [Google Drive](https://drive.google.com/drive/folders/1_awr49-evoKf2umHpMZX-DZkjJaTCUfU?usp=sharing) (Request for access from owner). 

# Sample inference

![YOLO-Examples](/assets/yolo-assets-examples.png)

For a sample video, you may look at this [Google Drive link](https://drive.google.com/file/d/11ZkKFuzX7ROK4D8aibv1qCen53INYlT0/view?usp=sharing).
