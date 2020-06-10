"""
----------------------------------------------------------------------------------------
This file is part of the Multimodal Multiview Distillation distribution
Copyright (c) 2020 - see AUTHORS file

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------------------------
"""

import argparse

parser = argparse.ArgumentParser()

# -------------------------------
# Arguments concerning the inputs
# -------------------------------
parser.add_argument('--fisheye', '-f', help="path to the fisheye video", type=str, default="../data/fisheye.avi")
parser.add_argument('--thermal', '-t', help="path to the thermal video", type=str, default="../data/thermal.avi")
parser.add_argument('--thermalbbox', '-tbb', help="path to the thermal bounding boxes", type=str, default="../data/thermal_bbox.json")
parser.add_argument('--groundtruthbbox', '-gtbb', help="path to the groundtruth bounding boxes", type=str, default="../data/groundtruth.json")
parser.add_argument('--fisheyebbox', '-fbb', help="path to the fisheye bounding boxes", type=str)
parser.add_argument('--bgs', '-b', help="path to the dilated bgs video", type=str, default="../data/vibe_dilated/ViBeFisheyeDilated50/%d.png")
parser.add_argument('--bgss', '-bs', help="path to the undilated bgs video", type=str, default="../data/vibe_undilated/ViBeFisheyeDilated/%d.png")
parser.add_argument('--calibration_mask', '-calmask', help="path to the calibration mask", type=str, default="../data/calibration_mask.png")
parser.add_argument('--calibration_mask_inner', '-calmaskin', help="path to the inner calibration mask for the region-based evaluation", type=str, default="../data/calibration_mask_inner.png")
parser.add_argument('--field_mask', '-fiemask', help="path to the field mask", type=str, default="../data/field_mask.png")

# --------------------------------
# Arguments concerning the outputs
# --------------------------------
parser.add_argument('--save', '-s', help="path to save the results", type=str, default="../output")
parser.add_argument('--outteacher', '-ot', help="output the teacher predictions", type=int, default=None)

# ---------------------------------------------
# Arguments concerning the network architecture
# ---------------------------------------------
parser.add_argument('--yoloconfig', '-yc', help="path to the yolo configuration", type=str, default="./networks/config/yolov3-tiny-channels_div_by_4-1class-9anchors.cfg")
parser.add_argument('--weights', '-w', help="path to pre-trained weights", type=str, default = None)
parser.add_argument('--device', '-dev', help="device to use", type=str, default="cuda:0")

# ---------------------------------
# Arguments concerning the training
# ---------------------------------
parser.add_argument('--learningrate', '-lr', help="learning rate", type=float, default=0.001)
parser.add_argument('--trainsetsize', '-trs', help="training set size", type=int, default=67)
parser.add_argument('--studentsubsample', '-ssub', help="student subsample rate", type=int, default=72) # 4 seconds
parser.add_argument('--teachersubsample', '-tsub', help="teacher subsample rate", type=int, default=36) # 3 seconds
parser.add_argument('--NMSconfidencethreshold', '-nct', help="NMS confidence threshold", type=float, default=0.5) 
parser.add_argument('--NMSthreshold', '-nt', help="NMS threshold", type=float, default=0.4)



args = parser.parse_args()