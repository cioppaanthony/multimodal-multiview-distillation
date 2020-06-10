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


import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import networks.yolo_utils
from tqdm import tqdm
import torch

height, width = 1280, 1280

def get_frame_AP(detections, groundtruth, calibration_mask, iou_thresh=0.5, shift=3588):
	all_frames = list()
	all_boxes = list()

	indexes = ((np.arange((42960-shift+120)/120)*120)+shift).astype("int")


	for index in indexes:

		if index > 42960:
			continue

		bbox = None
		n_pred = 0
		if str(index) in detections:
			bbox = detections[str(index)]
			n_pred = len(bbox["x"])
		gt = groundtruth[str(index-shift)]

		
		n_gt = len(gt["x"])

		pred_bb = torch.zeros(1,n_pred, 7)

		gt_bb = torch.zeros(n_gt, 6)

		for i in np.arange(n_pred):

			pred_bb[0, i, 0] = int((bbox["x"][i]-bbox["width"][i]/2)*width)
			pred_bb[0, i, 1] = int((bbox["y"][i]-bbox["height"][i]/2)*height)
			pred_bb[0, i, 2] = int((bbox["x"][i]+bbox["width"][i]/2)*width)
			pred_bb[0, i, 3] = int((bbox["y"][i]+bbox["height"][i]/2)*height)
			pred_bb[0, i, 4] = bbox["confidence"][i]
			pred_bb[0, i, 5] = 1
			pred_bb[0, i, 6] = 0

		for i in np.arange(n_gt):

			gt_bb[i, 0] = 0
			gt_bb[i, 1] = 0
			gt_bb[i, 2] = int((gt["x"][i])*width)
			gt_bb[i, 3] = int((gt["y"][i])*height)
			gt_bb[i, 4] = int((gt["x"][i]+gt["width"][i])*width)
			gt_bb[i, 5] = int((gt["y"][i]+gt["height"][i])*height)

		batch_metrics = networks.yolo_utils.get_batch_statistics(pred_bb, gt_bb, iou_thresh)

		all_frames.append(batch_metrics)
		all_boxes.append((index-shift, pred_bb, gt_bb))
		
	return all_frames, all_boxes, indexes

def get_frame_AP_region(detections, groundtruth, calibration_mask, iou_thresh=0.5, shift=3588):
	all_frames_inside = list()
	all_frames_outside = list()
	all_boxes_inside = list()
	all_boxes_outside = list()

	indexes = ((np.arange((42960-shift+120)/120)*120)+shift).astype("int")


	for index in indexes:

		if int(index) > 42960:
			continue

		bbox = None
		n_pred = 0
		if str(index) in detections:
			bbox = detections[str(index)]
			n_pred = len(bbox["x"])
			
		gt = groundtruth[str(index-shift)]

		n_gt = len(gt["x"])

		pred_bb_inside = list()
		pred_bb_outside = list()

		gt_bb_inside = list()
		gt_bb_outside = list()

		for i in np.arange(n_pred):

			x1 = int((bbox["x"][i]-bbox["width"][i]/2)*width)
			y1 = int((bbox["y"][i]-bbox["height"][i]/2)*height)
			x2 = int((bbox["x"][i]+bbox["width"][i]/2)*width)
			y2 = int((bbox["y"][i]+bbox["height"][i]/2)*height)
			conf = bbox["confidence"][i]
			
			tmp = torch.Tensor(1,1,7)
			tmp[0,0,0] = x1
			tmp[0,0,1] = y1
			tmp[0,0,2] = x2
			tmp[0,0,3] = y2
			tmp[0,0,4] = conf
			tmp[0,0,5] = 1
			tmp[0,0,6] = 0
			
			if calibration_mask[int((y1+y2)/2), int((x1+x2)/2)] == 0:
				pred_bb_outside.append(tmp)
			else:
				pred_bb_inside.append(tmp)
				   

		for i in np.arange(n_gt):
			x1 = int((gt["x"][i])*width)
			y1 = int((gt["y"][i])*height)
			x2 = int((gt["x"][i]+gt["width"][i])*width)
			y2 = int((gt["y"][i]+gt["height"][i])*height)
			
			tmp = torch.Tensor(1,6)
			tmp[0,0] = 0
			tmp[0,1] = 0
			tmp[0,2] = x1
			tmp[0,3] = y1
			tmp[0,4] = x2
			tmp[0,5] = y2
			if calibration_mask[int((y1+y2)/2), int((x1+x2)/2)] == 0:
				gt_bb_outside.append(tmp)
			else:
				gt_bb_inside.append(tmp)
				
		pred_bb_inside_tensor = torch.zeros(1, len(pred_bb_inside),7)
		for i in np.arange(len(pred_bb_inside)):
			pred_bb_inside_tensor[0,i] = pred_bb_inside[i][0,0]
			
		pred_bb_outside_tensor = torch.zeros(1, len(pred_bb_outside),7)
		for i in np.arange(len(pred_bb_outside)):
			pred_bb_outside_tensor[0,i] = pred_bb_outside[i][0,0]
			
		gt_bb_inside_tensor = torch.zeros(len(gt_bb_inside),6)
		for i in np.arange(len(gt_bb_inside)):
			gt_bb_inside_tensor[i] = gt_bb_inside[i][0]
			
		gt_bb_outside_tensor = torch.zeros(len(gt_bb_outside),6)
		for i in np.arange(len(gt_bb_outside)):
			gt_bb_outside_tensor[i] = gt_bb_outside[i][0]

		batch_metrics_inside = networks.yolo_utils.get_batch_statistics(pred_bb_inside_tensor, gt_bb_inside_tensor, iou_thresh)
		
		batch_metrics_outside = networks.yolo_utils.get_batch_statistics(pred_bb_outside_tensor, gt_bb_outside_tensor, iou_thresh)
  
		all_frames_inside.append(batch_metrics_inside)
		all_frames_outside.append(batch_metrics_outside)
		
		all_boxes_inside.append((index-shift, pred_bb_inside_tensor, gt_bb_inside_tensor))
		all_boxes_outside.append((index-shift, pred_bb_outside_tensor, gt_bb_outside_tensor))
		
	return all_frames_inside, all_frames_outside, all_boxes_inside, all_boxes_outside, indexes


def get_AP(detections, groundtruth, calibration_mask, iou_thresh=0.25, window_size=20, shift=3588):

	all_frames, _, _ = get_frame_AP(detections, groundtruth, calibration_mask, iou_thresh, shift=3588)
	all_frames_inside, all_frames_outside, _, _, _ = get_frame_AP_region(detections, groundtruth, calibration_mask, iou_thresh, shift=3588)

	start = 0
	stop = window_size
	total = len(all_frames)


	y = list()
	y_inside = list()
	y_outside = list()

	while stop <= total:

		add_list = list()
		add_list_inside = list()
		add_list_outside = list()

		for i in np.arange(stop-start)+start:
			add_list += all_frames[i]
			add_list_inside += all_frames_inside[i]
			add_list_outside += all_frames_outside[i]

		true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*add_list))]
		m  = networks.yolo_utils.ap_per_class(true_positives, pred_scores, pred_labels, pred_labels)
		v = m[2]
		y.append(m)
		
		true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*add_list_inside))]
		m  = networks.yolo_utils.ap_per_class(true_positives, pred_scores, pred_labels, pred_labels)
		v = m[2]
		if len(v) == 0:
			m = (0,0,0,0,0)
		y_inside.append(m)
		
		true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*add_list_outside))]
		m  = networks.yolo_utils.ap_per_class(true_positives, pred_scores, pred_labels, pred_labels)
		#precision, recall, AP, f1, ap_class  = networks.yolo_utils.ap_per_class(true_positives, pred_scores, pred_labels, pred_labels)
		v = m[2]
		if len(v) == 0:
			m = (0,0,0,0,0)
		y_outside.append(m)

		start += 1
		stop += 1


	x = np.array(np.arange(len(y)))
	x = (x*120+shift)/(12*60)
	y = np.array(y)
	y_inside = np.array(y_inside)
	y_outside = np.array(y_outside)
		
	return x, y, y_inside, y_outside