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

import torch
import numpy as np
import json
import os

# Function for saving the bounding boxes found in a json file
def save_bounding_box(filename, bbox, frame_index):

	def write_all(filename, bbox, frame_index):

		log = open(filename, 'a')
		log.write("\""+str(frame_index)+"\":{")
		write_line(log, "confidence", bbox[:,4])
		log.write(",\n")
		write_line(log, "x", bbox[:,0])
		log.write(",\n")
		write_line(log, "y", bbox[:,1])
		log.write(",\n")
		write_line(log, "width", bbox[:,2])
		log.write(",\n")
		write_line(log, "height", bbox[:,3])
		log.write("}\n")
		log.close()

	def write_line(log, name, values):

		numbers = values.numpy()

		log.write("\n\"" + name + "\":[")
		for i in np.arange(len(numbers)-1):
			log.write(str(numbers[i])+",")
		log.write(str(numbers[-1])+"]")

	if not os.path.isfile(filename):
		log = open(filename, 'w')
		log.write("{\n")
		log.close()
		write_all(filename, bbox, frame_index)
	else:
		log = open(filename, 'a')
		log.write(",\n")
		log.close()
		write_all(filename, bbox, frame_index)

def get_frame_perf(detections, groundtruth, shift=3588, iou_thresh=0.5, region=None):
    all_frames = list()
    all_boxes = list()

    indexes = ((np.arange((42960-shift+120)/120)*120)+shift).astype("int")


    for index in indexes:

        if int(index) > 42960:
            continue

        bbox = detections[str(index)]
        gt = groundtruth[str(index-shift)]

        n_pred = len(bbox["x"])
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