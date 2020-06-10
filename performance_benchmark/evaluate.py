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


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from tqdm import tqdm
import torch

import utils.argument_parser

import utils.compute_metrics


if __name__ == "__main__":

	# ------------------------
	# Parsing of the arguments
	# ------------------------

	from utils.argument_parser import args


	# ----------------------------
	# Definition of some variables
	# ----------------------------	
	
	path_groundtruth = args.groundtruthbbox
	path_detections = args.fisheyebbox
	save_folder = args.save
	calibration_mask = cv2.imread(args.calibration_mask_inner)[...,0]

	width = 1280
	height = 1280
	shift = 3588

	iou_thresh = 0.25
	fontsize = 25
	linewidth = 4

	# ----------------------
	# Loading the detections
	# ----------------------
	
	file_gt = open(path_groundtruth)
	file_detection = open(path_detections)

	groundtruth = json.load(file_gt)
	detections = json.load(file_detection)

	file_gt.close()
	file_detection.close()


	performances = utils.compute_metrics.get_AP(detections, groundtruth, calibration_mask, iou_thresh, window_size=18, shift=shift)

	# Load the curves
	y = performances[1][:,2]
	y_inside = performances[2][:,2]
	y_outside = performances[3][:,2]
	x = performances[0]

	max_value = max(np.max(y),np.max(y_inside),np.max(y_outside))
	y_ticks = np.arange(int(max_value*10)+1)*10

	# Plot the graph
	plt.figure(figsize=(16,12))
	ax_1 = plt.subplot(111)
	ax_1.spines["top"].set_visible(False)
	ax_1.spines["right"].set_visible(False)
	ax_1.get_xaxis().tick_bottom()
	ax_1.get_yaxis().tick_left()

	for y_tick in y_ticks:
	    plt.plot(x, np.zeros((len(x,)))+y_tick, "--", color="tab:grey", alpha = 0.3)
	    
	ax_1.plot(x, y_inside*100, linewidth=linewidth, color="tab:grey", alpha=0.95, label='Inside region')
	ax_1.plot(x, y_outside*100, linewidth=linewidth, color="tab:orange", alpha=0.95, label='Outside region')
	ax_1.plot(x, y*100, linewidth=linewidth, color="tab:blue", alpha=0.95, label='Whole frame')


	ax_1.set_xlabel("Video time (in minutes)", fontsize=fontsize)
	ax_1.set_ylabel("AP (%)", fontsize=fontsize, alpha=0.95)
	plt.yticks(y_ticks, fontsize=fontsize)
	plt.xticks([5,10,15,20,25,30,35,40,45,50,55], fontsize=fontsize)
	#ax_1.legend(fontsize=fontsize)
	plt.savefig(save_folder + "/performance_graph.png", bbox_inches='tight')
	plt.close()