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

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
import utils.preprocessing
import utils.argument_parser
import utils.graph
import networks.student
import networks.yolo
import networks.yolo_utils
import json
import torch.optim as optim
import utils.evaluation
import utils.calibration
import utils.data_loader
import utils.data_augmentation

# Function finish the json file if it has not 
# been finished correctly during the main program
def add_end_of_json(filepath):

	file = open(filepath, 'r')
	rows = []
	lines = file.readlines()
	for line in lines:
		rows.append(line)
	file.close()
	if rows[-1] != "}":
		print("adding last row")
		file = open("./tmp.json", 'w')
		for line in lines:
			file.write(line)
		file.write("\n}")
		file.close()
		return "./tmp.json"
	return filepath

if __name__ == "__main__":

	# ------------------------
	# Parsing of the arguments
	# ------------------------

	from utils.argument_parser import args

	# ----------------------------
	# Definition of some variables
	# ----------------------------	

	device = torch.device(args.device)
	num_classes = 1 #(person)


	# --------------------------
	# Loading the Fisheye images
	# --------------------------
	# Definition of the data container
	fisheye_video = cv2.VideoCapture(args.fisheye)

	# --------------------------
	# Loading the Fisheye images
	# --------------------------
	# Definition of the data container
	fisheye_video_nu = utils.data_loader.Video(args.fisheye, 2, device)

	# --------------------------
	# Loading the Thermal images
	# --------------------------

	#Definition of the data container
	thermal_video_nu = utils.data_loader.Video(args.thermal, 2, device)



	fisheye_images = fisheye_video_nu.next_chunk(args.studentsubsample, to_torch=True)
	thermal_images = thermal_video_nu.next_chunk(args.studentsubsample)

	# ----------------------------------
	# Loading the Thermal bounding boxes
	# ----------------------------------

	start_number = 2412
	filepath = add_end_of_json(args.fisheyebbox)
	print("Filepath: ", filepath)
	#fisheye_bounding_boxes, bounding_boxes_dictionary = utils.data_loader.load_bbox(args.fisheyebbox, thermal_video_nu.video_length, start_number)
	fisheye_bounding_boxes, bounding_boxes_dictionary = utils.data_loader.load_bbox_with_confidence(filepath, 43020, start_number)


	# ----------------------------------
	# Loading the BGS masks if asked
	# ----------------------------------
	#Definition of the data container
	bgs_video = None

	if args.bgs is not None:
		bgs_video = cv2.VideoCapture(args.bgs)

	

	# -------------------------
	# Loading some useful masks
	# -------------------------

	# Loading the calibration
	calibration = utils.calibration.Calibration(thermal_images[0], fisheye_images[0])
	calibration_mask = None
	# From a pre-computed mask (fast)
	if args.calibration_mask is not None:
		calibration_mask = cv2.imread(args.calibration_mask)[:,:,0]
		calibration_mask = torch.from_numpy(calibration_mask).type(torch.float)/255
	# Computing the calibration mask (slow)
	else:
		calibration_mask = calibration.calibrationMask(fisheye_images[0], thermal_images[0])

	# Loading the field mask (precomputed)
	field_mask = None
	if args.field_mask is not None:
		field_mask = cv2.imread(args.field_mask)[:,:,0]
		field_mask_device = (torch.from_numpy(field_mask).type(torch.float)/255).to(device)
	

	print("[UP]: All of the data is loaded")
	print("[UP]: The online training of the network can start")


	counter = 0


	fisheye_video.set(cv2.CAP_PROP_POS_FRAMES, start_number+counter)
	bgs_video.set(cv2.CAP_PROP_POS_FRAMES, start_number+counter)

	ret, image_numpy = fisheye_video.read()
	ret_bgs, bgs_numpy = bgs_video.read()

	# Video writer for storing the output frames with the bounding boxes
	out = cv2.VideoWriter(args.save + "/student_output.mp4",cv2.VideoWriter_fourcc(*'MP4V'), 12, (image_numpy.shape[1],image_numpy.shape[0]))

	# Load the data augmentation
	data_augmentation = utils.data_augmentation.DataAugmentation(None, device)

	mask_indexes = np.where(data_augmentation.calibration_mask_inner == 1)
	mask_field_index = np.where(data_augmentation.field_mask==1)

	counting = []

	# ---------------------------------------------
	# Perform the bgs post-processing on all images
	# ---------------------------------------------
	for i in tqdm(np.arange(len(fisheye_bounding_boxes))):
		
		# Transformation of the bounding boxes
		label = fisheye_bounding_boxes[counter+i]

		#print(label)

		counting.append(0)

		no_loss_mask = None
		if args.bgs is not None:

			final = image_numpy.copy()

			final[mask_indexes] = final[mask_indexes]*0.75

			all_boxes = list()


			for bbox in label:
				box = bbox.to("cpu").numpy()
				center_x = int(box[2]*1280)
				center_y = int(box[3]*1280)
				width = int(np.ceil(box[4]*1280))
				height = int(np.ceil(box[5]*1280))
				x = int(np.floor(center_x - width/2))
				y = int(np.floor(center_y - height/2))
				confidence = box[0]


				if bgs_numpy[center_y, center_x,0] == 0:
					continue

				all_boxes.append(np.array([box[2], box[3], box[4], box[5], box[0]]))

				counting[-1] += 1

				final = cv2.rectangle(final, (x,y), (x+width, y+height), (0,0,int(255*confidence)), 2)

			out.write(final)

			if len(all_boxes) > 0:
				utils.evaluation.save_bounding_box(args.save + "/detection_with_BGS.json", torch.from_numpy(np.array(all_boxes)), i + start_number)


			#sys.exit()
		
		
		ret, image_numpy = fisheye_video.read()
		ret_bgs, bgs_numpy = bgs_video.read()

	log = open(args.save + "/detection_with_BGS.json", 'a')
	log.write("\n}")
	log.close()

	np.save(args.save + ".npy", np.array(counting))
	out.release()

