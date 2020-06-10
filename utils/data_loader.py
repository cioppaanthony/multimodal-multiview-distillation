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
import cv2
import utils.preprocessing
import utils.argument_parser
import json
from tqdm import tqdm 
# ---------------------------------------------------------------------------
# Video class for loading the videos, the segmentation and the bounding boxes
# Allows to load chunks of the video which are overlapping efficiently
# ---------------------------------------------------------------------------

class Video:

	def __init__(self, video_path, chunk_size, device):

		# Loading the video and checking if it is opened
		self.video = cv2.VideoCapture(video_path)
		if not self.video.isOpened():
			print("[ERR]: Could not retrieve the following video: ", video_path)
			sys.exit()
		self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

		# Definition of some indexes
		#self.pos_start_trainset = 0
		#self.pos_stop_trainset = 0
		#self.pos_start_testset = 0
		#self.pos_stop_testset = 0

		# Definition of the chunk containers
		self.video_chunk = list()
		#self.segmentation_chunk = list()
		#self.bounding_box_chunk = list()

		# Definition of some variables
		#self.trainset_size = trainset_size
		#self.testset_size = testset_size
		#self.teacher_subsample = teacher_subsample
		self.chunksize = chunk_size
		self.device = device


	# load the next chunk but keeps the already loaded frames instead of re-loading them
	# [TBD] segmentation and bounding boxes
	def next_chunk(self, step, to_torch=False, segmentation=False, bounding_box_chunk=False):

		# Pop as many images as needed at the beginning of the list
		self.video_chunk = self.video_chunk[step:]

		# Fill the list back with the next frames
		while len(self.video_chunk) < self.chunksize:

			# Read the next frame
			ret, frame = self.video.read()
			# If the frame needs to be transformed to the pytorch format
			if to_torch:
				frame = utils.preprocessing.normalize(utils.preprocessing.opencvToTorch(frame, self.device)).to("cpu")
			self.video_chunk.append(frame)

			# Exit if last frame
			if not ret:
				print("[END]: Video finished, exiting program")
				self.video.release()
				sys.exit()

		return self.video_chunk


def load_bbox(json_file, video_length):

	thermal_bounding_boxes = list()

	file_bounding_boxes = open(json_file)
	bounding_boxes_dictionary = json.load(file_bounding_boxes)

	# Arranging the bounding boxes into the Yolo-like annotation format
	for i in tqdm(np.arange(video_length)):

		bbox = bounding_boxes_dictionary[str(i)]
		# Create the annotation tensor holding the results
		annotation = torch.Tensor(len(bbox["confidence"]), 6)
		# Insert the batch image index (always 0, batch of size 1 in our case)
		annotation[:,0] = 0
		# Insert the class (always 0, player, in our case)
		annotation[:,1] = 0
		# Insert the x coordinate of the center of the bounding box
		annotation[:,2] = torch.FloatTensor(bbox["x"])
		# Insert the y coordinate of the center of the bounding box
		annotation[:,3] = torch.FloatTensor(bbox["y"])
		# Insert the width of the bounding box
		annotation[:,4] = torch.FloatTensor(bbox["width"])
		# Insert the height of the bounding box
		annotation[:,5] = torch.FloatTensor(bbox["height"])

		# Append the tensor to the list
		thermal_bounding_boxes.append(annotation)

	return thermal_bounding_boxes, bounding_boxes_dictionary

def load_bbox_with_confidence(json_file, video_length, start_number=0):

	thermal_bounding_boxes = list()

	file_bounding_boxes = open(json_file)
	bounding_boxes_dictionary = json.load(file_bounding_boxes)

	# Arranging the bounding boxes into the Yolo-like annotation format
	for i in tqdm(np.arange(video_length-start_number)+start_number):

		if str(i) in bounding_boxes_dictionary:
			bbox = bounding_boxes_dictionary[str(i)]
			# Create the annotation tensor holding the results
			annotation = torch.Tensor(len(bbox["confidence"]), 6)
			# Insert the confidence image index (always 0, batch of size 1 in our case)
			annotation[:,0] = torch.FloatTensor(bbox["confidence"])
			# Insert the class (always 0, player, in our case)
			annotation[:,1] = 0
			# Insert the x coordinate of the center of the bounding box
			annotation[:,2] = torch.FloatTensor(bbox["x"])
			# Insert the y coordinate of the center of the bounding box
			annotation[:,3] = torch.FloatTensor(bbox["y"])
			# Insert the width of the bounding box
			annotation[:,4] = torch.FloatTensor(bbox["width"])
			# Insert the height of the bounding box
			annotation[:,5] = torch.FloatTensor(bbox["height"])

			# Append the tensor to the list
			thermal_bounding_boxes.append(annotation)
		else:
			annotation = torch.Tensor(0, 6)
			thermal_bounding_boxes.append(annotation)

	return thermal_bounding_boxes, bounding_boxes_dictionary