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

#*************************************************************************************
#------------------------------------------------------------------------------------*
# This code compute the performances for the method	using thermal and Fisheye cameras*
# It also computes the graphs of the evolution of the performances of the method	 *
# The arguments are detailed in ../utils/argument_parser.py							 *
#------------------------------------------------------------------------------------*
#*************************************************************************************

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


	# ---------------------------
	# Creating the saving folders
	# ---------------------------	

	if args.save is not None:
		print("[UP]: Creating the folders for saving the results")
		savepath = args.save + "/experiment_"
		experiment_counter = 0
		while os.path.isdir(savepath + str(experiment_counter)):
			experiment_counter += 1
		savepath = savepath + str(experiment_counter)
		os.mkdir(savepath)
		os.mkdir(savepath+"/networks")
		os.mkdir(savepath+"/student_outputs")
		os.mkdir(savepath+"/teacher_outputs")
		utils.graph.saveParameters(savepath + "_parameters.log", args)



	# --------------------------
	# Loading the Fisheye images
	# --------------------------

	# Definition of the data container
	fisheye_video = utils.data_loader.Video(args.fisheye, args.trainsetsize*args.teachersubsample + args.studentsubsample, device)

	# --------------------------
	# Loading the Thermal images
	# --------------------------

	#Definition of the data container
	thermal_video = utils.data_loader.Video(args.thermal, args.trainsetsize*args.teachersubsample + args.studentsubsample, device)

	

	# ----------------------------------
	# Loading the Thermal bounding boxes
	# ----------------------------------

	thermal_bounding_boxes, bounding_boxes_dictionary = utils.data_loader.load_bbox(args.thermalbbox, thermal_video.video_length)


	# ----------------------
	# Loading the next batch
	# ----------------------

	fisheye_images = fisheye_video.next_chunk(args.studentsubsample, to_torch=True)
	thermal_images = thermal_video.next_chunk(args.studentsubsample)

	# ----------------------------------
	# Loading the BGS masks if asked
	# ----------------------------------

	#Definition of the data container for the BGS
	bgs_video = None
	bgs_images = None
	# Load the dilated BGS 
	if args.bgs is not None:
		bgs_video = utils.data_loader.Video(args.bgs, args.trainsetsize*args.teachersubsample + args.studentsubsample, device)
		bgs_images = bgs_video.next_chunk(args.studentsubsample)

	# Load the undilated BGS
	bgs_video_s = None
	bgs_images_s = None
	if args.bgss is not None:
		bgs_video_s = utils.data_loader.Video(args.bgss, args.trainsetsize*args.teachersubsample + args.studentsubsample, device)
		bgs_images_s = bgs_video_s.next_chunk(args.studentsubsample)
	
	# -------------------------
	# Loading some useful masks
	# -------------------------
	
	# Loading the calibration
	calibration = utils.calibration.Calibration(thermal_images[0], fisheye_images[0])

	# Loading the calibration mask
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

	# |------------------------------|
	# |                              |
	# |Online training of the network|
	# |                              |
	# |------------------------------|

	# Definition of some variables
	network = None
	optimizer = None
	loss = None

	# Loop management variables
	index_start_trainset = 0
	index_stop_trainset = index_start_trainset + args.trainsetsize*args.teachersubsample
	index_start_testset = index_stop_trainset
	index_stop_testset = index_start_testset + args.studentsubsample

	# Loading of the network	
	print("[UP]: Choosing Yolo-v3 for the detection task with config: ", args.yoloconfig)
	network = networks.yolo.Darknet(args.yoloconfig).to(device)
	network.apply(networks.yolo_utils.weights_init_normal)
	optimizer = optim.Adam(network.parameters(), lr=args.learningrate)

	# If pre-trained weights are given, load them in the network
	if args.weights is not None:
		print("[UP]: Loading the pre-trained weights into the network")
		network.load_state_dict(torch.load(weightpath))


	# Load the data augmentation
	data_augmentation = utils.data_augmentation.DataAugmentation(savepath+"/data_augmentation/", device)

	# Iterate over the entire video
	while True:

		# -------------
		# Training part
		# -------------

		network.train()
		print("[ITER]: Training on the chunk: (", index_start_trainset, ",", index_stop_trainset, "), with a subsampling of: ", args.teachersubsample)
		images = fisheye_images[0:args.trainsetsize*args.teachersubsample:args.teachersubsample]
		if args.bgs is not None:
			bgs_masks = bgs_images[0:args.trainsetsize*args.teachersubsample:args.teachersubsample]
			calibration_mask_device = calibration_mask.to(device).type(torch.float)

		if args.bgss is not None:
			bgs_masks_s = bgs_images_s[0:args.trainsetsize*args.teachersubsample:args.teachersubsample]

		labels = None
		weights = None
		criterion = None
		
		# -------------------
		# Camera registration
		# -------------------

		# Transformation of the bounding boxes from thermal to fisheye via camera registration
		labels = list()
		labels_thermal = thermal_bounding_boxes[index_start_trainset:index_stop_trainset:args.teachersubsample]
		for t_box in tqdm(labels_thermal):
			f_box = calibration.thermaltofisheye(t_box)
			labels.append(f_box)
		

		# -------------------------------
		# Iterate over the online dataset
		# -------------------------------

		pbar_train = tqdm(total = len(images))
		# Performing the training for one epoch on the online dataset
		for i, (tensor, label) in enumerate(zip(images, labels)):
			# Getting the data on the GPU
			tensor = tensor.to(device)
			label = label.to(device)

			no_loss_mask = None
			if args.bgs is not None:
				bgs_mask_device = 1-(torch.from_numpy(bgs_masks[i][:,:,0]).to(device)/255).type(torch.float)

				# Mask where the loss needs to be zeroed out
				no_loss_mask = torch.where(bgs_mask_device > 0.0, bgs_mask_device, calibration_mask_device)

				# -----------------
				# Data augmentation
				# -----------------

				# Perform the data augmentation
				data_augmentation.frame_number = index_start_trainset + i*args.teachersubsample
				data_augmentation.trainset_start_index = index_start_trainset
				tensor, label, no_loss_mask = data_augmentation.augment(tensor, label, no_loss_mask=no_loss_mask, bgs_mask=bgs_masks_s[i][:,:,0])

				no_loss_mask.unsqueeze_(0)

			# Forward pass of the network
			loss, output = network.forward(tensor, label, no_loss_mask)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# Getting the data back to the cpu
			tensor = tensor.to("cpu")
			label = label.to("cpu")
			output = output.to("cpu")


			pbar_train.update(1)
		pbar_train.close()

		# [DEBUG] Saving the weights of the network
		network_weights = network.state_dict()
		savepath_weights = savepath + "/networks/weights.pt"
		torch.save(network_weights, savepath_weights)



		# -------------
		# Testing part
		# -------------

		network.eval()

		print("[ITER]: Testing on the chunk: (", index_start_testset, ",", index_stop_testset, ")")
		images = fisheye_images[args.trainsetsize*args.teachersubsample:]

		# Compute all predictions of the network
		predictions = list()
		for tensor in images:
			tensor = tensor.to(device)
			output = network.forward(tensor)
			tensor = tensor.to("cpu")
			output = output.to("cpu")
			predictions.append(output)
			# Empty the cuda cache
			torch.cuda.empty_cache()

		# Saving the detections in the JSON file
		for tensor, index in zip(predictions, np.arange(len(predictions))):

			# Apply Non-Maxima Suppression
			NMS_tensor = networks.yolo_utils.non_max_suppression(tensor, args.NMSconfidencethreshold, args.NMSthreshold)[0]

			if NMS_tensor is not None:
				NMS_tensor_normalized = NMS_tensor.clone()
				NMS_tensor_normalized[:,0] = ((NMS_tensor[:,0]+NMS_tensor[:,2])/2)/1280
				NMS_tensor_normalized[:,1] = ((NMS_tensor[:,1]+NMS_tensor[:,3])/2)/1280
				NMS_tensor_normalized[:,2] = (NMS_tensor[:,2]-NMS_tensor[:,0])/1280
				NMS_tensor_normalized[:,3] = (NMS_tensor[:,3]-NMS_tensor[:,1])/1280
				utils.evaluation.save_bounding_box(savepath+"/student_outputs/detections.json", NMS_tensor_normalized, index + index_start_testset)
	
		# Saving the teacher output if requested (for further analyses)
		if args.outteacher:
			
			# For the fisheye images
			# Iterating over all new images to save
			for i in np.arange(len(images)):

				# Create a copy of the image to print rectangles over it
				final = utils.preprocessing.torchToOpencv(utils.preprocessing.unnormalize(images[i][0].to("cpu")))

				# Iterate over all bounding boxes and draw rectangles on the original image
				bbox = calibration.thermaltofisheye(thermal_bounding_boxes[index_start_testset + i])
				for j in np.arange(bbox.size(0)):

					# Get the bounding box position in termes of frame indexes
					x1 = int(np.floor((bbox[j][2]-(bbox[j][4]/2))*final.shape[1]))
					y1 = int(np.floor((bbox[j][3]-(bbox[j][5]/2))*final.shape[0]))
					x2 = int(np.floor((bbox[j][2]+(bbox[j][4]/2))*final.shape[1]))
					y2 = int(np.floor((bbox[j][3]+(bbox[j][5]/2))*final.shape[0]))

					# Draw the rectangle
					final = cv2.rectangle(final, (x1,y1), (x2,y2), (0,0,255),2)

				# Save the image
				cv2.imwrite(savepath+"/teacher_outputs/detection_fisheye_"+str(index_start_testset + i)+".png", final)

			# For the thermal images
			images_thermal = thermal_images[args.trainsetsize*args.teachersubsample:]

			# If requested output the teacher bounding boxes

			#Iterating over all thermal images
			for i in tqdm(np.arange(len(images_thermal))):

				# Create a copy of the image to print rectangles over it
				final = np.copy(images_thermal[i])

				# Iterate over all bounding boxes and draw rectangles on the original image
				bbox = bounding_boxes_dictionary[str(index_start_testset + i)]
				for j in np.arange(len(bbox["confidence"])):

					# Get the bounding box position in termes of frame indexes
					x1 = int(np.floor((bbox["x"][j]-(bbox["width"][j]/2))*final.shape[1]))
					y1 = int(np.floor((bbox["y"][j]-(bbox["height"][j]/2))*final.shape[0]))
					x2 = int(np.floor((bbox["x"][j]+(bbox["width"][j]/2))*final.shape[1]))
					y2 = int(np.floor((bbox["y"][j]+(bbox["height"][j]/2))*final.shape[0]))

					# Draw the rectangle
					final = cv2.rectangle(final, (x1,y1), (x2,y2), (0,0,255),2)

				# Save the image
				cv2.imwrite(savepath+"/teacher_outputs/detection_thermal_"+str(index_start_testset + i)+".png", final)

		# ---------------
		# Loop management
		# ---------------

		# Set up the indexes for the next iteration
		index_start_trainset += args.studentsubsample
		index_stop_trainset = index_start_trainset + args.trainsetsize*args.teachersubsample
		index_start_testset = index_stop_trainset
		index_stop_testset = index_start_testset + args.studentsubsample

		# Empty the cuda cache
		#print("Memory used: ", torch.cuda.memory_allocated(device)/10**9)
		torch.cuda.empty_cache()

		# Stop condition and exit the program if condition is met
		if index_stop_testset >= fisheye_video.video_length:	
			log = open(savepath+"/student_outputs/detections.json", 'a')
			log.write("\n}")
			log.close()
			print("[END]: All the video has been treated, ending the program.")
			sys.exit(0)

		# Get the next images
		fisheye_images = fisheye_video.next_chunk(args.studentsubsample, to_torch=True)
		thermal_images = thermal_video.next_chunk(args.studentsubsample)
		if args.bgs is not None:
			bgs_images = bgs_video.next_chunk(args.studentsubsample)
		if args.bgss is not None:
			bgs_images_s = bgs_video_s.next_chunk(args.studentsubsample)