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

import cv2
import numpy as np
from utils.argument_parser import args
import utils.preprocessing
import time
import sys
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataAugmentation:

	def __init__(self, save_path=None, device=torch.device("cpu")):

		# Load the calibration mask in opencv format
		self.calibration_mask = cv2.imread(args.calibration_mask)[...,0]/255

		# Load the field mask in opencv format
		self.field_mask = cv2.imread(args.field_mask)[...,0]/255

		# Path to possibly save the results
		self.save_path = save_path

		# Device
		self.device = device

		# Set up some variables
		self.width = self.calibration_mask.shape[1]
		self.height = self.calibration_mask.shape[0]

		# Random variables selector
		self.random_contour_selector = 3

		# Border around the bounding boxes in number of pixels
		self.border = 5
		self.border_feather = np.linspace(0,1,self.border+2)[1:][::-1]

		# Extra border for the initial detections
		self.extra_border = 2

		# Border for the BGS verification
		self.bgs_border = 2

		# std for the histogram
		self.histogram_std = 8

		# Scaling for the transformation
		self.scaling_alpha = 0.5
		self.scaling_beta = -np.log(3)/280
		self.scaling_gamma = 0.5

		# Compute border of the calibration mask to remove false masks
		_ , self.theta_min = self.cartesian_to_polar_(963,220)
		_ , self.theta_max = self.cartesian_to_polar_(547,146)

		# Calibration mask border and inner mask
		self.calibration_mask_border = self.calibration_mask.copy()
		self.calibration_mask_border_()
		self.calibration_mask_inner = self.calibration_mask - self.calibration_mask_border

		# Compute the whole set of possible random anchor positions
		self.possible_positions_mask = (self.calibration_mask_inner*self.field_mask + self.field_mask) % 2
		self.possible_positions = np.where(self.possible_positions_mask > 0)
		self.r_max, _ = self.cartesian_to_polar_(269,387)
		self.r_min, _ = self.cartesian_to_polar_(642,521)
		self.possible_positions_list = list([list(), list()])
		self.possible_positions_()


		self.frame_number = 0
		self.trainset_start_index = 0



	# Load an image and bounding boxes and transforms them
	def augment(self, tensor, labels, no_loss_mask=None, bgs_mask=None):

		# Check if some labels exist otherwise perform no data augmentation
		if labels.size()[0] == 0:
			print("No labels")
			return tensor, labels, no_loss_mask


		# Format the input data into the correct format
		frame, bounding_boxes, no_loss_mask = self.format_data_(tensor, labels, no_loss_mask, bgs_mask)

		# Check if some bounding boxes still exist otherwise perform no data augmentation
		if len(bounding_boxes) == 0:
			#print("No more bounding boxes")
			return tensor, labels, no_loss_mask

		# Get the bounding box mask
		bounding_box_mask, bounding_box_mask_all = self.create_bounding_box_mask_(bounding_boxes)

		# Get the contours
		contours = self.select_contours_(bounding_box_mask_all)

		# Get the list of bounding boxes
		bounding_boxes_list, bounding_boxes_indexes = self.bbox_in_contours_(bounding_boxes, contours)

		# Get the tight bounding boxes for the selected groups
		tight_bounding_boxes_list = self.tight_bounding_boxes_(bounding_boxes_list)

		# Get the progressive masks
		progressive_masks_list = self.progressive_masks_(bounding_boxes_list, tight_bounding_boxes_list)

		# Get cropped templates
		cropped_thumbnails_list = self.crop_thumbnails_(frame, tight_bounding_boxes_list)

		# Perform the transformation
		new_frame, new_labels, no_loss_mask = self.random_transforms_(frame, bounding_boxes, cropped_thumbnails_list, progressive_masks_list ,tight_bounding_boxes_list, no_loss_mask)

		# Re-formart the data
		tensor, labels = self.unformat_data_(new_frame, new_labels)
		
		return tensor, labels, no_loss_mask
	
	#from the labels, draw the bounding boxes on a mask
	def segmentation_mask(self, labels):

		final = np.zeros((self.height, self.width))

		for label in labels:

			l = label.to("cpu").numpy()
			x1 = int((l[2] - l[4]/2)*self.width)
			y1 = int((l[3] - l[5]/2)*self.height)
			x2 = int((l[2] + l[4]/2)*self.width)
			y2 = int((l[3] + l[5]/2)*self.height)
			final = cv2.rectangle(final, (x1,y1), (x2,y2), 1, -1)

		target = torch.from_numpy(final).unsqueeze_(0).to(self.device).type(torch.float)
		percentage = torch.sum(target).to("cpu").numpy()/(self.width*self.height)
		weights = torch.Tensor(2) 
		if percentage == 0 or percentage == 1:
			weights[0]= 1
			weights[1]= 1
			return target, weights
		weights[0]= 1/percentage
		weights[1]= 1/(1-percentage)
		target = 1-target

		return target, weights

	# Format the input data into the correct format
	def format_data_(self, tensor, labels, no_loss_mask=None, bgs_mask=None):

		# Transform image to opencv format
		frame = utils.preprocessing.torchToOpencv(utils.preprocessing.unnormalize(tensor))

		# Transform the labels to numpy format in x,y,width, height
		bounding_boxes_norm = labels[:,2:].to("cpu").numpy()

		# Create a list to store the data
		bounding_boxes = list()

		for bbox in bounding_boxes_norm:

			# Create the new bounding box
			new_bbox = np.array([0,0,0,0])

			# Compute the coordinates
			new_bbox[0] = np.floor(((bbox[0]-bbox[2]/2)*self.width)-self.extra_border)
			new_bbox[1] = np.floor(((bbox[1]-bbox[3]/2)*self.height)-self.extra_border)
			new_bbox[2] = np.floor((bbox[2]*self.width) + 2*self.extra_border)
			new_bbox[3] = np.floor((bbox[3]*self.height) + 2*self.extra_border)

			if bgs_mask is not None:
				x_center = int(np.floor(bbox[0]*self.width))
				y_center = int(np.floor(bbox[1]*self.height))
				if np.mean(bgs_mask[y_center-self.bgs_border:y_center+self.bgs_border, x_center-self.bgs_border:x_center+self.bgs_border]) < 255:
					continue

			# Add it to the new bounding boxes only if it is comprised between the correct angles
			_ , theta = self.cartesian_to_polar_(new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2)

			if theta >= self.theta_min and theta <= self.theta_max:
				bounding_boxes.append(new_bbox)


		bounding_boxes = np.array(bounding_boxes)


		# Remove the border of the mask from the no loss mask
		if no_loss_mask is not None:
			no_loss_mask = no_loss_mask * (1-torch.from_numpy(self.calibration_mask_border).to(self.device).type(torch.float))

		return frame, bounding_boxes.astype("int"), no_loss_mask

	# Format the input data into the correct format
	def unformat_data_(self, frame, labels):

		# Transform the image to pytorch format
		tensor = utils.preprocessing.normalize(utils.preprocessing.opencvToTorch(frame, self.device))

		# Transform the labels in the correct format
		labels = torch.from_numpy(labels).to(self.device).type(torch.float)

		labels[:,0] = (labels[:,0] + labels[:,2]/2)/self.width
		labels[:,1] = (labels[:,1] + labels[:,3]/2)/self.height
		labels[:,2] = labels[:,2]/self.width
		labels[:,3] = labels[:,3]/self.height
		
		# To concatenate in the unformat function
		zero_tensor = torch.zeros(labels.size()[0],2).to(self.device).type(torch.float)

		labels = torch.cat((zero_tensor,labels),1)

		return tensor, labels

	# Create a bounding box mask with one channel per bounding box
	# Shape is (num_bbox, height, width) with value [0,1]
	# Also logical_or (height, width)
	def create_bounding_box_mask_(self, bounding_boxes):

		# Create the arrays
		bounding_box_mask = np.zeros((bounding_boxes.shape[0], self.height, self.width)).astype('uint8')
		bounding_box_mask_all = np.zeros((self.height, self.width)).astype('uint8')

		# For each bounding box, draw its mask
		for index in np.arange(bounding_boxes.shape[0]):

			# Get the bounding box
			bbox = bounding_boxes[index]

			# Draw a white rectangle around the bounding box
			cv2.rectangle(bounding_box_mask[index], (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 1,-1)

			# Logical or to have an aggregated mask
			bounding_box_mask_all = np.logical_or(bounding_box_mask_all, bounding_box_mask[index])

		bounding_box_mask_all = bounding_box_mask_all.astype("uint8")

		return bounding_box_mask, bounding_box_mask_all

	# Select groups of bounding boxes
	def select_contours_(self, bounding_box_mask_all):

		# Use the opencv function to get the contours
		contours, _ = cv2.findContours(bounding_box_mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


		# Select a subset of the contours
		number_of_selections = np.random.randint( len(contours), self.random_contour_selector*len(contours))+1
		random_indexes = np.random.randint(0,len(contours),size=number_of_selections)

		# Select only the interesting contours
		selected_indexes = list(random_indexes)
		selected_contours = [contours[i] for i in selected_indexes]


		return selected_contours

	# List all bounding boxes in contours
	def bbox_in_contours_(self, bounding_boxes, contours):

		bounding_boxes_list = list()
		bounding_boxes_indexes = list()

		# Check for each contour if each bounding box is comprised
		for contour in contours:

			bounding_boxes_list.append(list())
			bounding_boxes_indexes.append(list())

			for index, bbox in enumerate(bounding_boxes):

				# If the bounding box is in the contour
				if cv2.pointPolygonTest(contour, (int(bbox[0]+bbox[2]/2), int(bbox[1] + bbox[3]/2)), False) >= 0:
					bounding_boxes_list[-1].append(bbox)
					bounding_boxes_indexes[-1].append(index)

		return bounding_boxes_list, bounding_boxes_indexes

	# Get the tight bounding boxes around the group of bounding boxes
	# Returns list of tight bounding boxes in (x,y,width, height)
	def tight_bounding_boxes_(self, bounding_boxes_list):

		# Array to store the tight bounding boxes
		tight_bounding_boxes = list()

		for bounding_boxes in bounding_boxes_list:

			array_bbox = np.array(bounding_boxes)

			min_x = np.min(array_bbox[:,0]-self.border)
			min_y = np.min(array_bbox[:,1]-self.border)
			max_x = np.max(array_bbox[:,0]+array_bbox[:,2]+self.border+1)
			max_y = np.max(array_bbox[:,1]+array_bbox[:,3]+self.border+1)

			tight_bounding_boxes.append(np.array([min_x, min_y, max_x-min_x, max_y-min_y]))


		return tight_bounding_boxes

	# Compute the progressive masks of the form of list of (n_box, croped_height, croped_width)
	# With values between [0,1]
	def progressive_masks_(self, bounding_boxes_list, tight_bounding_boxes_list):

		# list to store the masks
		progressive_masks_list = list()

		# Iterate over all regions
		for bounding_boxes, tight_bbox in zip(bounding_boxes_list, tight_bounding_boxes_list):
			
			# Create a mask of the correct size
			mask = np.zeros((len(bounding_boxes), tight_bbox[-1], tight_bbox[-2]))

			# Draw each bounding box with its border in the correct mask
			for index, bbox in enumerate(bounding_boxes):

				# Start from the edge and go to the center
				tmp_border = self.border

				while tmp_border >= 0:

					x = bbox[0]-tight_bbox[0]-tmp_border
					y = bbox[1]-tight_bbox[1]-tmp_border
					width = bbox[2] + 2*tmp_border
					height = bbox[3] + 2*tmp_border
					cv2.rectangle(mask[index], (x,y), (x+width, x+height), self.border_feather[tmp_border], -1)
					tmp_border = tmp_border -1

			progressive_masks_list.append(mask)


		return progressive_masks_list

	# Get the cropped thumbnails corresponding to the tight bounding boxes
	def crop_thumbnails_(self, frame, tight_bounding_boxes_list):

		cropped_thumbnails_list = list()

		# Iterate over all regions
		for tight_bbox in tight_bounding_boxes_list:

			# Create the thumbnail
			thumbnail = np.copy(frame[tight_bbox[1]:tight_bbox[1]+tight_bbox[3], tight_bbox[0]:tight_bbox[0]+tight_bbox[2], :])

			# Store it
			cropped_thumbnails_list.append(thumbnail)

		return cropped_thumbnails_list

	# Perform the random transformations
	def random_transforms_(self, frame, labels, cropped_thumbnails_list, progressive_masks_list ,tight_bounding_boxes_list, no_loss_mask=None): 


		new_frame = frame.copy()
		#new_frame_feather = frame.copy()

		# Iterate over all new regions
		for index, (cropped_thumbnail, progressive_masks, tight_bbox) in enumerate(zip(cropped_thumbnails_list, progressive_masks_list, tight_bounding_boxes_list)):


			# Get random possible positions with region selection
			random_region = np.random.randint(0, len(self.possible_positions_list))
			random_index = np.random.randint(0, len(self.possible_positions_list[random_region]))

			# Retrieve position of new anchor
			x_new = self.possible_positions_list[random_region][random_index][0]
			y_new = self.possible_positions_list[random_region][random_index][1]

			# Transform it to the polar coordinates
			r_new, a_new = self.cartesian_to_polar_(x_new,y_new)

			r_current, a_current = self.cartesian_to_polar_(tight_bbox[0], tight_bbox[1])

			# Compute the rotation and scaling to apply with a random variations (+-5% for rotation, +-10% scaling)
			rotation = np.random.uniform(0.95*(a_new-a_current)*180/np.pi, 1.05*(a_new-a_current)*180/np.pi)
			scale = self.scaling_alpha * np.exp(self.scaling_beta*( r_new - r_current )) + self.scaling_gamma
			scale = np.random.uniform(0.9*scale,1.1*scale) 

			# Resize the thumbnail
			thumbnail_resized = cv2.resize(cropped_thumbnail, dsize=(int(cropped_thumbnail.shape[1]*scale), int(cropped_thumbnail.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)


			# Rotate the scaled thumbnail
			thumbnail_rotated = self.rotate_image(thumbnail_resized, rotation)


			# If mask is out of frame, ignore
			mask_width = thumbnail_rotated.shape[1]
			mask_height = thumbnail_rotated.shape[0]
			if mask_width + x_new >= self.width or mask_height + y_new >= self.height:
				print("Out of frame, removed: ", (mask_width + x_new, mask_height + y_new))
				continue

			# Create the concatenated progressive mask 
			overlay_mask = np.zeros((progressive_masks.shape[1], progressive_masks.shape[2]))
			for pro_mask in progressive_masks:

				# Compute the total mask
				overlay_mask = np.maximum(overlay_mask, pro_mask)

				# Send the different masks to get the new bounding box number

				pro_mask_resize = cv2.resize(pro_mask, dsize=(int(pro_mask.shape[1]*scale), int(pro_mask.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
				
				pro_mask_rotated = self.rotate_image(pro_mask_resize, rotation)

				new_bounding_box = self.compute_new_bounding_box(pro_mask_rotated, x_new, y_new)

				labels = np.concatenate((labels, new_bounding_box), axis=0)


			# Transform the mask to three channels
			overlay_mask = np.repeat(overlay_mask.reshape(overlay_mask.shape[0], overlay_mask.shape[1], 1), 3, axis=2)

			# Resize the thumbnail
			overlay_mask_resize = cv2.resize(overlay_mask, dsize=(int(overlay_mask.shape[1]*scale), int(overlay_mask.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)

			# Rotate the scaled thumbnail
			overlay_mask_rotated = self.rotate_image(overlay_mask_resize, rotation)

			# Apply the thumbnail to the image
			obj_mask = np.zeros(thumbnail_rotated.shape, thumbnail_rotated.dtype)
			obj_mask_index = np.where( overlay_mask_rotated >= 1-(1/self.border))
			obj_mask[obj_mask_index] = 255
			center = (int(x_new+thumbnail_rotated.shape[1]/2), int(y_new+thumbnail_rotated.shape[0]/2))
			
			new_frame = cv2.seamlessClone(thumbnail_rotated, new_frame, obj_mask, center ,cv2.NORMAL_CLONE)
			
			# Update the no loss mask
			if no_loss_mask is not None:
				no_loss_mask[y_new:y_new+thumbnail_rotated.shape[0], x_new:x_new+thumbnail_rotated.shape[1]] = 1.0

		return new_frame, labels, no_loss_mask


	# From the progressive mask, compute the new bounding box
	def compute_new_bounding_box(self, progressive_mask, x_new, y_new):
		
		list_indexes = np.where(progressive_mask > (1-1/self.border))

		if len(list_indexes) == 0:
			return np.expand_dims(np.array([x_new, y_new, progressive_mask.shape[1], progressive_mask.shape[0]]), axis=0)

		x_min = np.min(list_indexes[1])
		y_min = np.min(list_indexes[0])
		x_max = np.max(list_indexes[1])
		y_max = np.max(list_indexes[0])

		return np.expand_dims(np.array([x_min + x_new, y_min + y_new, x_max-x_min, y_max-y_min]), axis=0)

	# Compute the calibration mask border
	def calibration_mask_border_(self):

		list_indexes = np.where(self.calibration_mask == 1.0)

		for x, y in zip(list_indexes[1], list_indexes[0]):

			_ , theta = self.cartesian_to_polar_(x,y)

			if theta >= self.theta_min and theta <= self.theta_max:
				self.calibration_mask_border[y,x] = 0.0

	# overlay mask masked if pixel is from the field
	def overlay_mask_masked_(self, frame, thumbnail_rotated, overlay_mask_rotated):

		# Transform the frame and thumbnail to hsv
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		hsv_thumbnail = cv2.cvtColor(thumbnail_rotated, cv2.COLOR_BGR2HSV)

		# Compute the histogram of the frame field
		hist = cv2.calcHist([hsv],[0],None,[256],[0,256])[:,0][1:]

		# Get the peak
		argmax_hist = np.argmax(hist)

		# Get the mask where to remove field pixels
		print(thumbnail_rotated.shape)
		print(overlay_mask_rotated.shape)
		field_mask = np.where(np.abs(hsv_thumbnail[...,0]-argmax_hist) <= self.histogram_std)

		print(field_mask)

		# Remove them from the overlay mask
		overlay_mask_rotated[field_mask] = 0.0

		return overlay_mask_rotated


	def cartesian_to_polar_(self,x,y):
		x = x-self.width/2
		y = -(y-self.height/2)
		r = np.sqrt(x**2 + y**2)
		a = np.arctan2(y,x)
		return r, a

	def polar_to_cartesian_(self,r,a):
		x = int(r*np.cos(a) + self.width/2)
		y = -int(r*np.sin(a) - self.height/2)
		return x,y

	def draw_bbox_(self, frame, labels, name="augmented"):
		# Create a copy of the image to print rectangles over it
		
		final = np.copy(frame)

		# Iterate over all bounding boxes and draw rectangles on the original image
		for bbox in labels:

			# Get the bounding box position in termes of frame indexes
			x1 = bbox[0]
			y1 = bbox[1]
			x2 = bbox[0] + bbox[2]
			y2 = bbox[1] + bbox[3]

			# Draw the rectangle
			final = cv2.rectangle(final, (x1,y1), (x2,y2), (0,0,255),2)

		# Save the image
		cv2.imwrite(self.save_path + name + ".png", final)

	# separate into three regions
	def possible_positions_(self):

		for x, y in zip(self.possible_positions[1], self.possible_positions[0]):
			r, _ = self.cartesian_to_polar_(x,y)
			if r >= self.r_max:
				self.possible_positions_list[0].append((x,y))
			elif r >= self.r_min:
				self.possible_positions_list[1].append((x,y))

	def rotate_image(self, image, angle):
		"""
		Function by Aaron Snoswell
		Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
		(in degrees). The returned image will be large enough to hold the entire
		new image, with a black background
		"""

		# Get the image size
		# No that's not an error - NumPy stores image matricies backwards
		image_size = (image.shape[1], image.shape[0])
		image_center = tuple(np.array(image_size) / 2)

		# Convert the OpenCV 3x2 rotation matrix to 3x3
		rot_mat = np.vstack(
			[cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
		)

		rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

		# Shorthand for below calcs
		image_w2 = image_size[0] * 0.5
		image_h2 = image_size[1] * 0.5

		# Obtain the rotated coordinates of the image corners
		rotated_coords = [
			(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
			(np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
			(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
			(np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
		]

		# Find the size of the new image
		x_coords = [pt[0] for pt in rotated_coords]
		x_pos = [x for x in x_coords if x > 0]
		x_neg = [x for x in x_coords if x < 0]

		y_coords = [pt[1] for pt in rotated_coords]
		y_pos = [y for y in y_coords if y > 0]
		y_neg = [y for y in y_coords if y < 0]

		right_bound = max(x_pos)
		left_bound = min(x_neg)
		top_bound = max(y_pos)
		bot_bound = min(y_neg)

		new_w = int(abs(right_bound - left_bound))
		new_h = int(abs(top_bound - bot_bound))

		# We require a translation matrix to keep the image centred
		trans_mat = np.matrix([
			[1, 0, int(new_w * 0.5 - image_w2)],
			[0, 1, int(new_h * 0.5 - image_h2)],
			[0, 0, 1]
		])

		# Compute the tranform for the combined rotation and translation
		affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

		# Apply the transform
		result = cv2.warpAffine(
			image,
			affine_mat,
			(new_w, new_h),
			flags=cv2.INTER_LINEAR
		)

		return result


	def log(self, name="", value=""):

		value = str(value)

		myfile = open(self.log_file + str(self.trainset_start_index) + ".log", "a")
		myfile.write("[" + str(self.frame_number) + "] - ")
		myfile.write(name)
		myfile.write(": ")
		myfile.write(value)
		myfile.write("\n")
		myfile.close()