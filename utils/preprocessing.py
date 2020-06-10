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
import torch
import numpy


def opencvToTorch(frame, device):

	# Transform the array into a torch tensor
	tensor = torch.from_numpy(frame)

	# Sending the tensor to the GPU
	tensor = tensor.to(device)

	# Transform the type of the data from uint8 to float
	tensor = tensor.type(torch.float)

	# Transpose to [Channels, Height, Width]
	tensor = tensor.transpose(0,2).transpose(1,2)

	# Add the batch dimension
	tensor.unsqueeze_(0)

	return tensor

def torchToOpencv(tensor):

	# Remove the batch dimension
	tensor.squeeze_(0)

	# Transpose to [Height, Width, Channels]
	tensor = tensor.transpose(0,2).transpose(0,1)

	# Transform the type of the data from float to uint8
	tensor = tensor.type(torch.uint8)

	# Sending the tensor to the CPU
	tensor = tensor.to("cpu")

	# Transform the tensor to a numpy array
	array = tensor.numpy()

	return array

def normalize(tensor):

	# Dividing by half of the max value to rescale to [0,2]
	tensor = (tensor * 2) / 255

	# Removing the median to rescale to [-1,1]
	tensor = tensor - 1

	return tensor

def unnormalize(tensor):

	# Adding 1 to recover the [0,2] interval
	tensor = tensor + 1

	# Multiplying by half of the max value to recover the [0,255] interval
	tensor = (tensor * 255)/2

	return tensor