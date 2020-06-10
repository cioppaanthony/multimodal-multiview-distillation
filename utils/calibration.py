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
import cv2
import sys
from tqdm import tqdm


class Calibration:

	def __init__(self, thermal, fisheye):

		#Fisheye camera matrices
		self.F_scaled_K=np.array([[338.6863562, 0.0, 640.06219594], [0.0, 339.31227988, 666.90804806], [0.0, 0.0, 1.0]])
		self.F_distCoeffs=np.array([[0.044648874214807246], [-0.01588031369738762], [0.026328114058045533], [-0.00733933222733357]])
		self.M_fisheyetothermal= np.array([[ 1.71717189e+03, -3.85011041e+02,  2.48732393e+02],
		[ 1.03349351e+02,  1.06288123e+02,  1.78010673e+03],
		[ 6.10443736e-01, -2.24342705e+00,  1.00000000e+00]])

		#Thermal camera matrices
		self.T_K=np.array([[1, 0.0, 320], [0.0, 1, 240], [0.0, 0.0, 1.0]])
		self.T_distCoeffs=np.array([[-1.61169758e-06], [4.66585872e-12], [0.00000000e+00], [0.00000000e+00]])
		self.M_thermaltofisheye = np.array([[ 1.94157709e-02, -5.88414913e-04, -3.40124363e+00],
		[ 4.03322030e-03,  7.96877683e-03, -1.42013374e+01],
		[-1.14775066e-03,  1.68920914e-02,  1.00000000e+00]])

		self.thermal_width = thermal.shape[1]
		self.thermal_height = thermal.shape[0]

		self.fisheye_width = fisheye.size()[3]
		self.fisheye_height = fisheye.size()[2]

	def convert(self,box):
 
		toprx = box[0]-box[2]/2.0
		topry = box[1]-box[3]/2.0
		toplx = box[0]+box[2]/2.0
		toply = topry#box[0]+box[3]/2.0
		botrx = toprx
		botry = topry+box[3]
		botlx = toplx
		botly = toply+box[3]
		return (np.array([[[toprx,topry], [toplx,toply], [botrx,botry], [botlx,botly]]]))

	def convertback(self,points):

		width=points[2]-points[0]
		height=points[7]-points[3]
		centerx=points[0]+width/2.0
		centery=points[1]+height/2.0
		return (centerx, centery, width, height)

	def thermaltofisheye(self,thermal_tensor):

		fisheye_tensor = torch.zeros(thermal_tensor.size())

		counter = 0

		for thermal_bbox in thermal_tensor:

			bbox = (thermal_bbox[2]*self.thermal_width, thermal_bbox[3]*self.thermal_height, thermal_bbox[4]*self.thermal_width, thermal_bbox[5]*self.thermal_height)

			thermalPoints = self.convert(bbox)
			T_undistPoints = cv2.undistortPoints(thermalPoints, self.T_K, self.T_distCoeffs, P=self.T_K)
			Fisheye_undist = cv2.perspectiveTransform(T_undistPoints,self.M_thermaltofisheye)
			Fisheye_dist = cv2.fisheye.distortPoints(Fisheye_undist, self.F_scaled_K, self.F_distCoeffs)
			Fisheye_box = self.convertback(np.array(Fisheye_dist ).ravel())

			fisheye_tensor[counter,0] = thermal_bbox[0]
			fisheye_tensor[counter,1] = thermal_bbox[1]
			fisheye_tensor[counter,2] = np.asscalar(Fisheye_box[0]/self.fisheye_width)
			fisheye_tensor[counter,3] = np.asscalar(Fisheye_box[1]/self.fisheye_height)
			fisheye_tensor[counter,4] = np.asscalar(Fisheye_box[2]/self.fisheye_width)
			fisheye_tensor[counter,5] = np.asscalar(Fisheye_box[3]/self.fisheye_height)

			counter += 1

		return (fisheye_tensor)

	def correspondance(self,x,y):

		# [To be done] Identity at the moment, replace by the calibration
		tensor = torch.zeros(1,6)
		tensor[0,2] = float(x)
		tensor[0,3] = float(y)
		fisheye_new = self.thermaltofisheye(tensor)
		x_new, y_new = fisheye_new[0,2], fisheye_new[0,3]
		return x_new, y_new

	def calibrationMask(self, fisheye, thermal):

		mask = torch.zeros(fisheye.size()[2], fisheye.size()[3])

		for y in np.arange(self.thermal_height):
			print(y)
			for x in np.arange(self.thermal_width):
				x_new, y_new = self.correspondance(x/self.thermal_width, y/self.thermal_height)
				x_new = int(x_new*self.fisheye_width)
				y_new = int(y_new*self.fisheye_height)
				mask[y_new, x_new] = 1
		return mask

