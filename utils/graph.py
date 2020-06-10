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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------
# Updates a graph step by step during training
# --------------------------------------------

# [To be done]

class Graph:

	def __init__(self, folder, name):

		self.memory = ResultMemory()
		self.folder = folder
		self.name = name
		self.filename_graph = self.folder + "/" + self.name + "_graph_"
		self.filename_metric = self.folder + "/" + self.name + "_metrics.log"
		self.results_name = None
		self.number_of_column = 0
		self.title = ""

	def set_title(self, title):
		self.title = title

	def set_names(self, new_names):
		self.results_name = new_names
		self.number_of_column = len(new_names)
		self.memory.save_new(self.filename_metric,self.results_name)

	def update(self, new_result, save = False):
		self.memory.add_result(new_result)
		if save:
			for i in np.arange(self.number_of_column):
				self.save_graph(i)
		self.save_metric(new_result)

	def save_graph(self,column_number): 

		data = self.memory.get_column(column_number)
		number_of_steps = np.arange(len(data))

		plt.figure(figsize=(12, 14))

		#Axis transforms
		ax = plt.subplot(111)    
		ax.spines["top"].set_visible(False)    
		ax.spines["bottom"].set_visible(False)    
		ax.spines["right"].set_visible(False)    
		ax.spines["left"].set_visible(False)    
		ax.get_xaxis().tick_bottom()    
		ax.get_yaxis().tick_left() 


		plt.suptitle(self.title + self.results_name[column_number], fontsize=17, ha="center")

		plt.plot(number_of_steps, data, lw=2.5) 


		plt.savefig((self.filename_graph +self.results_name[column_number] + ".png"), bbox_inches="tight") 

	def save_metric(self, new_results):
		self.memory.save_new(self.filename_metric, new_results)



class ResultMemory():
	
	def __init__(self):

		self.result_holder = []	# Holds a list of lists of values

	def add_result(self,new_result):

		if type(new_result) == None:
			print("WARGNING : Result to be saved are empty - No result saved at this step (In PyDS.Evaluation.Graphs.ResultMemory)")
			return 


		if type(new_result) == type(np.ndarray([0,0])):
			new_result = new_result.tolist()

		if type(new_result) == type(self.result_holder):
			if len(self.result_holder) == 0:
				self.result_holder.append(new_result)
			else : 
				if len(new_result) != len(self.result_holder[0]):
					print("Warning : Length of result is changed, results might be inconsistent or missing (In PyDS.Evaluation.Graphs.ResultMemory)")
					self.result_holder.append(new_result)
				else:
					self.result_holder.append(new_result)

		else :
			print("Type of new result is not a list, it is : ", type(new_result))


	def get_results(self):
		return self.result_holder

	def get_column(self, column_number):
		column_result = []

		for iterator in self.result_holder:
			column_result.append(iterator[column_number])

		return column_result

	def save_new(self, filename, new_result):
		log = open(filename, "a")

		for result in new_result:
			log.write(str(result))
			log.write("  ;  ")
		log.write("\n")
		log.close()

def saveParameters(savepath, args):


	log = open(savepath, 'w')
	log.write("Fisheye camera file: " + args.fisheye+"\n")
	log.write("Thermal camera file: " + args.thermal+"\n")
	log.write("Device: " + args.device+"\n")
	log.write("Detection task"+"\n")
	log.write("Yolo configuration: "+ args.yoloconfig + "\n")
	if args.weights is not None:
		log.write("Using weights: " + args.weights+"\n")
	else:
		log.write("From scratch"+"\n")
	log.write("Learning rate: " + str(args.learningrate)+"\n")
	log.write("Training set size: " + str(args.trainsetsize)+"\n")
	log.write("Student subsample: " + str(args.studentsubsample)+"\n")
	log.write("Teacher subsample: " + str(args.teachersubsample)+"\n")
	log.close()