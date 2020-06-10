This code will save in the <code>output</code>code> folder. At each run of the code it will create an <code>experiment_i</code> folder containing three sub-fodlers:
1. A JSON file containing the output bounding boxes of the student network on the fisheye camera in the <code>student_outputs</code> folder.
2. The last weights of the network during the online distillation in the <code>networks</code> folder.
3. If requested by the <code>--outteacher 1</code> argument, it will save the output of the teacher network on the fisheye and thermal images in the <code>teacher_output</code> folder.