"""
This is a simple script to cover the faces of people and car license plates. This repo will not have the YOLO model available.
You will have create your own or ask others nicely. When you do have one just plug them into the code and good to go.
"""
#Import necessary libraries
from pathlib import Path
import numpy as np
import cv2
import os
from Helper import *

#Set the paths to the images
Rooth_path=str(Path.cwd());
Input_path=Rooth_path+'\Input';
dir_list = os.listdir(Input_path);
Output_path=Rooth_path+'\Output';

#Execute stuff
classes_names=['Censored']; #Define the class names

#pick a model
intype=str('FA');
while len(intype)!=1 and intype!='F' and intype!='P':
    intype=str(input('Type \'F\' for face or \'P\' for plate:\n'));
yoloweight,yoloconfig=TypeModel(intype);

#Load the model for object detection
model_OD=cv2.dnn.readNet(yoloweight,yoloconfig, framework='Darknet');
layer_names = model_OD.getLayerNames();
output_layers=[layer_names[i-1] for i in model_OD.getUnconnectedOutLayers()];



for name in dir_list:
    #Load one image
    image = cv2.imread(Input_path+f'\{name}')
    height, width, channels = image.shape;
    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=1/255, size=(416, 416),mean=(0,0,0), crop=False);
    # set the blob to the model
    model_OD.setInput(blob);
    # forward pass through the model to carry out the detection
    outputs = model_OD.forward(output_layers);
    #Split the outputs
    class_ids,confidences, boxes, indexes= Get_boxes(outputs,width, height);

    #Draw the boxes
    m=[boxes[i] for i in indexes]       #Check how many boxes found
    if len(m)!=0:                       #If nothing detected don't save
        Save_image(class_ids,confidences,boxes,indexes,classes_names,Output_path,image,name);


print('Finished detecting. Check the Output folder');

