"""
This is the file in which I store all the ugly bunch of codes.
"""
import numpy as np
import cv2

#Function to get the relevant information from predictions
def Get_boxes(modeloutput,width,height):
    class_ids = []
    confidences = []
    boxes = []
    for output in modeloutput:  
        for identi in output:
            scores = identi[5:]
        
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                centerx = int(identi[0] * width)
                centery = int(identi[1] * height)
                w = int(identi[2] * width)
                h = int(identi[3] * height)
                # Rectangle coordinates
                x = int(centerx - w / 2)
                y = int(centery - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)                #Remove boxes that are overlaying on each other
    return class_ids,confidences,boxes, indexes

#Draw the boxes and save image
def Save_image(class_ids,confidences,boxes,indexes,classes_names,Outputpath,image,name):
    COLORS = [(0,0,0)]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            confidence = str("{:.2f}".format(confidences[i]))
            label = str(classes_names[class_ids[i]]+confidence)
            color = COLORS[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)         #Put a box on detected image
    cv2.imwrite(Outputpath+f'\{name}',image)                                #comment out if you don't want to save

    #simple function to determin face or plate
def TypeModel(char_input):
    if char_input=='F':
        #people faces, add your own model
        yoloweight='Name_of_your_faces_YOLO.weights'
        yoloconfig='Name_of_your_faces_YOLO.cfg'
    else:
        #License plate, add your own model
        yoloweight='Name_of_your_plate_YOLO.weights'
        yoloconfig='Name_of_your_plate_YOLO.cfg'
    return yoloweight,yoloconfig
