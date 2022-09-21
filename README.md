# YOLO-face-and-car-plate-cover
## Introduction
A simple program that uses trained yolo models to cover up faces and car license plates. A company asked me to create one as they have 80000 images that needed to be checked if faces of people or license plates of cars were visible. I gathered trained models from other repositories and wrote a script that would cover the detected objects with a black geometric shape.

## How to use
Make sure to get your own YOLO models first and then add images into the "Input folder". Run the "FaceCover.py" file, if there are any detections they will be saved to the "Output" folder.
