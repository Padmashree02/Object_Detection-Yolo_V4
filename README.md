# Problem - Object Detection with Deep learning's YOLO V4 method.

Object detection using deep neural network are categories in two methods- 

  > One stage detector :- Effective detectors, that detects the object by analysing the whole image in a single pass.

  > Two stage detector :- Efficeint detectors good for its accuracy as these detectors detects the object first by gnerating region proposals and then on each proposed region object classification and bounding box are detected.

In this object detection part, the objects are detected by one stage method and using YOLO based model- YOLO V4

Solution :-

  > Network/Layer - yolov4

  > Framework - Darknet

  > Dataset (while training the network) - coco

  > Library - cv2, numpy, matplotlib.pyplot 

  > Pipeline (Inderfence/testing) :-

      : Read the image.

      : Load the network's model file (stored trained weights). 
      
      : Load the network's config file (stored network's configuration / architexture detail).

      : Load and extract the class names from the class file- coco dataset.

      : Read the network's files using Tensorflow framework.

      : Convert the image into blob format- helps the network to understand the image through blob format. 
        Note- the parameters of blob function to convert image into blob is as per the newtwork's configuration text file.
      
      : Set the blob image ready and pass the blob forward to the to the loaded model for detections.

      : The result of detection is list of list cardinally with detected (center_x, center_y) coordinate, width, height, confidence score of the box and then confidence score wrt other classes

      : Set the detection threshold value- detects how well the object is detected in the detected bounding box.
        Note- Based on model's output, set threshold- 0 (no detection will be display), 0.9 (multiple with irrelevant detections will be display) 

      : For each detections, extract the classID, confidence score. Then extract the bounding box dimensions and normalise each dimension wrt to image's shape.
      
      : Check whether the confidence score satisfies with the detection threshold value. 
      
        - If yes, then draw the bounding box with detected class as a label.

        - Else no, then will pass to the next detection.


Important note :- 

  > The inference time in the code has shown as 5289.31 ms but this is due to code had been executed under CPU runtime. 

  > Thus, I have share another image in which the inference time is shown as 1980.17 ms and this was due to the same code that had been executed under GPU runtime.

  > As the trained weight file's size is beyond github's limit. So, could not be able to upload .pb file, however the .pb file can be access from the SSD's respective github link.
