<p align="center">
  <img src="https://github.com/johnbumgarner/facial_prediction/blob/main/graphic/facial_recognition.png">
</p>

# Overview Facial Prediction

<p align="justify">

The repository [Facial Features Detection](https://github.com/johnbumgarner/facial_features_detection) discussed various <i>Python</i> methods used to identify human faces and multiple facial characteristics.  

</p>

## Primary objective of this repository (RE-WRITE)

<p align="justify">
This repository is going to examine various methods and algorithms that can be used to identify specific facial characteristics, such as the eye and mouth areas of a human face. The 3 images used in these tests are of the well-known female actress <i>Natalie Portman</i>.
  
Another objective of this repository is to determine the capabilities and limitations of the Python libraries used to perform these facial characteristics tests.
</p>

## Facial Detection and Facial Prediction

### Open Computer Vision Library (OpenCV):

<p align="justify">
  
This experiment used the CV2 modules <i>OpenCV-Python</i> and <i>OpenCV-Contrib-Python.</i>. These modules provide functions designed for real-time computer vision, image processing and machine learning. 

OpenCV is being used today for a wide range of applications which include:

- Automated inspection and surveillance
- Video/image search and retrieval
- Medical image analysis
- Criminal investigation
- Vehicle tag recognition
- Street view image stitching
- Robot and driver-less car navigation and control
- Signature pattern detection on documents
</p>


#### Haar Cascade Classifier - Facial Detection

<p align="justify">

One of the most basic Haar Cascade classifiers is <i>haarcascade_frontalface_default.xml</i>, whch is used to detect the facial area of a human face looking directly at the camera. This base-level algorithm comes pretrained, so it is able to identify images that have human face characteristics and their associated parameters and ones that have no human face characteristics, such as an image of a cat. 

It's worth noting that <i>Python</i> occasionally has issues locating the Haar Cascade classifiers on your system.  To solve this you can use the <i>Python</i> module <i>os</i> to find the absolute paths for the classifiers installed.  

```python
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_frontal_face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
```
The Haar Cascade classifier <i>haarcascade_frontalface_default.xml</i> is used in the following matter to located a human face within an image.  

```python
# This code was extraction from mutiple functions in the script facial_features_haar_cascade_classifiers.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)
for (x_coordinate, y_coordinate, width, height) in faces:
    cv2.rectangle(photograph, (x_coordinate, y_coordinate),
                  (x_coordinate + width, y_coordinate + height), (255, 0, 255), 2)
```

The image of <i>Natalie Portman</i> below has a <i>bounding box</i> drawn around the entire facial area identified by the Haar Cascade classifier  <i>haarcascade_frontalface_default.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/facial_front_detection.jpg">
</p>

</p>
