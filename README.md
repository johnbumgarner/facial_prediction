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


#### OpenCV Data Gathering




#### OpenCV Recognizer Training
<p align="justify">

In the previous phase facial data was extracted from a dataset of images using the frontal face Haar Cascade classifier.  In this phase the data will be run through the <i>OpenCV Recognizer.</i>. The LBPH face recognition algorithm was used in this training phase.  The Local Binary Pattern (LBP) is a simple and efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number.

<i>Parameters: the LBPH uses 4 parameters:</i>

1. Radius: the radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get.

2. Neighbors: the number of sample points to build a Circular Local Binary Pattern from. An appropriate value is to use 8 sample points. Keep in mind: the more sample points you include, the higher the computational cost. Max value is 8.

3. Grid X: the number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. Max value is 8

4. Grid Y: The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector. Max value is 8.

The initial dataset created in the Data Gathering phase contained facial boundary box coordinates, labels and identification numbers. A <i>pickle</i> file named   <i>face_labels.pickle</i> will be used to contain the associations between the dataset images and their labels. Pickling is a process where a Python object hierarchy is converted into a byte stream and dumps it into a file by using dump function. This character stream contains all the information necessary to reconstruct the object in another python script.   

The facial boundary box coordinates for each image will be processed using the <i>OpenCV LBPHFaceRecognizer</i>. The output will be written to a YML file.  This file is primarily associated with Javascript by YAML.  YAML stand for "YAML Ain't Markup Language." YAML uses a text file and organizes it into a format which is Human-readable. 

```python
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=4, grid_x=4, grid_y=4)

with open('face_labels.pickle', 'wb') as pickle_file:
    pickle.dump(image_names, pickle_file)
 
recognizer.train(x_train, np.array(y_labels))
recognizer.write('face_train_data.yml')
```
</p>

#### OpenCV Recognizer Prediction
<p align="justify">
  
In this phase the data elements within the pickle and YAML files created in the training phase will be used in conjunction with the Haar Cascade classifier <i>haarcascade_frontalface_default.xml</i>.  The classifier is used to obtain the boundary box coordinates from the photograph of the unknown person that we are looking for within the image dataset. The <i>OpenCV</i> function <i>face_FaceRecognizer.predict()</i> is used to compute a <i>confidence score,</i> which indicates the match potential between the target image and one within the dataset. A perfect match will have a <i>confidence score</i> of zero. <i>Confidence scores</i> can assigned to various thresholds levels, which will allow for the possibility of close matches and no related matches within the dataset.   

```python
recognizer.read(training_data)

with open(image_labels, 'rb') as pickle_file:
   pickle_labels = pickle.load(pickle_file)
   labels = {value: key for key, value in pickle_labels.items()}
   
image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
image = cv2.resize(image, (image_height, image_width), interpolation=cv2.INTER_AREA)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

for (x_coordinate, y_coordinate, width, height) in faces:
   cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (255, 0, 255), 2)
   roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
   identified_person, confidence_score = recognizer.predict(roi_gray)
   if confidence_score == 0:
      <DO SOMETHING>
   elif 0 < confidence_score <= 10:
      <DO SOMETHING>
   else:
      <DO SOMETHING>
```

<i>face_FaceRecognizer.predict()</i> was able to accurately predicated that the target image of Natalie Portman matched a face within the dataset that contained 73 photos of well-known female actresses.

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_prediction/blob/main/graphic/absolute_match.jpg" width="225" height="225">
</p>

The <i>FaceRecognizer</i> algorithm was also able to predicated that the mirrior image of Natalie Portman matched a face within the dataset. The <i>confidence score</i> for this match was 3 precent, which was well within the threshold level for probable matches.

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_prediction/blob/main/graphic/natalie_portman_mirror_confidence_score.jpg" width="225" height="225">
</p>
</p>
