#!/usr/local/bin/python3

##################################################################################
# “AS-IS” Clause
#
# Except as represented in this agreement, all work produced by Developer is
# provided “AS IS”. Other than as provided in this agreement, Developer makes no
# other warranties, express or implied, and hereby disclaims all implied warranties,
# including any warranty of merchantability and warranty of fitness for a particular
# purpose.
##################################################################################

##################################################################################
#
# Date Completed: July 24, 2019
# Author: John Bumgarner
#
# Date Revised: October 10, 2020
# Revised by: John Bumgarner
#
# This Python script is designed to use the OpenCV Haar Cascade Classifier for
# frontal faces and the LBPHFaceRecognizer module for facial prediction.
#
##################################################################################

#############################################################################################
# The OS module in provides functions for interacting with the operating system.
#
# OS.walk() generate the file names in a directory tree by walking the tree.
#############################################################################################
import os
from os import walk

#############################################################################################
# The math module is used to round up confidence scores linked to face_FaceRecognizer.predict()
#############################################################################################
import math

#############################################################################################
# The pickle module is used for serializing and de-serializing a Python object structure.
# Pickling is a way to convert a python object (list, dict, etc.) into a character stream.
# The data format used by pickle is Python-specific.
#############################################################################################
import pickle

#############################################################################################
# numpy is one of fundamental packages for scientific computing with Python
#############################################################################################
import numpy as np

#############################################################################################
# The OpenCV is a library of programming functions mainly aimed at real-time computer vision.
# The Python package is opencv-python
#
# reference: https://pypi.org/project/opencv-python
#############################################################################################
import cv2

#############################################################################################
# LBPH face recognition algorithm
#
# Local Binary Pattern (LBP) is a simple yet very efficient texture operator
# which labels the pixels of an image by thresholding the neighborhood of each
# pixel and considers the result as a binary number.
#
# Parameters: the LBPH uses 4 parameters:
#
# 1. Radius: the radius used for building the Circular Local Binary Pattern. The greater the
# radius, the smoother the image but more spatial information you can get.
#
# 2. Neighbors: the number of sample points to build a Circular Local Binary Pattern from.
# An appropriate value is to use 8 sample points. Keep in mind: the more sample points
# you include, the higher the computational cost. Max value is 8.
#
# 3. Grid X: the number of cells in the horizontal direction, 8 is a common value used in publications.
# The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
# Max value is 8
#
# 4. Grid Y: The number of cells in the vertical direction, 8 is a common value used in publications.
# The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
# Max value is 8.
#
#############################################################################################
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=4, grid_x=4, grid_y=4)

#############################################################################################
# A Haar Cascade is a machine learning object detection algorithm used to identify objects
# in an image or video and based on the concept of ​​ features.
#
# reference: https://ieeexplore.ieee.org/document/990517
#############################################################################################

# obtains the absolute paths to the haar cascade installed
# this path will change based on where the haar cascade files
# are located.
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_frontal_face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# Load the Haar Cascade Classifier used for frontal face images
face_cascade = cv2.CascadeClassifier(haar_frontal_face_model)


def get_image_files(directory_of_images):
    """
    This function is designed to traverse a directory tree and extract all the image names
    contained in the directory.

    :param directory_of_images:  the name of the target directory containing the images to be trained on.
    :return: list of images to be processed.
    """
    images_to_process = []
    for (directory_path, directory_names, filenames) in walk(directory_of_images):
        for filename in filenames:
            accepted_extensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.png', '.svg', '.tiff')
            if filename.endswith(accepted_extensions):
                images_to_process.append(os.path.join(directory_path, filename))
        return images_to_process


def prepare_training_data(image_dataset, height_aspect_ratio, width_aspect_ratio):
    """
     This function is designed to generate the data arrays used for labelling
     and the FaceRecognizer process.

     ref: https://docs.opencv.org/3.4/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63

    :param image_dataset: the names of the images that will be used in the training data
    :param height_aspect_ratio: this height attribute specifies the height of an image, in pixels
    :param width_aspect_ratio: this width attribute specifies the width of an image, in pixels
    :return: the image_names
             x_train contains the training data
             y_labels contains the association between the images and their labels.
    """

    # ID numbers associated with image labels
    current_id = 0

    # unique set of image names
    image_names = {}

    # list used to store ID numbers and image label information
    y_labels = []

    # list used to store the training data
    x_train = []

    for image in image_dataset:
        name = os.path.split(image)[-1].split(".")[0]
        if name not in image_names:
            image_names[name] = current_id
            current_id += 1
            id_ = image_names[name]

            # Read in the image
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

            # resizes the image to pre-established height and width dimensions
            img = cv2.resize(img, (height_aspect_ratio, width_aspect_ratio), interpolation=cv2.INTER_AREA)

            # cv2.cvtColor(input_image, flag) where flag determines the type of conversion
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
            faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=8,
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
            # (width, height) are the width and height of the rectangle
            for (x_coordinate, y_coordinate, width, height) in faces:
                print(f'Processing training data for image: {name}')

                # roi_gray is a numpy.ndarray based on the gray scale of the image
                roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

                # append the training data to the x_train list
                x_train.append(roi_gray)

                # append the label information to the y_labels list
                y_labels.append(id_)

    return image_names, x_train, y_labels


def create_training_data(image_names, x_train, y_labels):
    """
     This function is designed to generate the files used for labelling
     and the FaceRecognizer process.

    :param image_names: the names of the images that are in the training data
    :param x_train: the numpy.ndarray related to the images used in the training process.
    :param y_labels: labels associated with the images
    :return: pickle file containing the association between the images and their labels.
             YML file used by the FaceRecognizer process.
   """
    with open('face_labels.pickle', 'wb') as pickle_file:
        print('Creating relational information for the training data.')
        pickle.dump(image_names, pickle_file)

    # Trains the FaceRecognizer with given data and associated labels.
    # The training images, that means the faces you want to learn.
    # The image data and their corresponding labels have to be given
    # as a vector.
    recognizer.train(x_train, np.array(y_labels))
    print('Writing out the recognition training data.')

    # Saves a FaceRecognizer and its model state to a
    # given filename, either as XML or YML (aka YAML)
    recognizer.write('face_train_data.yml')
    print('Finished processing the image training data. \n')
    print("*** Items in the dataset generated ***")
    print("Total number of faces: ", len(x_train))
    print("Total number of labels: ", len(y_labels))


def facial_recognition(photo_name, aspect_ratio_height, aspect_ratio_width, training_data, photo_labels):
    """
    This function is designed to perform facial prediction on a target
    images against the training data previously generated.

    :param photo_name: the name of the image that will be processed
    :param aspect_ratio_height: this height attribute specifies the height of an image, in pixels
    :param aspect_ratio_width: this width attribute specifies the width of an image, in pixels
    :param training_data: YML file containing the FaceRecognizer data
    :param photo_labels:pickle file containing the association between the images and their labels
    :return: image with rectangle draw around facial area overlaid with image label and
             confidence score
   """
    # Loads a persisted FaceRecognizer model and state from a YML (aka YAML) file
    # face_train_data.yml
    recognizer.read(training_data)

    # Sets the font parameters to use with cv2.putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.60
    font_thickness = 2
    font_color = (0, 204, 0)

    # Open the pickle file containing the image names
    # and their associated labels.
    # face_labels.pickle
    with open(photo_labels, 'rb') as pickle_file:
        # loads the pickle file
        pickle_labels = pickle.load(pickle_file)
        # extract the key and values pairs from the pickle file.
        labels = {value: key for key, value in pickle_labels.items()}

    # Read in the image
    image = cv2.imread(photo_name, cv2.IMREAD_UNCHANGED)

    # resizes the image to pre-established height and width dimensions
    image = cv2.resize(image, (aspect_ratio_height, aspect_ratio_width), interpolation=cv2.INTER_AREA)

    # cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=8,
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # sets the starting location of the text
    y0, dy = 20, 12

    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in faces:

        # Draw bounding rectangle based on parameter dimensions
        # BGR color values (3 parameters)
        # BGR color (0, 255, 0) - https://rgb.to/0,255,0
        # Line width in pixels
        cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height),
                      (255, 0, 255), 2)

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # predicts a label and associated confidence (e.g. distance) for a given input image.
        identified_person, confidence_score = recognizer.predict(roi_gray)

        # The confidence score  (0-100) is adjustable, but setting the score
        # to high can produce invalid results.
        if confidence_score == 0:

            # extract the id number associated with a specific image label
            name = labels[identified_person]
            split_name = name.split('_')
            reformatted_name = ' '.join([i.capitalize() for i in split_name])

            # create the text to be overlaid on the image
            prediction_info = f'Name: {reformatted_name} \n\nAbsolute Match'

            # splits the label information based on new lines
            for i, line in enumerate(prediction_info.split('\n')):

                # calculates the y_coordinate of the text
                y_coordinate = y0 + i * dy

                # the function putText renders the specified text string in the image.
                cv2.putText(image, line, (x_coordinate - 15, y_coordinate - 2), font, font_size, font_color,
                            font_thickness)

        elif 0 < confidence_score <= 10:

            name = labels[identified_person]
            split_name = name.split('_')
            reformatted_name = ' '.join([i.capitalize() for i in split_name])

            prediction_info = f'Probability: {reformatted_name}\n\nConfidence score: ' \
                              f'{int(math.ceil(confidence_score))}%'

            for i, line in enumerate(prediction_info.split('\n')):

                # calculates the y_coordinate of the text
                y_coordinate = y0 + i * dy

                cv2.putText(image, line, (x_coordinate - 50, y_coordinate - 2), font, font_size, font_color,
                            font_thickness)

        else:
            # the function putText renders the specified text string in the image.
            cv2.putText(image, 'Unknown Person', (x_coordinate - 25, y_coordinate - 10), font, font_size, font_color,
                        font_thickness)

    return image


def display_facial_prediction_results(processed_photo):
    # write image out
    # rename as needed
    cv2.imwrite('absolute_match.jpg', processed_photo)

    # Display image with bounding rectangles
    # and title in a window. The window
    # automatically fits to the image size.
    cv2.imshow('Facial Prediction', processed_photo)

    # Displays the window infinitely
    key = cv2.waitKey(0) & 0xFF

    # Shuts down the display window and terminates
    # the Python process when a key is pressed on
    # the window
    if key == ord('q') or key == 113 or key == 27:
        cv2.destroyAllWindows()


image_directory = 'front_facing_images'
training_dataset = 'face_train_data.yml'
image_labels = 'face_labels.pickle'

images = get_image_files(image_directory)
image_height = 300
image_width = 300

# assemble the training data arrays and the associated labels
initial_training_data = prepare_training_data(images, image_height, image_width)
if initial_training_data:
    images = initial_training_data[0]
    training = initial_training_data[1]
    label_info = initial_training_data[2]

    # generate the training data and the associated labels
    create_training_data(images, training, label_info)

    ############################################################
    # This code below is running a single face prediction test.
    ###########################################################
    image_name = 'natalie_portman.jpeg'
    image_prediction = facial_recognition(image_name, image_height, image_width, training_dataset, image_labels)
    display_facial_prediction_results(image_prediction)
