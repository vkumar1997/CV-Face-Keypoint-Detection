
import cv2
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.utils import shuffle

class arrange_images:
    def prepare(self):
        training_data = pd.read_csv('data/training.csv')
        training_data = shuffle(training_data)
        training_data = training_data.dropna()
        print("Removed empty rows")
        print("Shape of training data (30 coordinates and 1 image pixel info)" + str(training_data.shape))

        testing_data = pd.read_csv('data/test.csv')
        print("Shape of testing data (1 image id and 1 image pixel info)" + str(testing_data.shape))


        size_of_image = 96

        training_input_Y = training_data.iloc[:,0:30]
        training_input_X_temp = training_data.iloc[:,30] #need to split column
        testing_input_X_temp = testing_data.iloc[:,1] #need to split column

        training_input_Y = training_input_Y.divide(size_of_image) #all values normalized from 0 to 1
        print("Normalized keypoint co-ordinates from 0 to 1 (training data)")

        training_faces=[]
        for face in training_input_X_temp:
            pixels = face.split(' ') #splitting pixels values
            pixels = list(map(float,pixels)) #mapping to float 
            pixels = [float(x/255) for x in pixels] #normalizing from 0 to 1
            training_faces.append(pixels) # append to list
            
        training_input_X = pd.DataFrame(training_faces) #convert back to dataframe
        print("Shape of training data input = " + str(training_input_X.shape))
        print("Shape of training data output = " + str(training_input_Y.shape))
        print("Normalized image pixel info from 0 to 1 (training data)")



        testing_faces=[]
        for face in testing_input_X_temp:
            pixels = face.split(' ') #splitting pixels values
            pixels = list(map(float,pixels)) #mapping to float 
            pixels = [float(x/255) for x in pixels] #normalizing from 0 to 1
            testing_faces.append(pixels) # append to list
            
        testing_input_X = pd.DataFrame(testing_faces) #convert back to dataframe
        print("Shape of testing data input = " + str(testing_input_X.shape))
        print("Normalized image pixel info from 0 to 1 (training data)")
        return [training_input_X,training_input_Y,testing_input_X]
