from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense

size_of_image = 96

model = Sequential()
model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(size_of_image,size_of_image,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
#model.add(Dropout(0.2))

model.add(Dense(30))

# Summarize the model
model.summary()