import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#========================================================
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(96, (11,11), input_shape = (150, 150, 3), activation = 'relu',padding = 'valid', strides = (4,4)))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding='valid'))
classifier.add(BatchNormalization())

# Step 2 - Convolution
classifier.add(Conv2D(256, (11,11), activation = 'relu',padding = 'valid', strides = (1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding='valid'))
classifier.add(BatchNormalization())
# Step 3 - Convolution
classifier.add(Conv2D(384, (3,3), activation = 'relu',padding = 'valid', strides = (1,1)))
classifier.add(BatchNormalization())
# Step 4 - Convolution
classifier.add(Conv2D(384, (3,3), activation = 'relu',padding = 'same', strides = (1,1)))
classifier.add(BatchNormalization())
# Step 5 - Convolution
classifier.add(Conv2D(256, (3,3), activation = 'relu',padding = 'same', strides = (1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding='same'))
classifier.add(BatchNormalization())
# Step 6 - Flatten
classifier.add(Flatten())
# Step 7 - Dense
classifier.add(Dense(4096, input_shape=(150*150*3,), activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
# Step 8 - Dense
classifier.add(Dense(4096, activation = 'relu'))
classifier.add(Dropout(0.4))  
classifier.add(BatchNormalization())
# Step 9 - Dense
classifier.add(Dense(1000, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())


# Step 10 - Dense
classifier.add(Dense(6, activation = 'softmax'))


# Part 2 - Fitting the CNN to the images
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




#+=========================================================================


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/saurav/Documents/Saurav/MLDL/practice/cnn/archived_data/intelImage/archive/seg_train/seg_train',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/home/saurav/Documents/Saurav/MLDL/practice/cnn/archived_data/intelImage/archive/seg_test/seg_test',
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = test_set,    
                         validation_steps = 500)

classifier.save("alexnet1.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




