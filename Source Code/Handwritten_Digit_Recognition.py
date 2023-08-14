import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the dataset
mnist = tf.keras.datasets.mnist #Directly loading the dataset from tensorflow.

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Splitting the data into Training data and Testing Data. 

x_train = tf.keras.utils.normalize(x_train, axis = 1) #Scaling it down so that every value is between 0 and 1.
x_test = tf.keras.utils.normalize(x_test, axis = 1)
#---------------------------------------------------------

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape =  (28, 28))) #Flatten - turns input shape into one flat layer, here 28*28 = 784 pixels, instead of having the grid.
model.add(tf.keras.layers.Dense(128, activation = 'relu')) #dense - basic neural network layer, where each neuron is connected to each other neuron.
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 100) #epochs - tells how many times the model will see the same data all over again.

model.save('Handwritten_Digit_Recognition_Model')
#---------------------------------------------------------

model = tf.keras.models.load_model('Handwritten_Digit_Recognition_Model')

loss, accuracy = model.evaluate(x_test, y_test)

print(f"The loss is: {loss}")
print(f"The accuracy of the model is: {accuracy}")
#---------------------------------------------------------

img_num = 1
while os.path.isfile(f"E:\VS Folder\Python Projects\Digits_Samples\Digit_{img_num}.png"):
    try:
        img = cv2.imread(f"E:\VS Folder\Python Projects\Digits_Samples\Digit_{img_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The digit recognized is: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        img_num = img_num + 1

