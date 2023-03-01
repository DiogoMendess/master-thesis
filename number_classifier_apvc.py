import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

import tensorflow as tf
from keras import layers

# loads the mnist dataset (a built-in dataset from tensorflow), and loads the train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(len(x_test))

# normalizes the pixel values to the [0 ... 1] interval, for better results
x_train = x_train / 255.0
x_test = x_test / 255.0

# prepares the ground truth to the proper format, using 10 classes
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# shows the first 25 images from the dataset (we can see that the images are shuffled)
fig, ax = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(x_train[i * 5 + j], cmap=plt.get_cmap('gray'))
plt.show()

# shows the dimensions of the matrices for train and test
print("\n*** Dataset Dimensions ***\n")
print(f"x_train -> {x_train.shape}")
print(f"y_train -> {y_train.shape}")
print(f"x_test -> {x_test.shape}")
print(f"y_test -> {y_test.shape}\n")

# sets the label's IDs and the image dimensions
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
img_width = x_train.shape[1]
img_height = x_train.shape[2]

# splits the train data into train and validation (90% train - 10% validation)
split = int(x_train.shape[0] * 0.9)

x_val = x_train[split:, :]
y_val = y_train[split:, :]
x_train = x_train[:split, :]
y_train = y_train[:split, :]

# defines the architecture of the neural network
# hidden layer -> we decided to use the relu activation function (for nonlinearity), since we are classifying images
# output layer -> we decided to use the softmax activation function, so we can have the probabilities between classes
number_model = tf.keras.Sequential([
    layers.Flatten(input_shape=(img_width, img_height)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# shows a summary of the model
print("\n*** Model Summary ***\n")
number_model.summary()

# compiles the model, defining a loss function and an optimization algorithm
# for multi-class classification problems, the cross entropy loss function is the most suitable
# after multiple tests, we decided to use the Adam optimization algorithm with a learning rate of 0.001
number_model.compile(loss=tf.losses.CategoricalCrossentropy(),
                     optimizer=tf.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'])

# trains the neural network, saving the results data in the history variable
# with a batch size of 64, we noticed that the model becomes more generalizable for classifying new images
# regarding the number of epochs, we noticed that above 7 the loss value of the validation set started to increase
print("\n*** Model Training ***\n")
history = number_model.fit(x_train, y_train, batch_size=64, epochs=7, validation_data=(x_val, y_val))

# shows the plots with the evolution of the accuracy and loss functions during the training of the neural network
plt.figure(num=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.figure(num=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

plt.show()

# obtains the IDs of the true classes, from the test set
y_true = np.argmax(y_test, axis=1)

# makes the predictions, using the test set, and obtains the IDs of the predicted classes
output_pred = number_model(x_test)
y_pred = np.argmax(output_pred, axis=1)

# generates the confusion matrix, based on the test results, and shows it
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)

plt.show()

# shows a text report with the main classification metrics, in particular:
# -> precision, recall (to check which digits are more easily confused with others) and f1-score of each class
# -> accuracy of the trained model
print("\n*** Classification Report ***\n")
print(classification_report(y_true, y_pred, digits=4))







import cv2

# Zone coordinates of time in image (xmin, xmax, ymin, ymax)
TIME_BBOX = (563, 596, 63, 107)
#TIME_BBOX = (563, 819, 63, 107)

def get_time_img(frame):
    crop_img = frame[TIME_BBOX[2]:TIME_BBOX[3], TIME_BBOX[0]:TIME_BBOX[1]]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return crop_img


img = cv2.imread("C:/Users/diogo/Desktop/Tese/master-thesis/first_frame.jpg", cv2.IMREAD_GRAYSCALE)
time_img = get_time_img(img)
resized = cv2.resize(time_img, (28, 28))

pred_list = []
for i in range(15):
    prediction = number_model.predict(np.reshape(resized, (-1, 28, 28)))
    y_pred = np.argmax(prediction, axis=1)
    pred_list.append(y_pred[0])

print(pred_list)


