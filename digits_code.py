import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models

#Reading the data
train = pd.read_csv(r'C:\Users\vihaa\Desktop\digits\train.csv')
X_test = pd.read_csv(r'C:\Users\vihaa\Desktop\digits\test.csv').values
X_train = train.iloc[:, 1:785].values
Y_train = train.iloc[:, :1].values

#Reshaping the data
rows, cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)
in_shape = (rows, cols, 1)
X_train = X_train.astype('float')
X_test = X_test.astype('float')

#Normalizing the data
X_train /= 255
X_test /= 255

#Converting to categorical vector
num_category = 10
Y_train = keras.utils.to_categorical(Y_train, num_category)

#Creating a Convulutional Neural Network Model
model = models.Sequential()

#Adding the layers
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=in_shape))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.20))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(num_category, activation='softmax'))

#Compiling the Model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#Fitting or Training the Model 
model.fit(X_train, Y_train,
          batch_size=128,
          epochs=10,
          verbose=1)

#Predicting the output for test dataset
Y_test = np.empty((X_test.shape[0]), int)
for i, j in zip(model.predict(X_test), range(X_test.shape[0])): 
    Y_test[j] = np.argmax(i)

# Checking to see if the Model makes correct predictions
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_test[i].reshape(28, 28), cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(Y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig

#Saving the Model and the output
if input("Look at the output plot. Satisfied with the results?(y/n) : ").lower() == 'y': 
    print("Saving the model.")
    model_digit_json = model.to_json()
    with open(r'C:\Users\vihaa\Desktop\digits\model_digit.json', "w") as json_file:
        json_file.write(model_digit_json)
    model.save_weights(r'C:\Users\vihaa\Desktop\digits\model_digit.h5')
    print("Saved model to disk")
    file = input("Saving the predicted output in pwd. Enter filename for saving: ")
    read = pd.read_csv(r'C:\Users\vihaa\Desktop\digits\sample_submission.csv')
    data = {"ImageId" : read['ImageId'].values,"Label" : Y_test}
    submit = pd.DataFrame(data)
    submit.to_csv(r'C:\Users\vihaa\Desktop\digits\{}.csv'.format(file), index = False)
