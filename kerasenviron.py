from cgi import test
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from keras.datasets import mnist
#import feed forward classes
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.utils import np_utils
#upload our dataset and split into training and test set
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
#make our X_train data contain 60,000 by 28 pixel images and X_test contain 10,000. 
#reshape the pixels as a 784pixel long array 
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
# We want to convert the labels into a 10-entry one-hot encoded vector comprised of zeroes and just one 1 in the entry corresponding to the digit.
classes = 10
Y_train = np_utils.to_categorical(Y_train,classes)
Y_test = np_utils.to_categorical(Y_test,classes)
#set the sizee of input layer,hidden layer and number of epochs and batch size
input_size = 784
batch_size = 100
hidden_neurons = 100
epochs = 50
#we use the sequential model, dense and sigmoid actvation function including one hidden layer and one softmax output
model = Sequential([
    Dense(hidden_neurons,input_dim=input_size),
    Activation('sigmoid'),
    Dense(classes),
    Activation('softmax')
])
#crossentropy as lost cost function
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='sgd')
#train the model
model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1)
#evaluate the accuracy of the test data
score = model.evaluate(X_test,Y_test,verbose=1)
print('Test Accuracy:', score[1])
#visualize the weights of the hidden layer
weights = model.layers[0].get_weights()
fig = plt.figure()
w = weights[0].T
for neuron in range(hidden_neurons):
    ax = fig.add_subplot(10,10,neuron + 1)
    ax.axis("off")
    ax.imshow(np.reshape(w[neuron],(28,28)),cmap=cm.Greys_r)
plt.savefig("neuron_images.png",dpi=300)
plt.show()