import NN
import tensorflow as tf
from keras.utils import to_categorical

mnist = tf.keras.datasets.mnist

(xTrain, yTrain,),(xTest, yTest) = mnist.load_data()

xTrain = xTrain.reshape(60000, 784)
xTrain = xTrain.astype('float32') / 255

yTrain = to_categorical(yTrain)

nn = NN.NeuralNetwork(784, "mnist image read")
nn.AddLayer(512, "relu")
nn.AddLayer(512, "relu")
nn.AddLayer(10, "softmax")

nn.Train(inputs=xTrain,targets=yTrain,epoches=100,decay=0.9,initLearningRate=0.01, plotTr=True)
nn.Save("models/test.txt")



