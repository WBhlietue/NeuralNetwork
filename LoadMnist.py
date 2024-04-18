import NN
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np

mnist = tf.keras.datasets.mnist

(xTrain, yTrain,),(xTest, yTest) = mnist.load_data()
print(len(xTest))

xTest = xTest.reshape(10000, 784)
xTest = xTest.astype('float32') / 255

yTest = to_categorical(yTest)

nLoad = NN.LoadNetwork("models/test.txt")
corretc = 0
for i in range(len(xTest)):
    out = nLoad.Forward(xTest[i])  
    result = np.argmax(out)
    ans = np.argmax(yTest[i])
    if(result == ans):
        corretc += 1
    print(f"{i+1}. predict: {result}, answer: {ans}, accuracy: {corretc/(i+1)*100:.2f}%")
