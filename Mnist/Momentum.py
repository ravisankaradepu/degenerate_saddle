import theano
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, metrics, datasets
from neupy import algorithms, layers, environment, init


theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

data = mnist.data / 255.
data = data - data.mean(axis=0)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6 / 7.)
)

network = layers.join(
        layers.Input(784),
        layers.Relu(25),
        layers.Softmax(10)
)
'''
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=2)
'''
gd=algorithms.Momentum(network)
gd.batchsize=128
l=[]
for i in range(10):
	gd.train(x_train,y_train,epochs=10)
	y_predicted = gd.predict(x_test).argmax(axis=1)
	y_testt = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))
	print(metrics.classification_report(y_testt, y_predicted))
	score = metrics.accuracy_score(y_testt, y_predicted)
	print("Validation accuracy: {:.2%}".format(score))
	l.append(round(score,4))
        print(l)
	
hess=algorithms.Hessian(network)
hess.iter=('momentum')
hess.train(x_train,y_train,epochs=1)

