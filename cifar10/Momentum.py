import theano
from neupy import init
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, metrics, datasets
from neupy import algorithms, layers, environment
import torchfile
#from zca_whiten import zca_whiten

#environment.reproducible(1)
theano.config.floatX = 'float32'

x_train=torchfile.load('model_out_train.t7')
y_train=torchfile.load('train_label.t7')

target_scaler = OneHotEncoder()
target=y_train.reshape((-1,1))
y_train = target_scaler.fit_transform(target).todense()

x_test=torchfile.load('model_out_test.t7')
y_test=torchfile.load('test_label.t7')

target_scaler = OneHotEncoder()
target=y_test.reshape((-1,1))
y_test = target_scaler.fit_transform(target).todense()


#x_train=zca_whiten(x_train)
#x_test=zca_whiten(x_test)
network = layers.join(
        layers.Input(256),
	layers.Relu(25),
        layers.Softmax(10)
)
'''
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=2)
'''
gd=algorithms.Momentum(network)
gd.batch_size=128
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


