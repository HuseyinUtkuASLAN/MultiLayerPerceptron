from sklearn.datasets import load_iris
import numpy as np 
from sklearn import preprocessing

from backpropagation import Backpropagation

def targetToVector(target,unique):
	newTarget = np.zeros([target.shape[0],unique])
	for i in range(target.shape[0]):
		newTarget[i,target[i]] = 1
	
	return newTarget

def treshhold(vector):

	tmp = vector
	for x in np.nditer(tmp, op_flags=['readwrite']):

		if x <= 0.5:
			x[...] = 0
		else:
			x[...] = 1



	return tmp   


def main():

	## number of neurons in input layer (bias is not included)
	numInputNeuron = 4
	## number of neurons in hidden layer (bias is not included)
	numHiddenNeuron = 10
	## number of neurons in output layer
	numOutputNeuron = 3

	## learning rate    0 < l < 1
	learningRate = 0.2
	## momentum         0 < m < 1 
	momentum = 0.7

	maxIteration = 100000
	acceptableError = 1e-5


	iris = load_iris()

	## seperates every nth element for test values
	targetDivideIndex = 5

	irisInput = (iris.data[np.arange(len(iris.data))% targetDivideIndex != 1],iris.target[np.arange(len(iris.target))% targetDivideIndex != 1])
	
	irisPrediction = (iris.data[np.arange(len(iris.data))% targetDivideIndex == 1],iris.target[np.arange(len(iris.target))% targetDivideIndex == 1])

	min_max_scaler = preprocessing.MinMaxScaler()

	bpIris = Backpropagation((numInputNeuron,numHiddenNeuron,numOutputNeuron))
	bpInput = irisInput[0]
	bpInput = min_max_scaler.fit_transform(bpInput)
	bpTarget = irisInput[1]
	bpPrerictionInput = irisPrediction[0]
	bpPrerictionInput = min_max_scaler.fit_transform(bpPrerictionInput)
	bpPrerictionTarget = irisPrediction[1]
	newTarget = targetToVector(bpTarget,len(np.unique(bpTarget)))
	newPredictionTarget = targetToVector(bpPrerictionTarget,len(np.unique(bpTarget)))
	"""
	bp = Backpropagation((2,2,1))

	lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	lvTarget = np.array([[0.05],[0.05],[0.95],[0.95]])
	"""

	for i in range(maxIteration + 1):
		
		err = bpIris.TrainEpoch(bpInput,newTarget,learningRate,momentum)
		if i % 5000 == 0:
			print "Iteration : ",i, "   error : ", err
		if err <= acceptableError:
			print "acceptable error reached : ", acceptableError
			break


	lvOutput = bpIris.Run(bpInput)
	numTarget = len(newTarget)
	numTrueValue = 0
	for i in range(len(lvOutput)):
		if np.array_equal(treshhold(lvOutput[i]), newTarget[i]):
			numTrueValue += 1
		#print treshhold(lvOutput[i]) , "\t" , newTarget[i]
	
	print "\n\n\n"

	print "Count of input neuron : ", numInputNeuron
	print "Count of hidden neuron : ", numHiddenNeuron
	print "Count of output neuron : ", numOutputNeuron
	print "Learning rate : ", learningRate
	print "Momentum : ", momentum

	print "\n"

	print "Number of cases : ", numTarget
	print "Number of true cases : ", numTrueValue




	print "\n\n\nPredictions : "


	numOfTruePredictions = 0
	predicted = bpIris.Run(bpPrerictionInput)
	for i in range(len(predicted)):
		print treshhold(predicted[i]) , "\t", newPredictionTarget[i]
		if np.array_equal(predicted[i],newPredictionTarget[i]):
			numOfTruePredictions += 1

	print "Total number of predictions : ", len(predicted)
	print "Number of true predictions : ", numOfTruePredictions

	# print bpIris.weights

	"""
	print bpIris.weights[0].shape
	print np.vstack([bpPrerictionInput[0].T, np.ones([1, bpPrerictionInput[0].shape[0]])])

	numOfPrediction = len(bpPrerictionInput)

	for i in range(numOfPrediction):
		print treshhold( bpIris.predict(bpPrerictionInput[i])), "\t", bpPrerictionTarget[i]
	"""
	"""
	for i in range(len(bpPrerictionInput)):
		bpIris.predict(bpPrerictionInput[i])
	"""
	#output = bpIris.Run(bpPrerictionInput[0])
	#print bpPrerictionInput[0] , "\t", output, "\t", bpPrerictionOutput[0]
main()