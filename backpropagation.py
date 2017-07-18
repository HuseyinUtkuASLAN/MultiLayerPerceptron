
from random import randint

import numpy as np 

class Backpropagation:

	layerCount = 0
	shape = None
	weights = []


	def __init__(self, layerSize):
		
		self.layerCount = len(layerSize) - 1
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		for (l1,l2) in zip(layerSize[:-1] , layerSize[1:]):
			
			self.weights.append(np.random.normal(  scale = 0.1, size = ( l2, l1 + 1 )  )  )
			self._previousWeightDelta.append(np.zeros((l2,l1+1)))


	def sigmoid(self,x, derivative = False):
		if derivative == False :
			return  1 / (1 + np.exp(-x))
		else :
			return self.sigmoid(x) * (1 - self.sigmoid(x))

	def Run(self,input):

		lnCases = input.shape[0]

		self._layerInput = []
		self._layerOutput = []

		# np.ones([1, lnCases]) for bias!
		for i in range(self.layerCount):
			if(i == 0):
				#print self.weights[0].shape, "\t", np.vstack([input.T, np.ones([1, lnCases])]).shape
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
			else :
				layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1,lnCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sigmoid(layerInput))

		return self._layerOutput[-1].T



	def TrainEpoch(self,input,target,traingRate = 0.2, momentum = 0.5):

		delta = []
		lnCases = input.shape[0]
		
		self.Run(input)

		for i in reversed(range(self.layerCount)):
			if i  == self.layerCount - 1:

				outputDelta = self._layerOutput[i] - target.T
				error = np.sum(outputDelta**2)
				delta.append(outputDelta * self.sigmoid(self._layerInput[i], True))
			## compare following layers
			else :
				deltaPullback = self.weights[i - 1].T.dot(delta[-1])
				delta.append(deltaPullback[:-1, :] * self.sigmoid(self._layerInput[i],True))

		## weight deltas
		for i in range(self.layerCount):
			deltaIndex = self.layerCount -1 - i

			if i == 0:
				layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
			else :
				layerOutput = np.vstack([self._layerOutput[i - 1], np.ones([1,self._layerOutput[i - 1].shape[1]])])

			currentWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[deltaIndex][None,:,:].transpose(2,1,0),axis = 0)

			weightDelta = traingRate * currentWeightDelta + momentum * self._previousWeightDelta[i] 

			self.weights[i] -= weightDelta

			self._previousWeightDelta[i] = weightDelta

		return error