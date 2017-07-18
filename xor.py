from sklearn.datasets import load_iris
import numpy as np 

from backpropagation import Backpropagation

def main():

	iris = load_iris()

	irisInput = (iris.data[:-10],iris.target[:-10])
	
	irisPrediction = (iris.data[-10:],iris.target[-10:])



	
	bp = Backpropagation((2,2,1))

	lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	lvTarget = np.array([[0.05],[0.05],[0.95],[0.95]])
	
	lnMax = 100000
	lnErr = 1e-5

	for i in range(lnMax-1):
		err = bp.TrainEpoch(lvInput,lvTarget)
		if i % 10000 == 5:
			print "Iteration : ", i, "   error : ", err
		if err <= lnErr:
			print "acceptable error reached : ", lnErr
			break

	lvOutput = bp.Run(lvInput)
	
	for i in range(len(lvOutput)):
		print lvOutput[i] 

	#print lvOutput
	"""
	bpIris = Backpropagation((4,9,3))
	bpInput = iris.data
	print bpInput
	bpOutput = bpIris.Run(bpInput)

	print bpOutput
	"""
if __name__ == "__main__":
	main()