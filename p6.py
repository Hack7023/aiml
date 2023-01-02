6. Write a program to implement the naïve Bayesian classifier for a sample training data set 
stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets. 
import csv
import random
import math
def loadcsv(filename):
 lines = csv.reader(open(filename, "r"))
 dataset = list(lines)
 for i in range(len(dataset)):
 dataset[i] = [float(x) for x in dataset[i]] 
 return dataset
def splitDataset(dataset, splitRatio):
 trainSize = int(len(dataset) * splitRatio)
 trainSet = []
 trainSet,testSet = dataset[:trainSize],dataset[trainSize:]
 return [trainSet, testSet]
def mean(numbers):
 return sum(numbers)/(len(numbers))
def stdev(numbers):
 avg = mean(numbers)
 v = 0
 for x in numbers:
 v += (x-avg)**2
 return math.sqrt(v/(len(numbers)-1))
def summarizeByClass(dataset):
 separated = {}
 for i in range(len(dataset)):
 vector = dataset[i]
 if (vector[-1] not in separated):
 separated[vector[-1]] = []
 separated[vector[-1]].append(vector)
 summaries = {}
 for classValue, instances in separated.items():
 summaries[classValue] = [(mean(attribute), stdev(attribute)) for attribute in 
zip(*instances)][:-1]
 return summaries
def calculateProbability(x, mean, stdev):
 exponent = math.exp((-(x-mean)**2)/(2*(stdev**2)))
 return (1 / ((2*math.pi)**(1/2)*stdev)) * exponent
def predict(summaries, inputVector):
 probabilities = {}
 for classValue, classSummaries in summaries.items():
 probabilities[classValue] = 1
 for i in range(len(classSummaries)):
 mean, stdev = classSummaries[i]
 x = inputVector[i]
 probabilities[classValue] *= calculateProbability(x, mean, stdev)
 bestLabel, bestProb = None, -1
 for classValue, probability in probabilities.items():
 if bestLabel is None or probability > bestProb:
 bestProb = probability
 bestLabel = classValue
 return bestLabel
def getPredictions(summaries, testSet):
 predictions = []
 for i in range(len(testSet)):
 result = predict(summaries, testSet[i])
 predictions.append(result)
 return predictions
def getAccuracy(testSet, predictions):
 correct = 0
 for i in range(len(testSet)):
 if testSet[i][-1] == predictions[i]:
 correct += 1
 return (correct/(len(testSet))) * 100.0
filename = 'pima-indians-diabetes.csv'
splitRatio = 0.67
dataset = loadcsv(filename)
trainingSet, testSet = splitDataset(dataset, splitRatio) 
summaries = summarizeByClass(trainingSet) 
predictions = getPredictions(summaries, testSet)
print("\nPredictions:\n",predictions) 
accuracy = getAccuracy(testSet, predictions)
print('Accuracy ',accuracy)
OUTPUT:
Naive Bayes Classifier for concept learning problem
Split 14 rows into
Number of Training data: 12
Number of Test Data: 2
The values assumed for the concept learning attributes are
OUTLOOK=> Sunny=1 Overcast=2 Rain=3
TEMPERATURE=> Hot=1 Mild=2 Cool=3
HUMIDITY=> High=1 Normal=2
WIND=> Weak=1 Strong=2
TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5
The Training set are:
[1.0, 1.0, 1.0, 1.0, 5.0]
The Test data set are:
[1.0, 1.0, 1.0, 2.0, 5.0]
The Test data set are:
[2.0, 1.0, 1.0, 1.0, 10.0
