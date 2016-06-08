'''
Created on Jun 8, 2016

@author: zluo
'''
from numpy import *
import operator
# Create a sample dataSet 
def createDataSet():
    dataSet = array([[1.1, 1.0],[1.0,1.1],[0,0.1],[0.1,0]])
    labels = array(['A','A','B','B'])
    return dataSet,labels

def classifer0(inX, dataSet, labels, k):
    diffMat = tile(inX, (dataSet.shape[0],1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount ={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]= classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    

def file2matrix(filename):
    fr = open(filename)
    lines = len(fr.readlines())
    returnMat = zeros((lines, 3))
    classLabelVector = []
    fr = open(filename)
    
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat, classLabelVector
    

import unittest
class TestKnn(unittest.TestCase):
    
    def test_createDataSet(self):
        group, labels = createDataSet()
        self.assertEqual(4, group.shape[0], 'DataSet shape[0] is not equal')
        self.assertEqual(2, group.shape[1], 'DataSet Size is not equal')

    
    def test_classifier0(self):
        group, labels = createDataSet()
        self.assertEqual('B', classifer0([0,0], group, labels, 3), 'Classifier should return B')
        self.assertEqual('A', classifer0([1,1], group, labels, 3), 'Classifier should return A')

if __name__ == '__main__':
    unittest.main()
