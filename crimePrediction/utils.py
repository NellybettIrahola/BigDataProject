from crimePrediction.crimePredictionDataPreprocessing import imbalancePreProcessing1,imbalancePreProcessing,stringProcessing,sampling
from crimePrediction.crimePredictionModels import treeClassifier,randomForestClassifier,bayes,oneVsRestTreeClassifier,randomForestClassifierOneVsRest,bayesOneVsRest
import matplotlib.pyplot as plt
import numpy as np

#Classifiers
overSamplingElements=imbalancePreProcessing(1)
underSamplingElements=imbalancePreProcessing1(1)

overSamplingElements1=imbalancePreProcessing(1)
underSamplingElements1=imbalancePreProcessing1(1)

def metricsResultFromSampling():
    print("Accuracy / f1_score:\n")
    print("Oversampling Tree Classifier: {:.2f} / {:.2f}".format(treeClassifier(overSamplingElements)[0].item(),treeClassifier(overSamplingElements)[1].item()))
    print("Oversampling RandomForest Classifier: {:.2f} / {:.2f}".format(randomForestClassifier(overSamplingElements)[0].item(),randomForestClassifier(overSamplingElements)[1].item()))
    print("Oversampling Naive Bayes: {:.2f} / {:.2f}".format(bayes(overSamplingElements)[0].item(),bayes(overSamplingElements)[1].item()))

    print("Undersampling Tree Classifier: {:.2f} / {:.2f}".format(treeClassifier(underSamplingElements)[0].item(),treeClassifier(underSamplingElements)[1].item()))
    print("Undersampling RandomForest Classifier: {:.2f} / {:.2f}".format(randomForestClassifier(underSamplingElements)[0].item(),randomForestClassifier(underSamplingElements)[1].item()))
    print("Undersampling Naive Bayes: {:.2f} / {:.2f}".format(bayes(underSamplingElements)[0].item(),bayes(underSamplingElements)[1].item()))

    return ""

metricsResultFromSampling()

def metricsResultFromSamplingOneVsRest():

    print("OneVsRest Tree Classifier: {:.2f} / {:.2f}".format(oneVsRestTreeClassifier(overSamplingElements1)[0],oneVsRestTreeClassifier(overSamplingElements1)[1]))
    print("OneVsRest RandomForest Classifier {:.2f} / {:.2f}".format(randomForestClassifierOneVsRest(overSamplingElements1)[0],randomForestClassifierOneVsRest(overSamplingElements1)[1]))
    print("OneVsRest Naive Bayes: {:.2f} / {:.2f}".format(bayesOneVsRest(overSamplingElements1)[0],bayesOneVsRest(overSamplingElements1)[1]))

    print("Tree Classifier: {:.2f} / {:.2f}".format(oneVsRestTreeClassifier(underSamplingElements1)[0],oneVsRestTreeClassifier(underSamplingElements1)[1]))
    print("RandomForest: {:.2f} / {:.2f}".format(randomForestClassifierOneVsRest(underSamplingElements1)[0],randomForestClassifierOneVsRest(underSamplingElements1)[1]))
    print("Naive Bayes: {:.2f} / {:.2f}".format(bayesOneVsRest(underSamplingElements1)[0],bayesOneVsRest(underSamplingElements1)[1]))

    return ""
metricsResultFromSamplingOneVsRest()

def graph():

##############################################################
    #plt.rcParams.update({'font.size': 22})


    x = np.arange(5)
    freq = [47422,47422, 47422, 47422, 47422]

    fig, ay = plt.subplots()

    a,b,c,d,e=plt.bar(x, freq)
    b.set_facecolor('b')
    c.set_facecolor('g')
    d.set_facecolor('r')
    e.set_facecolor('y')
    plt.xticks(x, ('Assault','Auto Theft','Break \n and Enter','Robbery','Theft Over'))
    ay.set_title('Solution 1 - Oversampling')
    ay.set_xlabel("Class Name")
    ay.set_ylabel("Frequency")
    plt.show()

    #############################################################

    x = np.arange(5)
    freq = [4097,4097,4097,4097,4097]

    fig, ay = plt.subplots()

    a,b,c,d,e=plt.bar(x, freq)
    b.set_facecolor('b')
    c.set_facecolor('g')
    d.set_facecolor('r')
    e.set_facecolor('y')
    plt.xticks(x, ('Assault','Auto Theft','Break \n and Enter','Robbery','Theft Over'))
    ay.set_title('Solution 2 - Undersampling')
    ay.set_xlabel("Class Name")
    ay.set_ylabel("Frequency")
    plt.show()

    ############################################################
    x = np.arange(6)
    freq = [48470,27269,22347,14549,13435,4097]

    fig, ay = plt.subplots()

    a,b,c,d,e,f=plt.bar(x, freq)
    b.set_facecolor('b')
    c.set_facecolor('g')
    d.set_facecolor('r')
    e.set_facecolor('y')

    plt.xticks(x, ('General \n Assault','Break \n and Enter','Assault','Robbery','Auto Theft','Theft Over'))
    ay.set_title('Solution 3 - New Class')
    ay.set_xlabel("Class Name")
    ay.set_ylabel("Frequency")
    plt.show()
    ####################################################

    x = np.arange(5)
    freq = [70817,13435,27269,14549,4097]

    fig, ay = plt.subplots()

    a,b,c,d,e=plt.bar(x, freq)
    b.set_facecolor('b')
    c.set_facecolor('g')
    d.set_facecolor('r')
    e.set_facecolor('y')

    plt.xticks(x, ('Assault','Auto Theft','Break \n and Enter','Robbery','Theft Over'))
    ay.set_title('Original Distribution')
    ay.set_xlabel("Class Name")
    ay.set_ylabel("Frequency")
    plt.show()