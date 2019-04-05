from crimePrediction.crimePredictionDataPreprocessing import imbalancePreProcessing1,imbalancePreProcessing
from crimePrediction.crimePredictionModels import treeClassifier,randomForestClassifier,bayes

#Classifiers
overSamplingElements=imbalancePreProcessing(1)
underSamplingElements=imbalancePreProcessing1(1)


def metricsResultFromSampling():

    print(treeClassifier(overSamplingElements))
    print(randomForestClassifier(overSamplingElements))
    print(bayes(overSamplingElements))

    print(treeClassifier(underSamplingElements))
    print(randomForestClassifier(underSamplingElements))
    print(bayes(underSamplingElements))

    return ""

metricsResultFromSampling()