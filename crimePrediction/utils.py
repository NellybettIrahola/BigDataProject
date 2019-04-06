from crimePrediction.crimePredictionDataPreprocessing import imbalancePreProcessing1,imbalancePreProcessing,stringProcessing,sampling
from crimePrediction.crimePredictionModels import treeClassifier,randomForestClassifier,bayes,oneVsRestTreeClassifier,randomForestClassifierOneVsRest,bayesOneVsRest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Classifiers
overSamplingElements=imbalancePreProcessing(1)
underSamplingElements=imbalancePreProcessing1(1)

overSamplingElements1=imbalancePreProcessing(0)
underSamplingElements1=imbalancePreProcessing1(0)

def metricsResultFromSampling():

    print(treeClassifier(overSamplingElements))
    print(randomForestClassifier(overSamplingElements))
    print(bayes(overSamplingElements))

    print(treeClassifier(underSamplingElements))
    print(randomForestClassifier(underSamplingElements))
    print(bayes(underSamplingElements))

    return ""

metricsResultFromSampling()

def metricsResultFromSamplingOneVsRest():

    print(oneVsRestTreeClassifier(overSamplingElements1))
    print(randomForestClassifierOneVsRest(overSamplingElements1))
    print(bayesOneVsRest(overSamplingElements1))

    print(oneVsRestTreeClassifier(underSamplingElements1))
    print(randomForestClassifierOneVsRest(underSamplingElements1))
    print(bayesOneVsRest(underSamplingElements1))

    return ""
metricsResultFromSamplingOneVsRest()

df= pd.read_csv('./data/crimes_toronto.csv', index_col=None)
ax = df['mci'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Number of elements per class")
ax.set_xlabel("Class Name")
ax.set_ylabel("Frequency")

plt.show()