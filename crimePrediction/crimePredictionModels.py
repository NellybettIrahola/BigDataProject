from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from crimePrediction.crimePredictionDataPreprocessing import imbalancePreProcessing1,imbalancePreProcessing
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def treeClassifier():
    features = ['total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday',
                'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment',
                'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X_train,X_test,y_train,y_test=imbalancePreProcessing1(1)


    treeClass = DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy',max_depth=19,
                                     random_state=1)

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    #df=pd.DataFrame({'feature': features, 'importance': treeClass.feature_importances_}).sort_values(['importance'],ascending=[0])
    accuracy=metrics.accuracy_score(y_test, y_pred)
    f1score=metrics.f1_score(y_test, y_pred, average='micro')
    print(accuracy)
    print(f1score)
treeClassifier()

def oneVsRestTreeClassifier():
    X_train, X_test, y_train, y_test =imbalancePreProcessing(0)
    treeClass = OneVsRestClassifier(DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy', max_depth=19,
                                     random_state=1))

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred)

    print(accuracy)
#oneVsRestTreeClassifier()

def randomForestClassifier():
    features = ['total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday',
                'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment',
                'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X_train, X_test, y_train, y_test = imbalancePreProcessing(1)

    rf = RandomForestClassifier(n_estimators=5,max_features=8)
    rf = rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)

    #print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True))
    print(accuracy)
#randomForestClassifier()

def randomForestClassifierOneVsRest():
    X_train, X_test, y_train, y_test = imbalancePreProcessing(0)

    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=5,max_features=8))
    rf = rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(accuracy)
#randomForestClassifierOneVsRest()

def bayes():
    X_train, X_test, y_train, y_test = imbalancePreProcessing(1)
    std_clf = make_pipeline(StandardScaler(), GaussianNB())
    std_clf.fit(X_train, y_train)
    predicted = std_clf.predict(X_test)

    accuracy=metrics.accuracy_score(y_test, predicted)
    print(accuracy)
#bayes()

def bayesOneVsRest():
    X_train, X_test, y_train, y_test = imbalancePreProcessing(0)
    std_clf = make_pipeline(StandardScaler(), OneVsRestClassifier(GaussianNB()))
    std_clf.fit(X_train, y_train)
    predicted = std_clf.predict(X_test)

    accuracy=metrics.accuracy_score(y_test, predicted)
    print(accuracy)
#bayesOneVsRest() f score NEAR MIST