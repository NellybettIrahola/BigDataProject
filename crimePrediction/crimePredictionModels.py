from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def treeClassifier(sample):
    features = ['total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday',
                'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment',
                'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X_train,X_test,y_train,y_test=sample


    treeClass = DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy',max_depth=19,
                                     random_state=1)

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    #df=pd.DataFrame({'feature': features, 'importance': treeClass.feature_importances_}).sort_values(['importance'],ascending=[0])
    accuracy=metrics.accuracy_score(y_test, y_pred)
    f1score=metrics.f1_score(y_test, y_pred, average='macro')

    return (accuracy,f1score)
#treeClassifier()

def oneVsRestTreeClassifier(sample):
    X_train, X_test, y_train, y_test =sample
    treeClass = OneVsRestClassifier(DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy', max_depth=19,
                                     random_state=1))

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')

    return (accuracy, f1score)
#oneVsRestTreeClassifier()

def randomForestClassifier(sample):
    features = ['total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday',
                'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment',
                'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X_train, X_test, y_train, y_test = sample

    rf = RandomForestClassifier(min_samples_split=10,max_depth=30,n_estimators=150,random_state=1) #,
    rf = rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    f1score = metrics.f1_score(y_test, predicted, average='macro')
    #sprint(metrics.confusion_matrix(y_test, predicted))
    #print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True))
    return (accuracy, f1score)
#randomForestClassifier(imbalancePreProcessing1(1))

def randomForestClassifierOneVsRest(sample):
    X_train, X_test, y_train, y_test = sample

    rf = OneVsRestClassifier(RandomForestClassifier(min_samples_split=10,max_depth=30,n_estimators=150,random_state=1)) #n_estimators=5,max_features=8 min_samples_split=10,max_depth=30,n_estimators=150,
    rf = rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    f1score = metrics.f1_score(y_test, predicted, average='macro')

    return (accuracy, f1score)
#randomForestClassifierOneVsRest(imbalancePreProcessing1(0))

def bayes(sample):
    X_train, X_test, y_train, y_test = sample
    std_clf = make_pipeline(StandardScaler(), GaussianNB())
    std_clf.fit(X_train, y_train)
    predicted = std_clf.predict(X_test)

    accuracy=metrics.accuracy_score(y_test, predicted)
    f1score = metrics.f1_score(y_test, predicted, average='macro')

    return (accuracy, f1score)
#bayes()

def bayesOneVsRest(sample):
    X_train, X_test, y_train, y_test = sample
    std_clf = make_pipeline(StandardScaler(), OneVsRestClassifier(GaussianNB()))
    std_clf.fit(X_train, y_train)
    predicted = std_clf.predict(X_test)

    accuracy=metrics.accuracy_score(y_test, predicted)
    f1score = metrics.f1_score(y_test, predicted, average='macro')

    return (accuracy, f1score)
#bayesOneVsRest() f score NEAR MIST