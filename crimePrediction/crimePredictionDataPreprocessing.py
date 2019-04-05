import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import tree

# Label Encoder, get_dummies
def stringProcessing(var):
    df = pd.read_csv('./data/crimes_toronto.csv', index_col=None)
    data_types_string=['mci','occurrencemonth', 'occurrencedayofweek'] #'premisetype', 'offence', 'reportedmonth', 'reporteddayofweek', 'occurrencemonth', 'occurrencedayofweek','mci']
    output = df.copy()
    for col in data_types_string:
        output[col] = LabelEncoder().fit_transform(output[col])

    X = pd.get_dummies(output, prefix_sep='_')

    return X
#stringProcessing(1)


def sampling(var):
    X_res, y_res =var
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def imbalancePreProcessing(var):
    crimes=stringProcessing(1)
    #print(list(crimes))
    features = ['total_area', 'total_population', 'home_prices', 'local_employment', 'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy', 'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday', 'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X=crimes[features]
    y=crimes.mci
    X_train, X_test, y_train, y_test=sampling((X,y))

    sm = SMOTENC(random_state=1,categorical_features=[16,17,18,19,20])
    X_res, y_res = sm.fit_sample(X_train, y_train)
    #print(len(y_res[y_res == 1]))
    #print(len(y_res[y_res == 0]))
    #print(len(y_res[y_res == 2]))
    if(var==0):
        y_res=pd.get_dummies(y_res, prefix_sep='_')
        y_test = pd.get_dummies(y_test, prefix_sep='_')

    #print(y_res)
    #print(X_test)
    return (X_res,X_test, y_res,y_test)
#imbalancePreProcessing(1)

def imbalancePreProcessing1(var):
    crimes=stringProcessing(1)
    #print(list(crimes))
    features = ['total_area', 'total_population', 'home_prices', 'local_employment', 'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy', 'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday', 'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X=crimes[features]
    y=crimes.mci
    X_train, X_test, y_train, y_test=sampling((X,y))

    rus = NearMiss(version=1,random_state=1)
    X_res, y_res = rus.fit_resample(X, y)
    #print(len(y_res[y_res == 1]))
    #print(len(y_res[y_res == 0]))
    #print(len(y_res[y_res == 2]))
    if(var==0):
        y_res=pd.get_dummies(y_res, prefix_sep='_')
        y_test = pd.get_dummies(y_test, prefix_sep='_')

    #print(y_res)
    #print(X_test)
    return (X_res,X_test, y_res,y_test)
#imbalancePreProcessing1(1)

def dataInitial():
    crimes = stringProcessing(1)
    features = ['total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrencemonth', 'occurrenceday',
                'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'premisetype_Apartment',
                'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside']

    X = crimes[features]
    y = crimes.mci
    return (X,y)

def paramEstimator():
    X, y = dataInitial()
    parameters = {'max_depth': range(5, 20),'min_samples_split':[50,150,450,600]}
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=1)
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4,cv=inner_cv)
    clf.fit(X=X, y=y)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
#paramEstimator()


def paramEstimatorRandomForest():
    X, y = dataInitial()
    parameters = {'n_estimators': range(5, 12), 'max_features': range(5, 10)}
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=1)
    clf = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=4,cv=inner_cv)
    clf.fit(X=X, y=y)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
#paramEstimatorRandomForest()

