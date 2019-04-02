import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics,tree

# Label Encoder, get_dummies
def stringProcessing(var):
    df = pd.read_csv('./data/crimes_toronto.csv', index_col=None)
    data_types_string=['mci'] #'premisetype', 'offence', 'reportedmonth', 'reporteddayofweek', 'occurrencemonth', 'occurrencedayofweek','mci']
    output = df.copy()
    for col in data_types_string:
        output[col] = LabelEncoder().fit_transform(output[col])

    X = pd.get_dummies(output, prefix_sep='_')

    return X
#stringProcessing(1)

def imbalancePreProcessing():
    crimes=stringProcessing(1)
    #print(list(crimes))
    features = ['neighbourhood_id', 'total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants', 'occurrenceyear', 'occurrenceday', 'occurrencedayofyear',
                'occurrencehour', 'lat',
                'long', 'premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House',
                'premisetype_Other', 'premisetype_Outside', 'occurrencemonth_April', 'occurrencemonth_August',
                'occurrencemonth_December',
                'occurrencemonth_February', 'occurrencemonth_January', 'occurrencemonth_July', 'occurrencemonth_June',
                'occurrencemonth_March', 'occurrencemonth_May', 'occurrencemonth_November', 'occurrencemonth_October',
                'occurrencemonth_September', 'occurrencedayofweek_Friday    ', 'occurrencedayofweek_Monday    ',
                'occurrencedayofweek_Saturday  ', 'occurrencedayofweek_Sunday    ', 'occurrencedayofweek_Thursday  ',
                'occurrencedayofweek_Tuesday   ', 'occurrencedayofweek_Wednesday ']
    X=crimes[features]
    y=crimes.mci

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    y_res=pd.get_dummies(y_res, prefix_sep='_')

    return (X_res, y_res)


def sampling(var):
    X_res, y_res =var
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def paramEstimator():
    X_train, X_test, y_train, y_test = sampling(imbalancePreProcessing())
    parameters = {'max_depth': range(5, 20),'min_samples_split':[50,150,450,600]}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
    clf.fit(X=X_train, y=y_train)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
#paramEstimator()

def treeClassifier():
    features = ['neighbourhood_id', 'total_area', 'total_population', 'home_prices', 'local_employment',
                'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy',
                'catholic_university_applicants','occurrenceyear', 'occurrenceday', 'occurrencedayofyear', 'occurrencehour', 'lat',
                'long', 'premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House',
                'premisetype_Other', 'premisetype_Outside','occurrencemonth_April', 'occurrencemonth_August', 'occurrencemonth_December',
                'occurrencemonth_February', 'occurrencemonth_January', 'occurrencemonth_July', 'occurrencemonth_June',
                'occurrencemonth_March', 'occurrencemonth_May', 'occurrencemonth_November', 'occurrencemonth_October',
                'occurrencemonth_September', 'occurrencedayofweek_Friday    ', 'occurrencedayofweek_Monday    ',
                'occurrencedayofweek_Saturday  ', 'occurrencedayofweek_Sunday    ', 'occurrencedayofweek_Thursday  ',
                'occurrencedayofweek_Tuesday   ', 'occurrencedayofweek_Wednesday ']

    X_train,X_test,y_train,y_test=sampling(imbalancePreProcessing())


    treeClass = DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy',max_depth=19,
                                     random_state=1)

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    df=pd.DataFrame({'feature': features, 'importance': treeClass.feature_importances_}).sort_values(['importance'],ascending=[0])
    accuracy=metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
#treeClassifier()

def oneVsRestTreeClassifier():
    X_train, X_test, y_train, y_test =sampling(imbalancePreProcessing())
    treeClass = OneVsRestClassifier(DecisionTreeClassifier(min_samples_split=50,
                                     criterion='entropy', max_depth=19,
                                     random_state=1))

    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred)

    print(accuracy)
#oneVsRestTreeClassifier()

