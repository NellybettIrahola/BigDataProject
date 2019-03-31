import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier

# Label Encoder
def stringProcessing(var):
    df = pd.read_csv('./data/crimes_toronto.csv', index_col=None)
    if(var==1):
        data_types_string=['premisetype', 'offence', 'reportedmonth', 'reporteddayofweek', 'occurrencemonth', 'occurrencedayofweek','mci']
        output = df.copy()
        for col in data_types_string:
            output[col] = LabelEncoder().fit_transform(output[col])

        X=output
    else:
        X = pd.get_dummies(df, prefix_sep='_')

    return X
#stringProcessing(1)

def labelModel():
    crimes=stringProcessing(1)
    features=['neighbourhood_id', 'total_area', 'total_population', 'home_prices', 'local_employment', 'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy', 'catholic_university_applicants', 'x', 'y', 'index', 'premisetype', 'ucr_code', 'ucr_ext', 'offence', 'reportedyear', 'reportedmonth', 'reportedday', 'reporteddayofyear', 'reporteddayofweek', 'reportedhour', 'occurrenceyear', 'occurrencemonth', 'occurrenceday', 'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'lat', 'long', 'fid', 'year_difference']
    X=crimes[features]
    y=crimes.mci

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)

    return (X_res, y_res)

def binaryModel():
    crimes = stringProcessing(0)
    #print(list(crimes))
    features = ['neighbourhood_id', 'total_area', 'total_population', 'home_prices', 'local_employment', 'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy', 'catholic_university_applicants', 'x', 'y', 'index', 'ucr_code', 'ucr_ext', 'reportedyear', 'reportedday', 'reporteddayofyear', 'reportedhour', 'occurrenceyear', 'occurrenceday', 'occurrencedayofyear', 'occurrencehour', 'lat', 'long', 'fid', 'year_difference', 'premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside', 'offence_Administering Noxious Thing', 'offence_Aggravated Aslt Peace Officer', 'offence_Aggravated Assault', 'offence_Aggravated Assault Avails Pros', 'offence_Air Gun Or Pistol: Bodily Harm', 'offence_Assault', 'offence_Assault - Force/Thrt/Impede', 'offence_Assault - Resist/ Prevent Seiz', 'offence_Assault Bodily Harm', 'offence_Assault Peace Officer', 'offence_Assault Peace Officer Wpn/Cbh', 'offence_Assault With Weapon', 'offence_B&E', 'offence_B&E - To Steal Firearm', 'offence_B&E Out', "offence_B&E W'Intent", 'offence_Crim Negligence Bodily Harm', 'offence_Disarming Peace/Public Officer', 'offence_Discharge Firearm - Recklessly', 'offence_Discharge Firearm With Intent', 'offence_Pointing A Firearm', 'offence_Robbery - Armoured Car', 'offence_Robbery - Atm', 'offence_Robbery - Business', 'offence_Robbery - Delivery Person', 'offence_Robbery - Financial Institute', 'offence_Robbery - Home Invasion', 'offence_Robbery - Mugging', 'offence_Robbery - Other', 'offence_Robbery - Purse Snatch', 'offence_Robbery - Swarming', 'offence_Robbery - Taxi', 'offence_Robbery - Vehicle Jacking', 'offence_Robbery With Weapon', 'offence_Set/Place Trap/Intend Death/Bh', 'offence_Theft - Misapprop Funds Over', 'offence_Theft From Mail / Bag / Key', 'offence_Theft From Motor Vehicle Over', 'offence_Theft Of Motor Vehicle', 'offence_Theft Of Utilities Over', 'offence_Theft Over', 'offence_Theft Over - Bicycle', 'offence_Theft Over - Distraction', 'offence_Theft Over - Shoplifting', 'offence_Traps Likely Cause Bodily Harm', 'offence_Unlawfully Causing Bodily Harm', 'offence_Unlawfully In Dwelling-House', 'offence_Use Firearm / Immit Commit Off', 'reportedmonth_April', 'reportedmonth_August', 'reportedmonth_December', 'reportedmonth_February', 'reportedmonth_January', 'reportedmonth_July', 'reportedmonth_June', 'reportedmonth_March', 'reportedmonth_May', 'reportedmonth_November', 'reportedmonth_October', 'reportedmonth_September', 'reporteddayofweek_Friday    ', 'reporteddayofweek_Monday    ', 'reporteddayofweek_Saturday  ', 'reporteddayofweek_Sunday    ', 'reporteddayofweek_Thursday  ', 'reporteddayofweek_Tuesday   ', 'reporteddayofweek_Wednesday ', 'occurrencemonth_April', 'occurrencemonth_August', 'occurrencemonth_December', 'occurrencemonth_February', 'occurrencemonth_January', 'occurrencemonth_July', 'occurrencemonth_June', 'occurrencemonth_March', 'occurrencemonth_May', 'occurrencemonth_November', 'occurrencemonth_October', 'occurrencemonth_September', 'occurrencedayofweek_Friday    ', 'occurrencedayofweek_Monday    ', 'occurrencedayofweek_Saturday  ', 'occurrencedayofweek_Sunday    ', 'occurrencedayofweek_Thursday  ', 'occurrencedayofweek_Tuesday   ', 'occurrencedayofweek_Wednesday ']
    mcis=['mci_Assault', 'mci_Auto Theft', 'mci_Break and Enter', 'mci_Robbery', 'mci_Theft Over']
    X = crimes[features]
    y = crimes[mcis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return (X_train, X_test, y_train, y_test)

def sampling(var):
    X_res, y_res =var
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def treeClassifier():
    X_train, X_test, y_train, y_test=sampling(labelModel())
    treeclf = DecisionTreeClassifier(min_samples_split=300,
                                     criterion='entropy', max_depth=4,
                                     random_state=1)

    treeclf.fit(X_train, y_train)
    y_pred = treeclf.predict(X_test)
    print(y_pred)
#treeClassifier()

def oneVsRestTreeClassifier():
    X_train, X_test, y_train, y_test = binaryModel()
    treeclf = OneVsRestClassifier(DecisionTreeClassifier(min_samples_split=300,
                                     criterion='entropy', max_depth=4,
                                     random_state=1))

    treeclf.fit(X_train, y_train)
    y_pred = treeclf.predict(X_test)
    print(y_pred)
oneVsRestTreeClassifier()