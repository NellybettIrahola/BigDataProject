import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Label Encode
def stringProcessing(var):
    df = pd.read_csv('./data/crimes_toronto.csv', index_col=None)

    data_types_string=['premisetype', 'offence', 'reportedmonth', 'reporteddayofweek', 'occurrencemonth', 'occurrencedayofweek','mci']
    output = df.copy()
    for col in data_types_string:
        output[col] = LabelEncoder().fit_transform(output[col])

    X=output
    if (var == 0):
        X = pd.get_dummies(df[data_types_string], prefix_sep='_')

    return X
#stringProcessing(1)

def labelModel():
    crimes=stringProcessing(1)
    features=['neighbourhood_id', 'total_area', 'total_population', 'home_prices', 'local_employment', 'social_assistance_recipients', 'catholic_school_graduation', 'catholic_school_literacy', 'catholic_university_applicants', 'x', 'y', 'index', 'premisetype', 'ucr_code', 'ucr_ext', 'offence', 'reportedyear', 'reportedmonth', 'reportedday', 'reporteddayofyear', 'reporteddayofweek', 'reportedhour', 'occurrenceyear', 'occurrencemonth', 'occurrenceday', 'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour', 'mci', 'lat', 'long', 'fid', 'year_difference']
    X=crimes[features]
    y=crimes.mci

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)

    return (X_res, y_res)

#def binaryModel():
    #df = stringProcessing(0)
    #features = ['premisetype_Apartment', 'premisetype_Commercial', 'premisetype_House', 'premisetype_Other', 'premisetype_Outside', 'offence_Administering Noxious Thing', 'offence_Aggravated Aslt Peace Officer', 'offence_Aggravated Assault', 'offence_Aggravated Assault Avails Pros', 'offence_Air Gun Or Pistol: Bodily Harm', 'offence_Assault', 'offence_Assault - Force/Thrt/Impede', 'offence_Assault - Resist/ Prevent Seiz', 'offence_Assault Bodily Harm', 'offence_Assault Peace Officer', 'offence_Assault Peace Officer Wpn/Cbh', 'offence_Assault With Weapon', 'offence_B&E', 'offence_B&E - To Steal Firearm', 'offence_B&E Out', "offence_B&E W'Intent", 'offence_Crim Negligence Bodily Harm', 'offence_Disarming Peace/Public Officer', 'offence_Discharge Firearm - Recklessly', 'offence_Discharge Firearm With Intent', 'offence_Pointing A Firearm', 'offence_Robbery - Armoured Car', 'offence_Robbery - Atm', 'offence_Robbery - Business', 'offence_Robbery - Delivery Person', 'offence_Robbery - Financial Institute', 'offence_Robbery - Home Invasion', 'offence_Robbery - Mugging', 'offence_Robbery - Other', 'offence_Robbery - Purse Snatch', 'offence_Robbery - Swarming', 'offence_Robbery - Taxi', 'offence_Robbery - Vehicle Jacking', 'offence_Robbery With Weapon', 'offence_Set/Place Trap/Intend Death/Bh', 'offence_Theft - Misapprop Funds Over', 'offence_Theft From Mail / Bag / Key', 'offence_Theft From Motor Vehicle Over', 'offence_Theft Of Motor Vehicle', 'offence_Theft Of Utilities Over', 'offence_Theft Over', 'offence_Theft Over - Bicycle', 'offence_Theft Over - Distraction', 'offence_Theft Over - Shoplifting', 'offence_Traps Likely Cause Bodily Harm', 'offence_Unlawfully Causing Bodily Harm', 'offence_Unlawfully In Dwelling-House', 'offence_Use Firearm / Immit Commit Off', 'reportedmonth_April', 'reportedmonth_August', 'reportedmonth_December', 'reportedmonth_February', 'reportedmonth_January', 'reportedmonth_July', 'reportedmonth_June', 'reportedmonth_March', 'reportedmonth_May', 'reportedmonth_November', 'reportedmonth_October', 'reportedmonth_September', 'reporteddayofweek_Friday    ', 'reporteddayofweek_Monday    ', 'reporteddayofweek_Saturday  ', 'reporteddayofweek_Sunday    ', 'reporteddayofweek_Thursday  ', 'reporteddayofweek_Tuesday   ', 'reporteddayofweek_Wednesday ', 'occurrencemonth_April', 'occurrencemonth_August', 'occurrencemonth_December', 'occurrencemonth_February', 'occurrencemonth_January', 'occurrencemonth_July', 'occurrencemonth_June', 'occurrencemonth_March', 'occurrencemonth_May', 'occurrencemonth_November', 'occurrencemonth_October', 'occurrencemonth_September', 'occurrencedayofweek_Friday    ', 'occurrencedayofweek_Monday    ', 'occurrencedayofweek_Saturday  ', 'occurrencedayofweek_Sunday    ', 'occurrencedayofweek_Thursday  ', 'occurrencedayofweek_Tuesday   ', 'occurrencedayofweek_Wednesday ']
    #mcis=['mci_Assault', 'mci_Auto Theft', 'mci_Break and Enter', 'mci_Robbery', 'mci_Theft Over']
    #X = df[features]
    #y = df[mcis]
    #sm = SMOTE(random_state=42)
    #X_res, y_res = sm.fit_sample(X, y)

labelModel()