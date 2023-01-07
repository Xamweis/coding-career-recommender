import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

def numerize(x):
    if x == 'Less than 1 year':
        return 0
    elif x == 'More than 50 years':
        return 51
    else:
        return x

def splitter(x: str):
    x = x.lower()
    return x.split('(')[0].strip()

def encode_multichoices(dataseries, column, i=0):
    df = pd.DataFrame(dataseries[column])
    df = df[column].str.get_dummies(sep=';')
    df.columns = list(map(splitter, df.columns))  
    return df


LR2 = pickle.load(open('models/LR2.sav', 'rb'))
scaler_years = pickle.load(open('scaler/YearsCode.sav', 'rb'))
scaler_comp = pickle.load(open('scaler/ConvertedCompYearly.sav', 'rb'))

dummy = pd.read_csv('csv-data/dummy.csv')
dummy = dummy.rename({
    'How many years of coding experience do you have?':'YearsCode',
    'Which annual salary do you approx. want in dollars? ':'ConvertedCompYearly'
}, axis=1)

survey = pd.read_csv('csv-data/response.csv').drop(['email:'], axis=1) #IRONHACK mid-project survey
survey = survey.rename({
    'How many years of coding experience do you have?':'YearsCode',
    'Which annual salary do you approx. want in dollars? ':'ConvertedCompYearly'
}, axis=1)

data = pd.concat([dummy, survey], axis=0)

data['YearsCode'] = data['YearsCode'].fillna(0)
data['YearsCode'] = data['YearsCode'].apply(numerize)
data['YearsCode'] = data['YearsCode'].astype(int)
data['ConvertedCompYearly'] = data['ConvertedCompYearly'].fillna(data['ConvertedCompYearly'].mean())
data['ConvertedCompYearly'] = data['ConvertedCompYearly'].astype(float)

cats = pd.DataFrame([])
i = 0
for col in data.select_dtypes(include=object).columns:
    encoded = encode_multichoices(data, col)
    encoded = encoded.add_prefix(f'{i}_')
    cats = pd.concat([cats, encoded], axis=1)
    i += 1

data['YearsCode'] = scaler_years.transform(pd.DataFrame(data['YearsCode']))
data['ConvertedCompYearly'] = scaler_comp.transform(pd.DataFrame(data['ConvertedCompYearly']))

# concat with numerical
treated = pd.concat([data.select_dtypes(include=np.number), cats], axis=1)

predictions = pd.DataFrame(LR2.predict(treated))[1:].values[0][0]
print()
print()
print('My Suggestion based on your answers: ', predictions.upper())
print()
print()