# import libraries and datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Marital Status
def classify_marital_status(name):
    if 'Mrs' in name:
        return 'Married'
    elif 'Miss' in name:
        return 'Unmarried'
    else:
        return 'Unknown'

# Is doctor
def is_doctor(name):
    if 'Dr' in name:
        return 'Yes'
    else:
        return 'No'

data = train
data['MaritalStatus'] = data['Name'].apply(classify_marital_status)
marriedSurvived = data.groupby('MaritalStatus')

data['IsDoctor'] = data['Name'].apply(is_doctor)
DoctorSurvived = data.groupby('IsDoctor')


print(train.head())