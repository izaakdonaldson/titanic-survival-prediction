# import libraries and datasets
import pandas as pd

expected_columns = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',
    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
]

def get_X_y(df):
    # Fills in missing age data with average
    df['Age'] = df['Age'].fillna(int(df['Age'].dropna().mean()))

    # Fills in missing age data with average
    df['Fare'] = df['Fare'].fillna(int(df['Fare'].dropna().mean()))

    #Fill missing name with John, Mr. Doe
    df['Name'].fillna('John, Mr. Doe')

    # Adds column for age category (0s, 10s, 20s, 30s, 40s, 50s, 60+)
    bins = [0, 10, 20, 30, 40, 50, 60, float('inf')]
    labels = ["0s", "10s", "20s", "30s", "40s", "50s", "60+"]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Converts parch into 3 categories (0, 1, 2+)
    df.loc[df['Parch'] > 2, 'Parch'] = 2

    # Converts sibling into 4 categories (0, 1, 2, 3+)
    df.loc[df['SibSp'] > 2, 'SibSp'] = 3

    # Add column for tittle (Mr, Ms, Miss, ect)
    titleType = {
        'Mr': 'Mr',
        'Don': 'Mr',
        'Miss': 'Miss',
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mrs': 'Mrs',
        'Mme': 'Mrs',
        'Master': 'Master',
        'Dr': 'Dr',
        'Rev': 'Other',
        'Col': 'Other',
        'Sir': 'Other',
        'Capt': 'Other',
        'Lady': 'Other',
        'the Countess': 'Other',
        'Jonkheer': 'Other',
        'Major': 'Other',
    }
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace(to_replace=titleType)

    # Subset the dataset into X and y
    input_X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title']]
    input_X = pd.get_dummies(input_X, columns=['Sex', 'Title'], drop_first=True)

    # Add missing expected columns
    input_X = input_X.reindex(columns=expected_columns, fill_value=0)

    if 'Survived' in df.columns:
        output_y = df['Survived']
        return input_X, output_y
    else:
        return input_X, None