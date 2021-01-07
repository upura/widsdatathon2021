import pandas as pd


if __name__ == '__main__':
    train = pd.read_csv('../input/TrainingWiDS2021.csv.zip')
    test = pd.read_csv('../input/UnlabeledWiDS2021.csv.zip')
    categorical_cols = [
        'hospital_id',
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        'icu_stay_type',
        'icu_type'
    ]
    numerical_cols = [c for c in train.columns if c not in categorical_cols]

    for c in categorical_cols:
        print(c)
        print(set(train[c]))
        print(set(test[c]))
        print(set(train[c]) & set(test[c]))
        print('==================')

    for n in numerical_cols:
        if len(train[n].unique()) == 1:
            print(n)
