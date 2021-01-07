import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kaggle_utils.features import count_null, count_encoding, count_encoding_interact, target_encoding, matrix_factorization
from kaggle_utils.features.category_encoding import CategoricalEncoder
from kaggle_utils.features.groupby import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer


if __name__ == '__main__':
    train = pd.read_csv('../input/TrainingWiDS2021.csv.zip')
    test = pd.read_csv('../input/UnlabeledWiDS2021.csv.zip')
    test = test.sort_values('encounter_id').reset_index(drop=True)

    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(train.shape, test.shape)
    # (130157, 181) (10234, 180)
    categorical_cols = [
        'hospital_id',
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        'icu_stay_type',
        'icu_type',
        'icu_id'
    ]
    numerical_cols = [c for c in train.columns if c not in categorical_cols]
    target_col = 'diabetes_mellitus'

    # submit file
    sub = test[['encounter_id']].copy()
    sub['diabetes_mellitus'] = np.nan
    sub.to_csv('../input/sample_submission.csv', index=False)

    # drop unnecessary column
    train_test.drop([
        'Unnamed: 0',
        'readmission_status',
        'encounter_id'
    ], axis=1, inplace=True)

    # label encoding
    ce = CategoricalEncoder(categorical_cols)
    train_test = ce.transform(train_test)
    train_test.drop([
        'hospital_id'
    ], axis=1).to_feather('../input/feather/train_test.ftr')

    # count null
    count_null(train_test,
               train_test.drop(target_col, axis=1).columns).to_feather('../input/feather/count_null.ftr')

    # count encoding
    count_encoding(train_test, [
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        'icu_stay_type',
        'icu_type'
    ]).to_feather('../input/feather/count_encoding.ftr')
    count_encoding_interact(train_test, [
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        'icu_stay_type',
        'icu_type'
    ]).to_feather('../input/feather/count_encoding_interact.ftr')

    # target encoding
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    _train = train_test.dropna(subset=[target_col]).copy()
    _test = train_test.loc[train_test[target_col].isnull()].copy()
    target_encoding(_train, _test, [
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        # 'icu_stay_type',
        'icu_type',
        'icu_id'
    ], target_col, cv).to_feather('../input/feather/target_encoding.ftr')
