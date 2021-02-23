import gc

import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from ayniy.utils import Data


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if (c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)
    return df


if __name__ == '__main__':
    # https://www.kaggle.com/siavrez/2020fatures
    train = pd.read_csv('../input/TrainingWiDS2021.csv.zip', index_col=[0])
    test = pd.read_csv('../input/UnlabeledWiDS2021.csv.zip', index_col=[0])
    test_id = test.encounter_id.values
    y = train.diabetes_mellitus.values
    del train['diabetes_mellitus']

    train = train.rename(columns={'pao2_apache': 'pao2fio2ratio_apache',
                                  'ph_apache': 'arterial_ph_apache'})
    test = test.rename(columns={'pao2_apache': 'pao2fio2ratio_apache',
                                'ph_apache': 'arterial_ph_apache'})
    train.loc[train.age == 0, 'age'] = np.nan
    train = train.drop(['readmission_status', 'encounter_id', 'hospital_id'], axis=1)
    test = test.drop(['readmission_status', 'encounter_id', 'hospital_id'], axis=1)
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    min_max_feats = [f[:-4] for f in train.columns if f[-4:] == '_min']
    for col in min_max_feats:
        train.loc[
            train[f'{col}_min'] > train[f'{col}_max'], [f'{col}_min', f'{col}_max']
        ] = train.loc[train[f'{col}_min'] > train[f'{col}_max'], [f'{col}_max', f'{col}_min']].values
        test.loc[
            test[f'{col}_min'] > test[f'{col}_max'], [f'{col}_min', f'{col}_max']
        ] = test.loc[test[f'{col}_min'] > test[f'{col}_max'], [f'{col}_max', f'{col}_min']].values

    lbls = {}
    for col in train.select_dtypes(exclude=np.number).columns.tolist():
        le = LabelEncoder().fit(pd.concat([train[col].astype(str), test[col].astype(str)]))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        lbls[col] = le
    print('Categorical columns:', list(lbls.keys()))

    train['comorbidity_score'] = (
        train['aids'].values * 23
        + train['cirrhosis'] * 4
        + train['hepatic_failure'] * 16
        + train['immunosuppression'] * 10
        + train['leukemia'] * 10
        + train['lymphoma'] * 13
        + train['solid_tumor_with_metastasis'] * 11
    )
    test['comorbidity_score'] = (
        test['aids'].values * 23
        + test['cirrhosis'] * 4
        + test['hepatic_failure'] * 16
        + test['immunosuppression'] * 10
        + test['leukemia'] * 10
        + test['lymphoma'] * 13
        + test['solid_tumor_with_metastasis'] * 11
    )
    train['comorbidity_score'] = train['comorbidity_score'].fillna(0)
    test['comorbidity_score'] = test['comorbidity_score'].fillna(0)
    train['gcs_sum'] = train['gcs_eyes_apache'] + train['gcs_motor_apache'] + train['gcs_verbal_apache']
    test['gcs_sum'] = test['gcs_eyes_apache'] + test['gcs_motor_apache'] + test['gcs_verbal_apache']
    train['gcs_sum'] = train['gcs_sum'].fillna(0)
    test['gcs_sum'] = test['gcs_sum'].fillna(0)
    train['apache_2_diagnosis_type'] = train.apache_2_diagnosis.round(-1).fillna(-100).astype('int32')
    test['apache_2_diagnosis_type'] = test.apache_2_diagnosis.round(-1).fillna(-100).astype('int32')
    train['apache_3j_diagnosis_type'] = train.apache_3j_diagnosis.round(-2).fillna(-100).astype('int32')
    test['apache_3j_diagnosis_type'] = test.apache_3j_diagnosis.round(-2).fillna(-100).astype('int32')
    train['bmi_type'] = train.bmi.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    test['bmi_type'] = test.bmi.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    train['height_type'] = train.height.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    test['height_type'] = test.height.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    train['weight_type'] = train.weight.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    test['weight_type'] = test.weight.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    train['age_type'] = train.age.fillna(0).apply(lambda x: 10 * (round(int(x) / 10)))
    test['age_type'] = test.age.fillna(0).apply(lambda x: 10 * (round(int(x) / 10)))
    train['gcs_sum_type'] = train.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x) / 2.5))).divide(2.5)
    test['gcs_sum_type'] = test.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x) / 2.5))).divide(2.5)
    train['apache_3j_diagnosis_x'] = train['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    train['apache_2_diagnosis_x'] = train['apache_2_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    test['apache_3j_diagnosis_x'] = test['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    test['apache_2_diagnosis_x'] = test['apache_2_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    train['apache_3j_diagnosis_split1'] = np.where(
        train['apache_3j_diagnosis'].isna(), np.nan, train['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[1]
    )
    test['apache_3j_diagnosis_split1'] = np.where(
        test['apache_3j_diagnosis'].isna(), np.nan, test['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[1]
    )
    train['apache_2_diagnosis_split1'] = np.where(train['apache_2_diagnosis'].isna(), np.nan, train['apache_2_diagnosis'].apply(lambda x: x % 10))
    test['apache_2_diagnosis_split1'] = np.where(test['apache_2_diagnosis'].isna(), np.nan, test['apache_2_diagnosis'].apply(lambda x: x % 10))

    IDENTIFYING_COLS = ['age_type', 'height_type', 'ethnicity', 'gender', 'bmi_type']
    train['profile'] = train[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis=1)
    test['profile'] = test[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis=1)
    print(f'Number of unique Profiles : {train["profile"].nunique()}')

    df = pd.concat([train['icu_id'], test['icu_id']])
    agg = df.value_counts().to_dict()
    train['icu_id_counts'] = np.log1p(train['icu_id'].map(agg))
    test['icu_id_counts'] = np.log1p(test['icu_id'].map(agg))
    df = pd.concat([train['age'], test['age']])
    agg = df.value_counts().to_dict()
    train['age_counts'] = np.log1p(train['age'].map(agg))
    test['age_counts'] = np.log1p(test['age'].map(agg))
    train["diff_bmi"] = train['bmi'].copy()
    train['bmi'] = train['weight'] / ((train['height'] / 100)**2)
    train["diff_bmi"] = train["diff_bmi"] - train['bmi']
    test["diff_bmi"] = test['bmi'].copy()
    test['bmi'] = test['weight'] / ((test['height'] / 100)**2)
    test["diff_bmi"] = test["diff_bmi"] - test['bmi']
    train['pre_icu_los_days'] = train['pre_icu_los_days'].apply(lambda x: special.expit(x))
    test['pre_icu_los_days'] = test['pre_icu_los_days'].apply(lambda x: special.expit(x))
    train['abmi'] = train['age'] / train['bmi']
    train['agi'] = train['weight'] / train['age']
    test['abmi'] = test['age'] / train['bmi']
    test['agi'] = test['weight'] / train['age']

    d_cols = [c for c in train.columns if(c.startswith("d1"))]
    h_cols = [c for c in train.columns if(c.startswith("h1"))]
    train["dailyLabs_row_nan_count"] = train[d_cols].isna().sum(axis=1)
    train["hourlyLabs_row_nan_count"] = train[h_cols].isna().sum(axis=1)
    train["diff_labTestsRun_daily_hourly"] = train["dailyLabs_row_nan_count"] - train["hourlyLabs_row_nan_count"]
    test["dailyLabs_row_nan_count"] = test[d_cols].isna().sum(axis=1)
    test["hourlyLabs_row_nan_count"] = test[h_cols].isna().sum(axis=1)
    test["diff_labTestsRun_daily_hourly"] = test["dailyLabs_row_nan_count"] - test["hourlyLabs_row_nan_count"]

    lab_col = [c for c in train.columns if((c.startswith("h1")) | (c.startswith("d1")))]
    lab_col_names = list(set(list(map(lambda i: i[3: -4], lab_col))))

    print("len lab_col", len(lab_col))
    print("len lab_col_names", len(lab_col_names))
    print("lab_col_names\n", lab_col_names)

    first_h = []
    for v in lab_col_names:
        first_h.append(v + "_started_after_firstHour")
        colsx = [x for x in test.columns if v in x]
        train[v + "_nans"] = train.loc[:, colsx].isna().sum(axis=1)
        test[v + "_nans"] = test.loc[:, colsx].isna().sum(axis=1)
        train[v + "_d1_value_range"] = train[f"d1_{v}_max"].subtract(train[f"d1_{v}_min"])
        train[v + "_h1_value_range"] = train[f"h1_{v}_max"].subtract(train[f"h1_{v}_min"])
        train[v + "_d1_h1_max_eq"] = (train[f"d1_{v}_max"] == train[f"h1_{v}_max"]).astype(np.int8)
        train[v + "_d1_h1_min_eq"] = (train[f"d1_{v}_min"] == train[f"h1_{v}_min"]).astype(np.int8)
        train[v + "_d1_zero_range"] = (train[v + "_d1_value_range"] == 0).astype(np.int8)
        train[v + "_h1_zero_range"] = (train[v + "_h1_value_range"] == 0).astype(np.int8)
        train[v + "_tot_change_value_range_normed"] = abs((train[v + "_d1_value_range"].div(train[v + "_h1_value_range"])))
        train[v + "_started_after_firstHour"] = ((train[f"h1_{v}_max"].isna()) & (train[f"h1_{v}_min"].isna())) & (~train[f"d1_{v}_max"].isna())
        train[v + "_day_more_extreme"] = ((train[f"d1_{v}_max"] > train[f"h1_{v}_max"]) | (train[f"d1_{v}_min"] < train[f"h1_{v}_min"]))
        train[v + "_day_more_extreme"].fillna(False)
        test[v + "_d1_value_range"] = test[f"d1_{v}_max"].subtract(test[f"d1_{v}_min"])
        test[v + "_h1_value_range"] = test[f"h1_{v}_max"].subtract(test[f"h1_{v}_min"])
        test[v + "_d1_h1_max_eq"] = (test[f"d1_{v}_max"] == test[f"h1_{v}_max"]).astype(np.int8)
        test[v + "_d1_h1_min_eq"] = (test[f"d1_{v}_min"] == test[f"h1_{v}_min"]).astype(np.int8)
        test[v + "_d1_zero_range"] = (test[v + "_d1_value_range"] == 0).astype(np.int8)
        test[v + "_h1_zero_range"] = (test[v + "_h1_value_range"] == 0).astype(np.int8)
        test[v + "_tot_change_value_range_normed"] = abs((test[v + "_d1_value_range"].div(test[v + "_h1_value_range"])))
        test[v + "_started_after_firstHour"] = ((test[f"h1_{v}_max"].isna()) & (test[f"h1_{v}_min"].isna())) & (~test[f"d1_{v}_max"].isna())
        test[v + "_day_more_extreme"] = ((test[f"d1_{v}_max"] > test[f"h1_{v}_max"]) | (test[f"d1_{v}_min"] < test[f"h1_{v}_min"]))
        test[v + "_day_more_extreme"].fillna(False)

    train["total_Tests_started_After_firstHour"] = train[first_h].sum(axis=1)
    test["total_Tests_started_After_firstHour"] = test[first_h].sum(axis=1)
    gc.collect()
    print(train["total_Tests_started_After_firstHour"].describe())

    groupers = ['apache_3j_diagnosis', 'profile']
    for g in groupers:
        for v in lab_col_names:
            temp = pd.concat([train[[f"d1_{v}_max", g]], test[[f"d1_{v}_max", g]]], axis=0).groupby(g)[f"d1_{v}_max"].mean().to_dict()
            train[f'mean_diff_d1_{v}_{g}_max'] = train[f"d1_{v}_max"] - train[g].map(temp)
            test[f'mean_diff_d1_{v}_{g}_max'] = test[f"d1_{v}_max"] - test[g].map(temp)
            temp = pd.concat([train[[f"d1_{v}_min", g]], test[[f"d1_{v}_min", g]]], axis=0).groupby(g)[f"d1_{v}_min"].mean().to_dict()
            train[f'mean_diff_d1_{v}_{g}_min'] = train[f"d1_{v}_min"] - train[g].map(temp)
            test[f'mean_diff_d1_{v}_{g}_min'] = test[f"d1_{v}_min"] - test[g].map(temp)
            temp = pd.concat([train[[f"h1_{v}_max", g]], test[[f"h1_{v}_max", g]]], axis=0).groupby(g)[f"h1_{v}_max"].mean().to_dict()
            train[f'mean_diff_h1_{v}_{g}_max'] = train[f"h1_{v}_max"] - train[g].map(temp)
            test[f'mean_diff_h1_{v}_{g}_max'] = test[f"h1_{v}_max"] - test[g].map(temp)
            temp = pd.concat([train[[f"h1_{v}_min", g]], test[[f"h1_{v}_min", g]]], axis=0).groupby(g)[f"h1_{v}_min"].mean().to_dict()
            train[f'mean_diff_h1_{v}_{g}_min'] = train[f"h1_{v}_min"] - train[g].map(temp)
            test[f'mean_diff_h1_{v}_{g}_min'] = test[f"h1_{v}_min"] - test[g].map(temp)
    gc.collect()

    train['diasbp_indicator'] = (
        (train['d1_diasbp_invasive_max'] == train['d1_diasbp_max']) & (train['d1_diasbp_noninvasive_max'] == train['d1_diasbp_invasive_max'])
        | (train['d1_diasbp_invasive_min'] == train['d1_diasbp_min']) & (train['d1_diasbp_noninvasive_min'] == train['d1_diasbp_invasive_min'])
        | (train['h1_diasbp_invasive_max'] == train['h1_diasbp_max']) & (train['h1_diasbp_noninvasive_max'] == train['h1_diasbp_invasive_max'])
        | (train['h1_diasbp_invasive_min'] == train['h1_diasbp_min']) & (train['h1_diasbp_noninvasive_min'] == train['h1_diasbp_invasive_min'])
    ).astype(np.int8)

    train['mbp_indicator'] = (
        (train['d1_mbp_invasive_max'] == train['d1_mbp_max']) & (train['d1_mbp_noninvasive_max'] == train['d1_mbp_invasive_max'])
        | (train['d1_mbp_invasive_min'] == train['d1_mbp_min']) & (train['d1_mbp_noninvasive_min'] == train['d1_mbp_invasive_min'])
        | (train['h1_mbp_invasive_max'] == train['h1_mbp_max']) & (train['h1_mbp_noninvasive_max'] == train['h1_mbp_invasive_max'])
        | (train['h1_mbp_invasive_min'] == train['h1_mbp_min']) & (train['h1_mbp_noninvasive_min'] == train['h1_mbp_invasive_min'])
    ).astype(np.int8)

    train['sysbp_indicator'] = (
        (train['d1_sysbp_invasive_max'] == train['d1_sysbp_max']) & (train['d1_sysbp_noninvasive_max'] == train['d1_sysbp_invasive_max'])
        | (train['d1_sysbp_invasive_min'] == train['d1_sysbp_min']) & (train['d1_sysbp_noninvasive_min'] == train['d1_sysbp_invasive_min'])
        | (train['h1_sysbp_invasive_max'] == train['h1_sysbp_max']) & (train['h1_sysbp_noninvasive_max'] == train['h1_sysbp_invasive_max'])
        | (train['h1_sysbp_invasive_min'] == train['h1_sysbp_min']) & (train['h1_sysbp_noninvasive_min'] == train['h1_sysbp_invasive_min'])
    ).astype(np.int8)

    train['d1_mbp_invnoninv_max_diff'] = train['d1_mbp_invasive_max'] - train['d1_mbp_noninvasive_max']
    train['h1_mbp_invnoninv_max_diff'] = train['h1_mbp_invasive_max'] - train['h1_mbp_noninvasive_max']
    train['d1_mbp_invnoninv_min_diff'] = train['d1_mbp_invasive_min'] - train['d1_mbp_noninvasive_min']
    train['h1_mbp_invnoninv_min_diff'] = train['h1_mbp_invasive_min'] - train['h1_mbp_noninvasive_min']
    train['d1_diasbp_invnoninv_max_diff'] = train['d1_diasbp_invasive_max'] - train['d1_diasbp_noninvasive_max']
    train['h1_diasbp_invnoninv_max_diff'] = train['h1_diasbp_invasive_max'] - train['h1_diasbp_noninvasive_max']
    train['d1_diasbp_invnoninv_min_diff'] = train['d1_diasbp_invasive_min'] - train['d1_diasbp_noninvasive_min']
    train['h1_diasbp_invnoninv_min_diff'] = train['h1_diasbp_invasive_min'] - train['h1_diasbp_noninvasive_min']
    train['d1_sysbp_invnoninv_max_diff'] = train['d1_sysbp_invasive_max'] - train['d1_sysbp_noninvasive_max']
    train['h1_sysbp_invnoninv_max_diff'] = train['h1_sysbp_invasive_max'] - train['h1_sysbp_noninvasive_max']
    train['d1_sysbp_invnoninv_min_diff'] = train['d1_sysbp_invasive_min'] - train['d1_sysbp_noninvasive_min']
    train['h1_sysbp_invnoninv_min_diff'] = train['h1_sysbp_invasive_min'] - train['h1_sysbp_noninvasive_min']

    test['diasbp_indicator'] = (
        (test['d1_diasbp_invasive_max'] == test['d1_diasbp_max']) & (test['d1_diasbp_noninvasive_max'] == test['d1_diasbp_invasive_max'])
        | (test['d1_diasbp_invasive_min'] == test['d1_diasbp_min']) & (test['d1_diasbp_noninvasive_min'] == test['d1_diasbp_invasive_min'])
        | (test['h1_diasbp_invasive_max'] == test['h1_diasbp_max']) & (test['h1_diasbp_noninvasive_max'] == test['h1_diasbp_invasive_max'])
        | (test['h1_diasbp_invasive_min'] == test['h1_diasbp_min']) & (test['h1_diasbp_noninvasive_min'] == test['h1_diasbp_invasive_min'])
    ).astype(np.int8)

    test['mbp_indicator'] = (
        (test['d1_mbp_invasive_max'] == test['d1_mbp_max']) & (test['d1_mbp_noninvasive_max'] == test['d1_mbp_invasive_max'])
        | (test['d1_mbp_invasive_min'] == test['d1_mbp_min']) & (test['d1_mbp_noninvasive_min'] == test['d1_mbp_invasive_min'])
        | (test['h1_mbp_invasive_max'] == test['h1_mbp_max']) & (test['h1_mbp_noninvasive_max'] == test['h1_mbp_invasive_max'])
        | (test['h1_mbp_invasive_min'] == test['h1_mbp_min']) & (test['h1_mbp_noninvasive_min'] == test['h1_mbp_invasive_min'])
    ).astype(np.int8)

    test['sysbp_indicator'] = (
        (test['d1_sysbp_invasive_max'] == test['d1_sysbp_max']) & (test['d1_sysbp_noninvasive_max'] == test['d1_sysbp_invasive_max'])
        | (test['d1_sysbp_invasive_min'] == test['d1_sysbp_min']) & (test['d1_sysbp_noninvasive_min'] == test['d1_sysbp_invasive_min'])
        | (test['h1_sysbp_invasive_max'] == test['h1_sysbp_max']) & (test['h1_sysbp_noninvasive_max'] == test['h1_sysbp_invasive_max'])
        | (test['h1_sysbp_invasive_min'] == test['h1_sysbp_min']) & (test['h1_sysbp_noninvasive_min'] == test['h1_sysbp_invasive_min'])
    ).astype(np.int8)

    test['d1_mbp_invnoninv_max_diff'] = test['d1_mbp_invasive_max'] - test['d1_mbp_noninvasive_max']
    test['h1_mbp_invnoninv_max_diff'] = test['h1_mbp_invasive_max'] - test['h1_mbp_noninvasive_max']
    test['d1_mbp_invnoninv_min_diff'] = test['d1_mbp_invasive_min'] - test['d1_mbp_noninvasive_min']
    test['h1_mbp_invnoninv_min_diff'] = test['h1_mbp_invasive_min'] - test['h1_mbp_noninvasive_min']
    test['d1_diasbp_invnoninv_max_diff'] = test['d1_diasbp_invasive_max'] - test['d1_diasbp_noninvasive_max']
    test['h1_diasbp_invnoninv_max_diff'] = test['h1_diasbp_invasive_max'] - test['h1_diasbp_noninvasive_max']
    test['d1_diasbp_invnoninv_min_diff'] = test['d1_diasbp_invasive_min'] - test['d1_diasbp_noninvasive_min']
    test['h1_diasbp_invnoninv_min_diff'] = test['h1_diasbp_invasive_min'] - test['h1_diasbp_noninvasive_min']

    test['d1_sysbp_invnoninv_max_diff'] = test['d1_sysbp_invasive_max'] - test['d1_sysbp_noninvasive_max']
    test['h1_sysbp_invnoninv_max_diff'] = test['h1_sysbp_invasive_max'] - test['h1_sysbp_noninvasive_max']
    test['d1_sysbp_invnoninv_min_diff'] = test['d1_sysbp_invasive_min'] - test['d1_sysbp_noninvasive_min']
    test['h1_sysbp_invnoninv_min_diff'] = test['h1_sysbp_invasive_min'] - test['h1_sysbp_noninvasive_min']

    for v in ['albumin', 'bilirubin', 'bun', 'glucose', 'hematocrit', 'pao2fio2ratio', 'arterial_ph', 'resprate', 'sodium', 'temp', 'wbc', 'creatinine']:
        train[f'{v}_indicator'] = (((train[f'{v}_apache'] == train[f'd1_{v}_max']) & (train[f'd1_{v}_max'] == train[f'h1_{v}_max']))
                                   | ((train[f'{v}_apache'] == train[f'd1_{v}_max']) & (train[f'd1_{v}_max'] == train[f'd1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'd1_{v}_max']) & (train[f'd1_{v}_max'] == train[f'h1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_max']) & (train[f'h1_{v}_max'] == train[f'd1_{v}_max']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_max']) & (train[f'h1_{v}_max'] == train[f'h1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_max']) & (train[f'h1_{v}_max'] == train[f'd1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'd1_{v}_min']) & (train[f'd1_{v}_min'] == train[f'd1_{v}_max']))
                                   | ((train[f'{v}_apache'] == train[f'd1_{v}_min']) & (train[f'd1_{v}_min'] == train[f'h1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'd1_{v}_min']) & (train[f'd1_{v}_min'] == train[f'h1_{v}_max']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_min']) & (train[f'h1_{v}_min'] == train[f'h1_{v}_max']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_min']) & (train[f'h1_{v}_min'] == train[f'd1_{v}_min']))
                                   | ((train[f'{v}_apache'] == train[f'h1_{v}_min']) & (train[f'h1_{v}_min'] == train[f'd1_{v}_max']))
                                   ).astype(np.int8)
        test[f'{v}_indicator'] = (((test[f'{v}_apache'] == test[f'd1_{v}_max']) & (test[f'd1_{v}_max'] == test[f'h1_{v}_max']))
                                  | ((test[f'{v}_apache'] == test[f'd1_{v}_max']) & (test[f'd1_{v}_max'] == test[f'd1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'd1_{v}_max']) & (test[f'd1_{v}_max'] == test[f'h1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_max']) & (test[f'h1_{v}_max'] == test[f'd1_{v}_max']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_max']) & (test[f'h1_{v}_max'] == test[f'h1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_max']) & (test[f'h1_{v}_max'] == test[f'd1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'd1_{v}_min']) & (test[f'd1_{v}_min'] == test[f'd1_{v}_max']))
                                  | ((test[f'{v}_apache'] == test[f'd1_{v}_min']) & (test[f'd1_{v}_min'] == test[f'h1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'd1_{v}_min']) & (test[f'd1_{v}_min'] == test[f'h1_{v}_max']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_min']) & (test[f'h1_{v}_min'] == test[f'h1_{v}_max']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_min']) & (test[f'h1_{v}_min'] == test[f'd1_{v}_min']))
                                  | ((test[f'{v}_apache'] == test[f'h1_{v}_min']) & (test[f'h1_{v}_min'] == test[f'd1_{v}_max']))
                                  ).astype(np.int8)

    more_extreme_cols = [c for c in train.columns if(c.endswith("_day_more_extreme"))]
    train["total_day_more_extreme"] = train[more_extreme_cols].sum(axis=1)
    test["total_day_more_extreme"] = test[more_extreme_cols].sum(axis=1)
    train["d1_resprate_div_mbp_min"] = train["d1_resprate_min"].div(train["d1_mbp_min"])
    train["d1_resprate_div_sysbp_min"] = train["d1_resprate_min"].div(train["d1_sysbp_min"])
    train["d1_lactate_min_div_diasbp_min"] = train["d1_lactate_min"].div(train["d1_diasbp_min"])
    train["d1_heartrate_min_div_d1_sysbp_min"] = train["d1_heartrate_min"].div(train["d1_sysbp_min"])
    train["d1_hco3_div"] = train["d1_hco3_max"].div(train["d1_hco3_min"])
    train["d1_resprate_times_resprate"] = train["d1_resprate_min"].multiply(train["d1_resprate_max"])
    train["left_average_spo2"] = (2 * train["d1_spo2_max"] + train["d1_spo2_min"]) / 3
    test["d1_resprate_div_mbp_min"] = test["d1_resprate_min"].div(test["d1_mbp_min"])
    test["d1_resprate_div_sysbp_min"] = test["d1_resprate_min"].div(test["d1_sysbp_min"])
    test["d1_lactate_min_div_diasbp_min"] = test["d1_lactate_min"].div(test["d1_diasbp_min"])
    test["d1_heartrate_min_div_d1_sysbp_min"] = test["d1_heartrate_min"].div(test["d1_sysbp_min"])
    test["d1_hco3_div"] = test["d1_hco3_max"].div(test["d1_hco3_min"])
    test["d1_resprate_times_resprate"] = test["d1_resprate_min"].multiply(test["d1_resprate_max"])
    test["left_average_spo2"] = (2 * test["d1_spo2_max"] + test["d1_spo2_min"]) / 3
    train["total_chronic"] = train[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    train["total_cancer_immuno"] = train[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
    test["total_chronic"] = test[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    test["total_cancer_immuno"] = test[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
    train["has_complicator"] = train[["aids", "cirrhosis", 'hepatic_failure',
                                      'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)
    test["has_complicator"] = test[["aids", "cirrhosis", 'hepatic_failure',
                                    'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)
    gc.collect()
    print(train[["has_complicator", "total_chronic", "total_cancer_immuno", "has_complicator"]].describe())

    more_extreme_cols = [c for c in train.columns if(c.endswith("_day_more_extreme"))]
    train["total_day_more_extreme"] = train[more_extreme_cols].sum(axis=1)
    test["total_day_more_extreme"] = test[more_extreme_cols].sum(axis=1)
    train["d1_resprate_div_mbp_min"] = train["d1_resprate_min"].div(train["d1_mbp_min"])
    train["d1_resprate_div_sysbp_min"] = train["d1_resprate_min"].div(train["d1_sysbp_min"])
    train["d1_lactate_min_div_diasbp_min"] = train["d1_lactate_min"].div(train["d1_diasbp_min"])
    train["d1_heartrate_min_div_d1_sysbp_min"] = train["d1_heartrate_min"].div(train["d1_sysbp_min"])
    train["d1_hco3_div"] = train["d1_hco3_max"].div(train["d1_hco3_min"])
    train["d1_resprate_times_resprate"] = train["d1_resprate_min"].multiply(train["d1_resprate_max"])
    train["left_average_spo2"] = (2 * train["d1_spo2_max"] + train["d1_spo2_min"]) / 3
    test["d1_resprate_div_mbp_min"] = test["d1_resprate_min"].div(test["d1_mbp_min"])
    test["d1_resprate_div_sysbp_min"] = test["d1_resprate_min"].div(test["d1_sysbp_min"])
    test["d1_lactate_min_div_diasbp_min"] = test["d1_lactate_min"].div(test["d1_diasbp_min"])
    test["d1_heartrate_min_div_d1_sysbp_min"] = test["d1_heartrate_min"].div(test["d1_sysbp_min"])
    test["d1_hco3_div"] = test["d1_hco3_max"].div(test["d1_hco3_min"])
    test["d1_resprate_times_resprate"] = test["d1_resprate_min"].multiply(test["d1_resprate_max"])
    test["left_average_spo2"] = (2 * test["d1_spo2_max"] + test["d1_spo2_min"]) / 3
    train["total_chronic"] = train[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    train["total_cancer_immuno"] = train[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
    test["total_chronic"] = test[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    test["total_cancer_immuno"] = test[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
    train["has_complicator"] = train[["aids", "cirrhosis", 'hepatic_failure',
                                      'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)
    test["has_complicator"] = test[["aids", "cirrhosis", 'hepatic_failure',
                                    'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)
    gc.collect()
    print(train[["has_complicator", "total_chronic", "total_cancer_immuno", "has_complicator"]].describe())

    trainf = pd.read_pickle('../input/featxx/X.pkl.zip')
    testf = pd.read_pickle('../input/featxx/X_test.pkl.zip')
    trainf = trainf.rename(columns={'pao2_apache': 'pao2fio2ratio_apache', 'ph_apache': 'arterial_ph_apache'})
    testf = testf.rename(columns={'pao2_apache': 'pao2fio2ratio_apache', 'ph_apache': 'arterial_ph_apache'})
    col_order = train.columns.tolist()
    train = train[col_order]
    test = test[col_order]
    col_order = trainf.columns.tolist()
    trainf = trainf[col_order]
    testf = testf[col_order]

    train = pd.concat([trainf.reset_index(drop=True), train.reset_index(drop=True)], axis=1)
    test = pd.concat([testf.reset_index(drop=True), test.reset_index(drop=True)], axis=1)
    train = train.fillna(0)
    test = test.fillna(0)
    gc.collect()
    print(train.shape, test.shape)

    Cols = list(train.columns)
    for i, item in enumerate(train.columns):
        if item in train.columns[:i]:
            Cols[i] = "toDROP"
    train.columns = Cols
    test.columns = Cols
    train = train.drop("toDROP", 1)
    test = test.drop("toDROP", 1)
    print(train.shape, test.shape)
    col_order = train.columns.tolist()
    train = train[col_order]
    test = test[col_order]

    drop_cols = ['abmi', 'age_type', 'aids', 'albumin_apache', 'albumin_h1_value_range',
                 'albumin_h1_zero_range', 'albumin_tot_change_value_range_normed', 'apache_3j_diagnosis-cat_age',
                 'apache_post_operative', 'apache_post_operative_std_d1_temp_max', 'arf_apache_std_d1_hemaglobin_max',
                 'arterial_pco2_d1_h1_max_eq', 'arterial_pco2_d1_h1_min_eq', 'arterial_pco2_d1_zero_range',
                 'arterial_pco2_h1_zero_range', 'arterial_ph_apache', 'arterial_ph_d1_h1_max_eq', 'arterial_ph_d1_value_range',
                 'arterial_ph_d1_zero_range', 'arterial_ph_h1_zero_range', 'arterial_po2_d1_h1_max_eq', 'arterial_po2_d1_h1_min_eq',
                 'arterial_po2_d1_value_range', 'bilirubin_h1_value_range', 'bilirubin_h1_zero_range',
                 'bilirubin_tot_change_value_range_normed', 'bmi_type', 'bun_d1_h1_max_eq', 'bun_d1_zero_range',
                 'bun_h1_value_range', 'bun_h1_zero_range', 'calcium_d1_zero_range', 'calcium_h1_value_range',
                 'calcium_h1_zero_range', 'creatinine_h1_zero_range', 'd1_albumin_min', 'd1_arterial_pco2_min',
                 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_calcium_max', 'd1_diasbp_max', 'd1_diasbp_min',
                 'd1_hematocrit_min', 'd1_inr_max', 'd1_inr_min', 'd1_mbp_invasive_max', 'd1_mbp_invasive_min',
                 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_pao2fio2ratio_max',
                 'd1_pao2fio2ratio_min', 'd1_platelets_max', 'd1_resprate_max', 'd1_sysbp_invasive_min', 'd1_temp_min',
                 'd1_wbc_min', 'diasbp_d1_h1_max_eq', 'diasbp_d1_zero_range', 'diasbp_invasive_d1_h1_max_eq',
                 'diasbp_invasive_d1_value_range', 'diasbp_invasive_d1_zero_range', 'diasbp_invasive_h1_value_range',
                 'diasbp_invasive_h1_zero_range', 'diasbp_noninvasive_d1_h1_max_eq', 'diasbp_noninvasive_d1_zero_range',
                 'diasbp_noninvasive_h1_zero_range', 'diff_bmi', 'elective_surgery_mean_d1_sysbp_min',
                 'gcs_unable_apache', 'h1_albumin_max', 'h1_albumin_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min',
                 'h1_arterial_ph_min', 'h1_arterial_po2_max', 'h1_bilirubin_max', 'h1_bun_max', 'h1_creatinine_min',
                 'h1_diasbp_noninvasive_max', 'h1_heartrate_max', 'h1_heartrate_min', 'h1_hemaglobin_min',
                 'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_lactate_max', 'h1_lactate_min', 'h1_mbp_invasive_max',
                 'h1_mbp_invasive_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max',
                 'h1_mbp_noninvasive_min', 'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min', 'h1_platelets_max',
                 'h1_platelets_min', 'h1_resprate_max', 'h1_resprate_min', 'h1_sodium_max', 'h1_spo2_max', 'h1_spo2_min',
                 'h1_sysbp_max', 'h1_sysbp_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min', 'h1_temp_max',
                 'h1_temp_min', 'h1_wbc_max', 'h1_wbc_min', 'hco3_d1_h1_max_eq', 'hco3_d1_h1_min_eq',
                 'hco3_h1_value_range', 'hco3_h1_zero_range', 'heartrate_d1_zero_range', 'heartrate_h1_zero_range',
                 'height', 'hemaglobin_d1_value_range', 'hemaglobin_d1_zero_range', 'hematocrit_apache',
                 'hematocrit_d1_h1_min_eq', 'hematocrit_d1_value_range', 'hematocrit_d1_zero_range', 'inr_d1_h1_max_eq',
                 'inr_d1_h1_min_eq', 'inr_d1_value_range', 'inr_d1_zero_range', 'inr_day_more_extreme',
                 'inr_h1_value_range', 'inr_h1_zero_range', 'inr_started_after_firstHour',
                 'intubated_apache_mean_d1_spo2_max', 'lactate_h1_value_range', 'lactate_h1_zero_range',
                 'lymphoma', 'map_apache', 'mbp_d1_zero_range', 'mbp_h1_zero_range', 'mbp_invasive_d1_h1_min_eq',
                 'mbp_invasive_d1_value_range', 'mbp_invasive_d1_zero_range', 'mbp_invasive_h1_zero_range',
                 'mbp_noninvasive_d1_h1_max_eq', 'mbp_noninvasive_d1_h1_min_eq', 'mbp_noninvasive_d1_zero_range',
                 'mbp_noninvasive_h1_zero_range', 'mean_diff_d1_inr_min', 'mean_diff_h1_bilirubin_min',
                 'mean_diff_h1_inr_max', 'paco2_apache', 'paco2_for_ph_apache', 'pao2fio2ratio_apache',
                 'pao2fio2ratio_h1_value_range', 'pao2fio2ratio_h1_zero_range', 'rank_frqenc_leukemia',
                 'wbc_h1_value_range', 'platelets_d1_value_range', 'platelets_h1_zero_range', 'potassium_d1_h1_max_eq',
                 'potassium_h1_value_range', 'potassium_h1_zero_range', 'rank_frqenc_apache_2_diagnosis',
                 'resprate_apache', 'resprate_d1_h1_min_eq', 'resprate_d1_zero_range', 'sodium_d1_h1_min_eq',
                 'sodium_d1_zero_range', 'spo2_d1_h1_max_eq', 'sysbp_d1_zero_range', 'sysbp_h1_zero_range',
                 'sysbp_invasive_d1_h1_min_eq', 'sysbp_invasive_d1_zero_range', 'sysbp_noninvasive_d1_h1_min_eq',
                 'sysbp_noninvasive_d1_zero_range', 'sysbp_noninvasive_h1_zero_range', 'temp_d1_zero_range',
                 'ventilated_apache_std_d1_glucose_min', 'wbc_apache', 'wbc_d1_h1_min_eq', 'wbc_d1_value_range',
                 'wbc_d1_zero_range', 'wbc_h1_zero_range', 'gcs_eyes_apache_mean_d1_bun_min', 'rank_frqenc_aids']
    drop_cols = list(set(drop_cols))
    print(len(drop_cols))
    cats = ['elective_surgery', 'icu_id', 'arf_apache', 'intubated_apache', 'ventilated_apache', 'cirrhosis',
            'hepatic_failure', 'immunosuppression', 'leukemia', 'solid_tumor_with_metastasis', 'apache_3j_diagnosis_x',
            'apache_2_diagnosis_x', 'apache_3j_diagnosis_split1',
            'apache_2_diagnosis_split1', 'gcs_sum_type', 'hospital_admit_source', 'glucose_rate',
            'glucose_wb', 'gcs_eyes_apache', 'glucose_normal', 'total_cancer_immuno', 'gender',
            'total_chronic', 'icu_stay_type', 'apache_2_diagnosis_type', 'apache_3j_diagnosis_type']
    print(len(cats))
    for col in cats:
        train_only = list(set(train[col].unique()) - set(test[col].unique()))
        test_only = list(set(test[col].unique()) - set(train[col].unique()))
        both = list(set(test[col].unique()).union(set(train[col].unique())))
        train.loc[train[col].isin(train_only), col] = np.nan
        test.loc[test[col].isin(test_only), col] = np.nan
        try:
            lbl = OrdinalEncoder(dtype='int')
            train[col] = lbl.fit_transform(train[col].astype('str').values.reshape(-1, 1))
            test[col] = lbl.transform(test[col].astype('str').values.reshape(-1, 1))
        except:
            lbl = OrdinalEncoder(dtype='int')
            train[col] = lbl.fit_transform(train[col].astype('str').fillna('-1').values.reshape(-1, 1))
            test[col] = lbl.transform(test[col].astype('str').fillna('-1').values.reshape(-1, 1))
        temp = pd.concat([train[[col]], test[[col]]], axis=0)
        temp_mapping = temp.groupby(col).size() / len(temp)
        temp['enc'] = temp[col].map(temp_mapping)
        temp['enc'] = stats.rankdata(temp['enc'])
        temp = temp.reset_index(drop=True)
        train[f'rank_frqenc_{col}'] = temp[['enc']].values[:train.shape[0]]
        test[f'rank_frqenc_{col}'] = temp[['enc']].values[train.shape[0]:]
        test[col] = test[col].astype('category')
        train[col] = train[col].astype('category')

    drop_cols = list(set(drop_cols))
    print(len(drop_cols))
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    gc.collect()
    print(train.shape, test.shape)

    test['encounter_id'] = test_id
    test = test.sort_values('encounter_id').reset_index(drop=True)

    fe_name = 'fe_siavrez'
    Data.dump(train, f'../input/pickle/X_train_{fe_name}.pkl')
    # Data.dump(y, f'../input/pickle/y_train_{fe_name}.pkl')
    Data.dump(test.drop('encounter_id', axis=1), f'../input/pickle/X_test_{fe_name}.pkl')
