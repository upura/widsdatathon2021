from ayniy.utils import FeatureStore, Data


if __name__ == '__main__':

    target_col = 'diabetes_mellitus'

    features = FeatureStore(
        feature_names=[
            '../input/feather/train_test.ftr',
            '../input/feather/count_null.ftr',
            '../input/feather/count_encoding.ftr',
            "../input/feather/count_encoding_interact.ftr",
            # "../input/feather/features_lda.ftr",
            # "../input/feather/features_svd.ftr",
            "../input/feather/aggregation.ftr",
            # "../input/feather/aggregation2.ftr",  # by hospital_id
            "../input/feather/target_encoding10.ftr",
        ],
        target_col=target_col,
    )

    X_train = features.X_train
    y_train = features.y_train
    X_test = features.X_test

    print(X_train.shape)
    print(X_train.columns)

    fe_name = 'fe007'
    Data.dump(X_train, f'../input/pickle/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/pickle/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/pickle/X_test_{fe_name}.pkl')
