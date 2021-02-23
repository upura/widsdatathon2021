import pandas as pd

from ayniy.utils import FeatureStore, Data


if __name__ == '__main__':

    target_col = 'diabetes_mellitus'

    features = FeatureStore(
        feature_names=[
            '../input/feather/train_test.ftr',
            '../input/feather/count_null.ftr',
            '../input/feather/count_encoding.ftr',
            "../input/feather/count_encoding_interact.ftr",
            "../input/feather/aggregation.ftr",
            "../input/feather/target_encoding.ftr",
        ],
        target_col=target_col,
    )

    X_train_u = features.X_train
    y_train = features.y_train
    X_test_u = features.X_test

    fe_id_u = 'fe006'
    run_id = 'run021'
    N_FEATURES = 100

    X_train_u = Data.load(f'../input/pickle/X_train_{fe_id_u}.pkl')
    X_test_u = Data.load(f'../input/pickle/X_test_{fe_id_u}.pkl')
    fi = pd.read_csv(f'../output/importance/{run_id}-fi.csv')['Feature'][:N_FEATURES]
    X_train_u = X_train_u[fi]
    X_test_u = X_test_u[fi].reset_index(drop=True)
    X_train_u.columns = [f'u_{c}' for c in fi]
    X_test_u.columns = [f'u_{c}' for c in fi]

    fe_id = 'fe_siavrez'
    X_train = Data.load(f'../input/pickle/X_train_{fe_id}.pkl')
    X_test = Data.load(f'../input/pickle/X_test_{fe_id}.pkl')

    print(X_train.shape, X_train_u.shape)
    print(X_test.shape, X_test_u.shape)

    X_train = pd.concat([X_train, X_train_u], axis=1)
    X_test = pd.concat([X_test, X_test_u], axis=1)
    print(X_train.shape, X_test.shape)

    fe_name = 'fe008'
    Data.dump(X_train, f'../input/pickle/X_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/pickle/X_test_{fe_name}.pkl')
    Data.dump(y_train, f'../input/pickle/y_train_{fe_name}.pkl')
