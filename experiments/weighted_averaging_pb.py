import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from ayniy.utils import Data


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if run_id in ('run004',):
        oof = oof.reshape(-1, )
        pred = pred.reshape(-1, )
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


def f(x):
    pred = 0
    for i, d in enumerate(data):
        if i < len(x):
            pred += d[0] * x[i]
        else:
            pred += d[0] * (1 - sum(x))
    score = -1 * roc_auc_score(y_train, pred)
    Data.dump(pred, f'../output/pred/{run_name}-train.pkl')
    return score


def make_predictions(data: list, weights: list):
    pred = 0
    for i, d in enumerate(data):
        if i < len(weights):
            pred += d[1] * weights[i]
        else:
            pred += d[1] * (1 - sum(weights))
    Data.dump(pred, f'../output/pred/{run_name}-test.pkl')
    return pred


def make_submission(pred, run_name: str):
    sub = pd.read_csv('../input/SolutionTemplateWiDS2021.csv')
    sub['diabetes_mellitus'] = pred
    sub.to_csv(f'../output/submissions/submission_{run_name}.csv', index=False)


run_ids = [
    'run017',
    'run018',
    'run021',
    'run022',
    'run026',
    'run027',
    'run028',
    'run029',
    'run030',
    'run031',
]
run_name = 'weight017_pb'

if __name__ == '__main__':
    y_train = Data.load('../input/pickle/y_train_fe000.pkl')
    data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]

    for d in data:
        print(roc_auc_score(y_train, d[0]))

    init_state = [round(1 / len(data), 3) for _ in range(len(data) - 1)]
    result = minimize(f, init_state, method='Nelder-Mead')
    print('optimized CV: ', result['fun'])
    print('w: ', result['x'])
    upura_preds = make_predictions(data, result['x'])
    pb_preds = pd.read_csv('../output/submissions/submission_0872.csv')
    pb_preds = pb_preds.sort_values('encounter_id').reset_index(drop=True)
    pb_preds = pb_preds['diabetes_mellitus'].values
    preds = 0.5 * upura_preds + pb_preds * 0.5
    make_submission(preds, run_name)
