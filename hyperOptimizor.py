import optuna
import lgbm
import numpy as np

class hyperOpt:
    def __init__(self, x_train, y_train):
        self.columns_to_try = [
            'glutamate_receptor_antagonist',
            'dna_inhibitor',
            'serotonin_receptor_antagonist',
            'dopamine_receptor_antagonist',
            'cyclooxygenase_inhibitor'
        ]
        self.x_train = x_train
        self.y_train = y_train

    def objective(self, trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'boost_from_average': True,
            'num_threads': 4,
            'random_state': 42,
            
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 1, 100),
            'bagging_fraction': trial.suggest_loguniform('bagging_fraction', .5, .99),
            'feature_fraction': trial.suggest_loguniform('feature_fraction', .5, .99),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.1, 2),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.1, 2)
        }
        
        model = lgbm.lightGBM()
        scores = []
        for column in self.columns_to_try:
            _, _, score = model.fit_predict(3, params, self.x_train, self.y_train[column], None)
            scores.append(score)

        return np.mean(scores)

    def optimize(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=100)
        return study.best_trial
