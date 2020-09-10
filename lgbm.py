from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss

import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hyperOptimizor

class lightGBM:
    def __init__(self):
        self.columns_scored = [
            '5-alpha_reductase_inhibitor',
            '11-beta-hsd1_inhibitor',
            'adenylyl_cyclase_activator',
            'aldehyde_dehydrogenase_inhibitor',
            'ampk_activator',
            'analgesic',
            'antiarrhythmic',
            'anticonvulsant',
            'antifungal',
            'antihistamine',
            'antimalarial',
            'antiviral',
            'atm_kinase_inhibitor',
            'atp-sensitive_potassium_channel_antagonist',
            'atp_synthase_inhibitor',
            'atr_kinase_inhibitor',
            'autotaxin_inhibitor',
            'bacterial_membrane_integrity_inhibitor',
            'calcineurin_inhibitor',
            'caspase_activator',
            'catechol_o_methyltransferase_inhibitor',
            'cck_receptor_antagonist',
            'chk_inhibitor',
            'coagulation_factor_inhibitor',
            'diuretic',
            'elastase_inhibitor',
            'erbb2_inhibitor',
            'farnesyltransferase_inhibitor',
            'focal_adhesion_kinase_inhibitor',
            'free_radical_scavenger',
            'fungal_squalene_epoxidase_inhibitor',
            'glutamate_inhibitor',
            'gonadotropin_receptor_agonist',
            'histone_lysine_demethylase_inhibitor',
            'hsp_inhibitor',
            'ikk_inhibitor',
            'laxative',
            'leukotriene_inhibitor',
            'lipase_inhibitor',
            'lxr_agonist',
            'mdm_inhibitor',
            'monoacylglycerol_lipase_inhibitor',
            'monopolar_spindle_1_kinase_inhibitor',
            'nicotinic_receptor_agonist',
            'nitric_oxide_production_inhibitor',
            'norepinephrine_reuptake_inhibitor',
            'nrf2_activator',
            'pdk_inhibitor',
            'progesterone_receptor_antagonist',
            'proteasome_inhibitor',
            'protein_phosphatase_inhibitor',
            'protein_tyrosine_kinase_inhibitor',
            'ras_gtpase_inhibitor',
            'retinoid_receptor_antagonist',
            'steroid',
            'syk_inhibitor',
            'tgf-beta_receptor_inhibitor',
            'thrombin_inhibitor',
            'tlr_antagonist',
            'transient_receptor_potential_channel_antagonist',
            'tropomyosin_receptor_kinase_inhibitor',
            'trpv_agonist',
            'ubiquitin_specific_protease_inhibitor',
            'vitamin_d_receptor_agonist'
        ]
        self.params_1 = {}
        self.params_2 = {}

    def fit_predict(self, n_splits, params, x_train, y_train, x_test):
        # out-of-fold predictions
        oof = np.zeros(x_train.shape[0])
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_preds = []

        for train_idx, valid_idx in cv.split(x_train, y_train):
            x_train_train = x_train.iloc[train_idx]
            y_train_train = y_train.iloc[train_idx]
            x_train_valid = x_train.iloc[valid_idx]
            y_train_valid = y_train.iloc[valid_idx]

            lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))
            lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

            estimator = lgb.train(params, lgb_train, 10000, valid_sets=lgb_valid, early_stopping_rounds=25, verbose_eval=0)

            oof_part = estimator.predict(x_train_valid, num_iteration=estimator.best_iteration)
            oof[valid_idx] = oof_part
            
            if x_test is not None:
                y_part = estimator.predict(x_test, num_iteration=estimator.best_iteration)
                y_preds.append(y_part)
            
        score = log_loss(y_train, oof)
        print('LogLoss Score:', score)
        
        y_pred = np.mean(y_preds, axis=0)
        
        return y_pred, oof, score

    def set_hyper_param(self, useDefault, x_train=[], y_train=[]):
        if useDefault:
            self.params_1 = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'boost_from_average': True,
                'num_threads': 4,
                'random_state': 42,
                'learning_rate': 0.01,
                'verbose': -1,
                
                # from Optuna result in Version 7
                'num_leaves': 212,
                'min_data_in_leaf': 92,
                'min_child_weight': 0.0010123391323415569,
                'max_depth': 35,
                'bagging_fraction': 0.7968351296815959,
                'feature_fraction': 0.7556374471450119,
                'lambda_l1': 0.23497601594060086,
                'lambda_l2': 0.15889208239516134
            }

            self.params_2 = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'boost_from_average': True,
                'num_threads': 4,
                'random_state': 42,
                'learning_rate': 0.01,
                'verbose': -1,
                
                # from Optuna result in Version 15
                'num_leaves': 106,
                'min_data_in_leaf': 176,
                'min_child_weight': 0.08961015929882983,
                'max_depth': 3,
                'bagging_fraction': 0.5672004837454858,
                'feature_fraction': 0.611628226420641,
                'lambda_l1': 1.293005852529098,
                'lambda_l2': 1.6012450757049599
            }
        
        else:
            opt = hyperOptimizor.hyperOpt(x_train, y_train)
            self.params_1 = self.params_2 = opt.optimize()

    def predict(self, x_train, y_train, x_test):
        n_splits = 3
        y_pred = pd.DataFrame()
        oof = pd.DataFrame()

        scores = []

        self.set_hyper_param(useDefault = True)

        for column in y_train.columns:
            print('Column:', column)
            
            if column in self.columns_scored:
                params = self.params_1
            else:
                params = self.params_2
            
            y_pred[column], oof[column], score = self.fit_predict(n_splits, params, x_train, y_train[column], x_test)
            scores.append(score)

        score = pd.DataFrame()
        score['feature'] = y_train.columns
        score['score'] = scores + [0] * (len(y_train.columns) - len(scores))

        plt.figure(figsize=(10,40))
        sns.barplot(x="score", y="feature", data=score)
        plt.show()

        submission = pd.read_csv('input/sample_submission.csv')
        columns = list(set(submission.columns) & set(y_pred.columns))
        submission[columns] = y_pred[columns]

        submission.to_csv('output/lgbm/submission.csv', index=False)

        oof.to_csv('output/lgbm/oof.csv', index=False)
        score.to_csv('output/lgbm/score.csv', index=False)
