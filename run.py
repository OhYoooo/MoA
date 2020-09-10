import numpy as np
import pandas as pd
import os
import lgbm

for dir_name, _, file_names in os.walk('input'):
    for file_name in file_names:
        print(os.path.join(dir_name, file_name))


train = pd.read_csv('input/train_features.csv')
train_target = pd.read_csv('input/train_targets_scored.csv')
test = pd.read_csv('input/test_features.csv')

"""
train.shape: (23814, 876)
train_target.shape: (23814, 207)
test.shape: (3982, 876)
"""

# List (agonist, antagonist) pair
antagonists = []

for column in train_target.columns:
    if column.endswith('_agonist'):
        antagonist = column.replace('_agonist', '_antagonist')
        if antagonist in train_target.columns:
            antagonists.append((column, antagonist))

"""
[
 ('acetylcholine_receptor_agonist', 'acetylcholine_receptor_antagonist'),
 ('adenosine_receptor_agonist', 'adenosine_receptor_antagonist'),
 ('adrenergic_receptor_agonist', 'adrenergic_receptor_antagonist'),
 ('androgen_receptor_agonist', 'androgen_receptor_antagonist'),
 ('cannabinoid_receptor_agonist', 'cannabinoid_receptor_antagonist'),
 ('dopamine_receptor_agonist', 'dopamine_receptor_antagonist'),
 ('estrogen_receptor_agonist', 'estrogen_receptor_antagonist'),
 ('gaba_receptor_agonist', 'gaba_receptor_antagonist'),
 ('glutamate_receptor_agonist', 'glutamate_receptor_antagonist'),
 ('histamine_receptor_agonist', 'histamine_receptor_antagonist'),
 ('opioid_receptor_agonist', 'opioid_receptor_antagonist'),
 ('ppar_receptor_agonist', 'ppar_receptor_antagonist'),
 ('progesterone_receptor_agonist', 'progesterone_receptor_antagonist'),
 ('retinoid_receptor_agonist', 'retinoid_receptor_antagonist'),
 ('serotonin_receptor_agonist', 'serotonin_receptor_antagonist'),
 ('sigma_receptor_agonist', 'sigma_receptor_antagonist'),
 ('tlr_agonist', 'tlr_antagonist'),
 ('trpv_agonist', 'trpv_antagonist')
]
"""

for pair in antagonists:
    n = train_target[(train_target[pair[0]] == 1) & (train_target[pair[1]] == 1)].shape[0]
    # if n > 0:
        # print(pair[0], '-', pair[1])
        # print('Number of cases:', n)

"""
gaba_receptor_agonist - gaba_receptor_antagonist
Number of cases: 6
progesterone_receptor_agonist - progesterone_receptor_antagonist
Number of cases: 6
serotonin_receptor_agonist - serotonin_receptor_antagonist
Number of cases: 7
sigma_receptor_agonist - sigma_receptor_antagonist
Number of cases: 6
"""

# by https://www.kaggle.com/carlmcbrideellis/moa-setting-ctl-vehicle-0-improves-score
# set ctl-vehicle = 0
train.at[train['cp_type'].str.contains('ctl_vehicle'),train.filter(regex='-.*').columns] = 0.0
test.at[test['cp_type'].str.contains('ctl_vehicle'),test.filter(regex='-.*').columns] = 0.0

# one-hot-encoding
train_size = train.shape[0]
traintest = pd.concat([train, test])

traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_type'], prefix='cp_type')], axis=1)
traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_time'], prefix='cp_time')], axis=1)
traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_dose'], prefix='cp_dose')], axis=1)

traintest = traintest.drop(['cp_type', 'cp_time', 'cp_dose'], axis = 1)

train = traintest[:train_size]
test  = traintest[train_size:]

del traintest

# drop sig_id
x_train = train.drop('sig_id', axis = 1)
y_train = train_target.drop('sig_id', axis = 1)
x_test = test.drop('sig_id', axis = 1)


# LightGBM model
lightGBM = lgbm.lightGBM()
lightGBM.predict(x_train, y_train, x_test)