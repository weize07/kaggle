# coding: utf-8

import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

df = pd.DataFrame.from_csv('./data/train.csv')

df3 = df
# print(len(df3))
df3 = df3.drop(df3[(df3['GrLivArea']>4000)].index)
# print(len(df3))

x_df = df3.drop('SalePrice', 1)

for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MSSubClass", "MasVnrType"):
    x_df[col] = x_df[col].fillna("None")

x_df["LotFrontage"] = x_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    x_df[col] = x_df[col].fillna(0)

x_df['MSZoning'] = x_df['MSZoning'].fillna(x_df['MSZoning'].mode()[0])
x_df['Electrical'] = x_df['Electrical'].fillna(x_df['Electrical'].mode()[0])
x_df['KitchenQual'] = x_df['KitchenQual'].fillna(x_df['KitchenQual'].mode()[0])
x_df['Exterior1st'] = x_df['Exterior1st'].fillna(x_df['Exterior1st'].mode()[0])
x_df['Exterior2nd'] = x_df['Exterior2nd'].fillna(x_df['Exterior2nd'].mode()[0])
x_df['SaleType'] = x_df['SaleType'].fillna(x_df['SaleType'].mode()[0])
x_df["Functional"] = x_df["Functional"].fillna(x_df['Functional'].mode()[0])

x_df = x_df.drop(['Utilities'], axis=1)

for col in x_df:
    if (x_df[col].dtype == np.object):
        x_df[col] = pd.factorize(x_df[col])[0]
    x_df[col].replace({-1: x_df[x_df[col] != -1][col].mean()}, inplace=True)

# x_df = x_df.fillna(x_df.mean())
y_df = df3['SalePrice']

def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

selector = SelectKBest(score_func=multivariate_pearsonr, k=10)
X = selector.fit_transform(x_df, y_df)

y = y_df.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# reg = DecisionTreeRegressor(max_depth=5)
reg = SVR(kernel='rbf', C=700000, gamma=0.0000005) # C:松弛损失系数，gamma：高斯核精度
reg.fit(X_train, y_train) #训练模型
score = reg.score(X_test, y_test)
print(score)
selector.get_support(True)
mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, df3.columns):
    if bool:
        new_features.append(feature)
print(new_features)

reg.fit(X, y)
dft = pd.DataFrame.from_csv('./data/test.csv')
ids = dft.index.values

for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MSSubClass", "MasVnrType"):
    dft[col] = dft[col].fillna("None")

dft["LotFrontage"] = dft.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    dft[col] = dft[col].fillna(0)

dft['MSZoning'] = dft['MSZoning'].fillna(dft['MSZoning'].mode()[0])
dft['Electrical'] = dft['Electrical'].fillna(dft['Electrical'].mode()[0])
dft['KitchenQual'] = dft['KitchenQual'].fillna(dft['KitchenQual'].mode()[0])
dft['Exterior1st'] = dft['Exterior1st'].fillna(dft['Exterior1st'].mode()[0])
dft['Exterior2nd'] = dft['Exterior2nd'].fillna(dft['Exterior2nd'].mode()[0])
dft['SaleType'] = dft['SaleType'].fillna(dft['SaleType'].mode()[0])
dft["Functional"] = dft["Functional"].fillna(dft['Functional'].mode()[0])

dft = dft.drop(['Utilities'], axis=1)

for col in dft:
    if (dft[col].dtype == np.object):
        dft[col] = pd.factorize(dft[col])[0]
    dft[col].replace({-1: dft[dft[col] != -1][col].mean()}, inplace=True)

# dft = dft.fillna(dft.mean())
list = selector.get_support(True)
dft = dft.iloc[:, list]

result = reg.predict(dft)

with open('sub_svr.csv', 'w') as output:
    output.write('Id,SalePrice\n')
    for i in range(len(ids)):
        output.write(str(ids[i]) + ',' + str(result[i]) + '\n')

