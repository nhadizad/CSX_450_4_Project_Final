import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline


housing_df = pd.read_csv('data/AmesHousing.tsv',sep='\t')
housing_df.rename(columns={'Order': 'Id'}, inplace=True)
housing_df.drop('PID', axis=1, inplace=True)

housing_df.set_index("Id", inplace=True)

for column in housing_df.select_dtypes(['object']).columns:
    housing_df[column] = housing_df[column].astype('category')

housing_df.MSSubClass = housing_df.MSSubClass.astype('category')
housing_df.OverallQual = housing_df.OverallQual.astype('category')
housing_df.OverallCond = housing_df.OverallCond.astype('category')
housing_df.BsmtFullBath = housing_df.BsmtFullBath.astype('category')
housing_df.BsmtHalfBath = housing_df.BsmtHalfBath.astype('category')
housing_df.FullBath = housing_df.FullBath.astype('category')
housing_df.HalfBath = housing_df.HalfBath.astype('category')
housing_df.BedroomAbvGr = housing_df.BedroomAbvGr.astype('category')
housing_df.KitchenAbvGr = housing_df.KitchenAbvGr.astype('category')
housing_df.TotRmsAbvGrd = housing_df.TotRmsAbvGrd.astype('category')
housing_df.Fireplaces = housing_df.Fireplaces.astype('category')
housing_df.GarageCars = housing_df.GarageCars.astype('category')
housing_df.MoSold = housing_df.MoSold.astype('category')


empty_means_without = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2", "FireplaceQu","GarageType","GarageFinish",
                        "GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

def replace_empty(feature, value):
    housing_df[feature].cat.add_categories([value], inplace=True)
    housing_df[feature].fillna(value, inplace=True)


for feature in empty_means_without:
    replace_empty(feature, "None")

housing_df.dropna(inplace=True)

#housing_df = pd.get_dummies(housing_df)


ames_data_df = housing_df.drop('SalePrice', axis=1)
ames_labels_srs = housing_df['SalePrice']


ames_train_data, \
    ames_test_data, \
    ames_train_labels, \
    ames_test_labels = train_test_split(ames_data_df, ames_labels_srs, test_size=0.20, random_state=42)
ames_train_data = ames_train_data.copy()
ames_test_data = ames_test_data.copy()
ames_train_labels = ames_train_labels.copy()
ames_test_labels = ames_test_labels.copy()


ames_train_data.LotFrontage.fillna(ames_train_data.LotFrontage.mean(), inplace=True)
ames_train_data.MasVnrArea.fillna(ames_train_data.MasVnrArea.mean(), inplace=True)
ames_train_data.GarageYrBlt.fillna(ames_train_data.GarageYrBlt.mean(), inplace=True)

ames_test_data.LotFrontage.fillna(ames_test_data.LotFrontage.mean(), inplace=True)
ames_test_data.MasVnrArea.fillna(ames_test_data.MasVnrArea.mean(), inplace=True)
ames_test_data.GarageYrBlt.fillna(ames_test_data.GarageYrBlt.mean(), inplace=True)


# STORE
ames_train_data.to_pickle('data/ames_train_data.p')
ames_test_data.to_pickle('data/ames_test_data.p')
ames_train_labels.to_pickle('data/ames_train_labels.p')
ames_test_labels.to_pickle('data/ames_test_labels.p')

# RETRIEVE
ames_train_data = pd.read_pickle('data/ames_train_data.p')
ames_train_labels = pd.read_pickle('data/ames_train_labels.p')
ames_test_data = pd.read_pickle('data/ames_test_data.p')
ames_test_labels = pd.read_pickle('data/ames_test_labels.p')

data = {
    'ames' : {
        'train' : {
            'raw_data' : ames_train_data,
            'labels' : ames_train_labels
        },
        'test' : {
            'raw_data' : ames_test_data,
            'labels' : ames_test_labels
        }
    }
}

numeric_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']


def ames_feature_engineering(train_data, test_data):


    # ensure column order of test is exactly the same as train
    test_data = test_data.reindex(columns = train_data.columns)
    
    train_numeric_data = train_data[numeric_features].copy()
    train_numeric_data_scaled = (train_numeric_data - train_numeric_data.mean())/(2*train_numeric_data.std())
    train_data[numeric_features] = train_numeric_data_scaled
    
    
    test_numeric_data = test_data[numeric_features].copy()
    test_numeric_data_scaled = (test_numeric_data - test_numeric_data.mean())/(2*test_numeric_data.std())
    test_data[numeric_features] = test_numeric_data_scaled
    

    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)

    
    return train_data, test_data


data['ames']['train']['engineered'], \
    data['ames']['test']['engineered'] = \
        ames_feature_engineering(data['ames']['train']['raw_data'].copy(),
                                  data['ames']['test']['raw_data'].copy())




