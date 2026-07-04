# data_loader.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats

def load_and_preprocess_data():
    """
    Loads sales, stores, and features data, handles merges, 
    imputes missing values, and runs feature engineering.
    """
    # 1. Load Data
    sales = pd.read_csv("../data/sales.csv")
    stores = pd.read_csv("../data/stores.csv")
    features = pd.read_csv("../data/Features.csv")

    # 2. Convert Dates
    sales["Date"] = pd.to_datetime(sales["Date"], format="%d/%m/%Y")
    features["Date"] = pd.to_datetime(features["Date"], format="%d/%m/%Y")

    # 3. Merge
    df1 = pd.merge(sales, features, how='left', on=['Store','Date','IsHoliday'])
    df = pd.merge(df1, stores, how='left', on='Store')

    # 4. Handle Types & Missing Values
    columns_to_object = ['Store', 'Dept', 'IsHoliday', 'Type']
    df[columns_to_object] = df[columns_to_object].astype('object')
    
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    # 5. Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['LogSales'] = np.log1p(df['Weekly_Sales'])
    df['TypeEnc'] = LabelEncoder().fit_transform(df['Type'])

    # Set visualization global parameters
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (14, 6)

    print("Data successfully loaded and preprocessed! Shape:", df.shape)
    return df