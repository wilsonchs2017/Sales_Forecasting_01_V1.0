#Using ML Algorithms
####################
#Import Library
import pandas as pd #data processing, csv file
import numpy as np #linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

#csv folder link https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/code?select=features.csv
#df_test = pd.read_csv("E:\ML Projects\Sales Forecasting\Walmart Sales Forecast Dataset\Test.csv")

#ETL Phase
#---------
#df stores check
df_stores = pd.read_csv("E:\ML Projects\Sales Forecasting\Walmart Sales Forecast Dataset\Stores.csv")
df_stores.shape

df_stores.head()

df_stores.info()

#df features check
df_features = pd.read_csv("E:\ML Projects\Sales Forecasting\Walmart Sales Forecast Dataset\Features.csv")
df_features.shape

df_features.head()

df_features.info()

df_features.isna().sum()

#filling missing values
df_features['CPI'].fillna(df_features['CPI'].median(), inplace=True)
df_features['Unemployment'].fillna(df_features['Unemployment'].median(),inplace=True)

df_features['MarkDown1'].value_counts().unique()

for i in range(1,6):
    df_features["MarkDown"+str(i)]=df_features["MarkDown"+str(i)].apply(lambda x:0 if x<0 else x)
    df_features["MarkDown"+str(i)].fillna(value=0,inplace=True)

df_features.head()

df_features.info()

#df train check
df_train = pd.read_csv("E:\ML Projects\Sales Forecasting\Walmart Sales Forecast Dataset\Train.csv")
df_train.shape

df_train.head()

df_train.info()

df_train.isna().sum()

#train data merge with stores data
merged_data = pd.merge(df_train,df_stores, on = 'Store', how='left')

#merged_data merges with features data, expanding merged data, this will be main data, nd data checkpoint
main_data = pd.merge(merged_data,df_features, on=['Store','Date'], how='left')
main_data.head()

#check for nulls
main_data.info()

main_data['Date'] = pd.to_datetime(main_data['Date'], errors = 'coerce')
main_data.sort_values(by=['Date'], inplace=True)
main_data.set_index(main_data.Date, inplace=True)
main_data.head()

#checkpoint, whether the column for IsHoliday_x and IsHoliday_y are same or not
main_data['IsHoliday_x'].isin(main_data['IsHoliday_y']).all()

#removing duplicated column and renaming the column
main_data.drop(columns='IsHoliday_x', inplace=True)
main_data.rename(columns={'IsHoliday_y':'IsHoliday'},inplace=True)
main_data.info()

#data checkpoint
main_data.head()

main_data['Year'] = main_data['Date'].dt.year
main_data['Month'] = main_data['Date'].dt.month
main_data['Week'] = main_data['Date'].dt.week

#data checkpoint
main_data.head()

#outlier detection and abnormalities in data
agg_data = main_data.groupby(['Store','Dept']).Weekly_Sales.agg(['max','min','mean','median','std']).reset_index()
agg_data.head()

#null checkpoint
agg_data.isnull().sum()

#merging main data and aggregated data into store data
store_data = pd.merge(left=main_data, right=agg_data, on = ['Store','Dept'], how = 'left')
store_data.head(2)

#removing all rows with nulls
store_data.dropna(inplace=True)

#set main data variable as store data copy
main_data=store_data.copy()

#convert argument to datetime, sort the values, data checkpoint for any adjustments
main_data['Date'] = pd.to_datetime(main_data['Date'],errors='coerce')
main_data.sort_values(by=['Date'],inplace=True)
main_data.set_index(main_data.Date, inplace=True)
main_data.head()

#create Total MarkDown, remove all MarkDown# variables, data checkpoint for any adjustments
main_data['Total_MarkDown'] = main_data['MarkDown1']+main_data['MarkDown2']+main_data['MarkDown3']\
                              +main_data['MarkDown4']+main_data['MarkDown5']
main_data.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis=1,inplace=True)
main_data.head()

#check the main data shape
main_data.shape

#extract Weekly Sales, Size, Temperature, Fuel Price, CPI, Unemployment, Total MarkDown
numeric_col=['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment','Total_MarkDown']
data_numeric = main_data[numeric_col].copy()

#check data numeric
data_numeric.head()

#show only rows with z score less than 2.5, check for the data shape for changes
main_data = main_data[(np.abs(stats.zscore(data_numeric)) < 2.5).all(axis=1)]
main_data.shape

#show only rows with Weekly Sales greater than or equals to 0, check for the data shape for changes
main_data = main_data[main_data['Weekly_Sales'] >=0]
main_data.shape

#convert IsHoliday to Integer, checkpoint for main data for any adjustments
main_data['IsHoliday'] = main_data['IsHoliday'].astype('int')
main_data.head()

#Average Monthly Sales
plt.figure(figsize=(14,8))
sns.barplot(x='Month', y='Weekly_Sales', data = main_data)
plt.title('Average Monthly Sales', fontsize=16)
plt.xlabel('Months',fontsize=14)
plt.ylabel('Sales',fontsize=14)
plt.savefig('avg_monthly_sales.png')
plt.grid()

#Monthly Sales for Each Year
monthly_data = pd.crosstab(main_data['Year'], main_data['Month'], values=main_data['Weekly_Sales'], aggfunc='sum')
monthly_data

#create 12 different lineplots for each monthly sales per year, assorted each month per plot, lineplot
fig,axes = plt.subplots(3,4,figsize=(16,8))
plt.suptitle('Monthly Sales For Each Year', fontsize=18)
k = 1
for i in range(3):
    for j in range(4):
        sns.lineplot(data = monthly_data[k], ax=axes[i,j])
        plt.subplots_adjust(wspace=0.8,hspace=0.6)
        plt.xlabel('Years', fontsize=12)
        plt.ylabel(k,fontsize=12)
        k+=1
plt.savefig('monthly_sales_every_year.png')
plt.show()

#Average Weekly Sales per Store, barplot
plt.figure(figsize = (20,10))
sns.barplot(x='Store', y='Weekly_Sales', data=main_data)
plt.grid()
plt.title('Average Sales per Store', fontsize=18)
plt.xlabel('Store', fontsize=16)
plt.ylabel('Sales',fontsize=16)
plt.savefig('stores_avg_sales.png')
plt.show()

#Average Sales per Department, barplot
plt.figure(figsize = (20,10))
sns.barplot(x='Dept',y='Weekly_Sales', data = main_data)
plt.grid()
plt.title('Average Sales per Department', fontsize=18)
plt.xlabel('Department',fontsize=16)
plt.ylabel('Sales',fontsize=16)
plt.savefig('depts_avg_sales.png')
plt.show()

#Temperature's Effects on Sales, distribution plot
plt.figure(figsize=(10,10))
sns.distplot(main_data['Temperature'])
plt.title('Effect of Temperature on Sale',fontsize =16)
plt.xlabel('Density', fontsize = 16)
plt.ylabel('Temperature', fontsize = 16)
plt.savefig('temps_effects_sales.png')
plt.show()

#Holiday Distribution, piechart
plt.figure(figsize=(10,10))
plt.pie(main_data['IsHoliday'].value_counts(), labels=['No Holiday', 'Holiday'],autopct = '%0.2f%%')
plt.title("Pie Chart Distrubtion", fontsize=16)
plt.legend()
plt.savefig('holiday_dist_pie.png')
plt.show()

#Time Series Decompose
sm.tsa.seasonal_decompose(main_data['Weekly_Sales'].resample('MS').mean(),model='addictive').plot()
plt.savefig('seasonal_decomp.png')
plt.show()

#Encoding List
main_data.dtypes

#create new variables cat_col and data_cat
cat_col = ['Store','Dept','Type']
data_cat = main_data[cat_col].copy()
data_cat.head()

#train with dummies, check the new data table
data_cat = pd.get_dummies(data_cat,columns=cat_col)
data_cat.head()

#check main_data shape
main_data.shape

#Concat main_data and data_cat
main_data = pd.concat([main_data,data_cat], axis = 1)

#check the newly updated main_data shape
main_data.shape

#drop columns cat_col
main_data.drop(columns = cat_col, inplace=True)

#drop columns Date
main_data.drop(columns = ['Date'], inplace=True)

#check the updated main_data shape
main_data.shape

#Data Normalization
num_col = ['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment',
           'Total_MarkDown','max','min','mean','median','std']

minmax_scale= MinMaxScaler(feature_range=(0,1))
def normalization(df,col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
    return df

#Load main data head
main_data.head()

#update main data with normalization and check the headers
main_data = normalization(main_data.copy(), num_col)
main_data.head()

#Finding correlation between features
plt.figure(figsize=(15,10))
corr = main_data[num_col].corr()
sns.heatmap(corr,vmax=1.0, annot=True)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('correlation_matrix_map.png')
plt.show()

#Feature Elimination
feature_col = main_data.columns.difference(['Weekly_Sales'])
feature_col

#set radm_clf for RandomForestRegressor in rank table use
radm_clf = RandomForestRegressor (oob_score=True, n_estimators=23)
radm_clf.fit(main_data[feature_col],main_data['Weekly_Sales'])

#creating feature rank table
indices = np.argsort(radm_clf.feature_importances_)[::-1]
feature_rank = pd.DataFrame(columns=['rank','feature','importance'])
for f in range(main_data[feature_col].shape[1]):
    feature_rank.loc[f] = [f+1, main_data[feature_col].columns[indices[f]], radm_clf.feature_importances_[indices[f]]]
feature_rank

#print feature rank
x=feature_rank.loc[0:25,['feature']]
x=x['feature'].tolist()
print(x)

#create and set variables for X and Y
X = main_data[x]
Y = main_data['Weekly_Sales']

#checkpoint for the updated main data heads
main_data = pd.concat([X,Y], axis = 1)
main_data.head()

#Building the models
#-------------------
#Splitting data into train data and test data
X = main_data.drop(['Weekly_Sales'],axis=1)
Y = main_data.Weekly_Sales

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=50)

#Linear Regression Model
lr = LinearRegression()
lr.fit(X_train,y_train)

linear_regression_accuracy = lr.score(X_test,y_test)*100
print("Linear Regressor Accuracy - ", linear_regression_accuracy)

y_pred = lr.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))

lr_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
lr_df.head()

plt.figure(figsize=(15,10))
plt.title('Actual Values and Predicted Values Comparison', fontsize = 16)
plt.plot(lr.predict(X_test[:100]), label = 'prediction', linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label = "real_values", linewidth=3.0, color='red')
plt.legend(loc='best')
plt.savefig('lr_real_pred_model.png')
plt.show()

#Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

random_forest_accuracy = rf.score(X_test,y_test)*100
print("Random Forest Regressor Accuracy - ", random_forest_accuracy)

y_pred = rf.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))

rf_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
rf_df.head()

plt.figure(figsize=(15,10))
plt.title('Actual Values and Predicted Values Comparison', fontsize = 16)
plt.plot(rf.predict(X_test[:100]), label = 'prediction', linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label = "real_values", linewidth=3.0, color='red')
plt.legend(loc='best')
plt.savefig('rf_real_pred_model.png')
plt.show()

#KNN
knn = KNeighborsRegressor(n_neighbors = 1,weights = 'uniform')
knn.fit(X_train,y_train)

knn_accuracy = knn.score(X_test,y_test)*100
print("KNeighbors Regressor Accuracy - ", knn_accuracy)

y_pred = knn.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))

knn_df = pd.DataFrame ({'Actual':y_test,'Predicted':y_pred})
knn_df.head()

plt.figure(figsize=(15,10))
plt.title('Actual Values and Predicted Values Comparison', fontsize = 16)
plt.plot(knn.predict(X_test[:100]), label = 'prediction', linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label = "real_values", linewidth=3.0, color='red')
plt.legend(loc='best')
plt.savefig('knn_real_pred_model.png')
plt.show()

#XGBoost Regressor
xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)

xgboost_accuracy = xgbr.score(X_test,y_test)*100
print("XGBoost Regressor Accuracy - ", xgboost_accuracy)

y_pred = xgbr.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))

xgb_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
xgb_df.head()

plt.figure(figsize=(15,10))
plt.title('Actual Values and Predicted Values Comparison', fontsize = 16)
plt.plot(xgbr.predict(X_test[:100]), label = 'prediction', linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label = "real_values", linewidth=3.0, color='red')
plt.legend(loc='best')
plt.savefig('xgb_real_pred_model.png')
plt.show()

#Model's Accuracy Comparison
acc = {'model':['linear_regression_accuracy','random_forest_accuracy','knn_accuracy',
                'xgboost_accuracy'],'accuracy':[linear_regression_accuracy,
                random_forest_accuracy,knn_accuracy,xgboost_accuracy]}

acc_df = pd.DataFrame(acc)
acc_df

plt.figure(figsize = (10,10))
plt.title("Comparing accuracy of models", fontsize=16)
sns.barplot(x='model',y='accuracy',data=acc_df)
plt.savefig('models_comparison.png')
plt.show

#Using DL Algorithms
#####################
#Import Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#One Run ---------
input_shape = [X_train.shape[1]]
model = keras.Sequential([layers.BatchNormalization(input_shape = input_shape),
                          layers.Dense(8, activation = 'relu'),
                          layers.Dropout(0,3),
                          layers.BatchNormalization(),
                          layers.Dense(24, activation = 'relu'),
                          layers.Dense(10)
                          ])

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mean_squared_error'])

early_stopping = keras.callbacks.EarlyStopping(patience = 5, min_delta = 0.01, restore_best_weights = True,)

history = model.fit(X_train, y_train, validation_split = 0.2, batch_size = 25,
                    epochs =1000, callbacks = [early_stopping])
#One Run ----------

mse, mae = model.evaluate(X_test, y_test)
print("mean square error is :", mse)
print("mean absolute error is :", mae)