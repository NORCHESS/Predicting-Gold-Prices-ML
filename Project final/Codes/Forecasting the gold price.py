#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Authors: Hayk Mkrtchyan, Norayr Hayruni
# Class:   Financial Instruments: Stocks, Bonds, Derivatives and Hedge Funds
# Date:    14.06.2021
# Topic:   'Forecasting Gold Prices: Evidence from Support Vector Machines and Neural Networks' 


# In[2]:


# Make sure to have Pycaret installed on your machine. The following code may do it for you.     
# pip install pycaret                   


# In[79]:


# Importing libraries  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from datetime import datetime
from datetime import datetime, timedelta
import seaborn as sns
from pycaret.regression import *
from pycaret.classification import *
import tensorflow as tf 
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# In[84]:


# Importing dataset, filling in the missing values and Illustrating it
data = pd.read_excel('data.xlsx', sheet_name='All Variables')
data = data.fillna(method="ffill",axis=0)
data = data.fillna(method="bfill",axis=0)
data.head().round(decimals=2)


# In[5]:


# Checking for missing values and confirming that there is none 
print(data.isna().sum())


# In[6]:


# Now let's import the ticker names from another excel sheet and 
# store the tickers and their respective names in separate variables for further use
ticker_details  = pd.read_excel('data.xlsx', sheet_name='Ticker Names')           # reading the excel file
ticker = ticker_details.loc[:, 'Ticker'].to_list()                                # extracting the tickers
names = ticker_details.loc[:, 'Description'].to_list()                            # extracting the names 
names.insert(0,'Date')                                                     
#data.columns = names                                   
cols=data.columns.drop('Date')                                                    
data[cols] = data[cols].apply(pd.to_numeric,errors='coerce').round(decimals=1)    # Coercing all data values to numeric values


# In[7]:


# Some descriptive statistics on our variables
data.describe().round(decimals=2)


# In[8]:


# Correlation Matrix
corr = data.corr(method = 'pearson')   
plt.figure(figsize= (8, 8))
sns.heatmap(corr, cbar=True, square=True, fmt = '.2f', annot=True, annot_kws={'size': 7.8}, cmap = 'Blues')
plt.savefig('correlation_matrix.png',  transparent=True, dpi=200)
# From the below corrlation matrix, one can observe that gold is highly positively correlated with silver, platinum 
# and other metals. On the other hand, gold is negatively correlated with equities.   


# In[9]:


# The Historical development of the gold price (2010-2021)
data_illustration = pd.read_excel('data.xlsx', sheet_name='Gold Price')
plt.figure(figsize=(12,7))
plt.plot('Date', 'Gold', data=data_illustration)
plt.grid(axis='y')
plt.ylabel('Price (USD)\n', size=14)
plt.title('Gold Price', size = 17)
plt.savefig('Gold Price.png', dpi=200, transparent=True)


# In[10]:


# calculating short-term returns of all explanatory variables (we have tried various short-term returns of variables
# however only the 21 days return are worth to include in our regression analysis, 
# since the other returns turned out to be highly insignificant predictors.
rets = pd.DataFrame(data=data['Date'])
x = data[cols].pct_change(periods=21).add_suffix('-t-21')
rets = pd.concat(objs=(rets, x), axis = 1)


# In[11]:


#calculating long-term historical returns of selected instruments (we have tried to see whether the long-term returns of 
#these variables are able to explain the price of the Gold)
period = [60,90,180,250]
sel_var = ['Gold','Silver', 'Crude Oil', 'S&P500','MSCI EM ETF', 'Soybean Futures', 'NYSE']
for i in period:
    x = data[sel_var].pct_change(periods=i).add_suffix('-t-'+str(i))
    rets=pd.concat(objs=(rets, x), axis=1)


# In[12]:


rets.tail()  # Having a look at the explainatory variables we have created so far


# In[13]:


# Creatig a new dataframe to compute the long-term moving averages of the gold price. We believe that the moving averages 
# of gold for 90 and 180 days are able to predict the future gold price. Again, we tried many combinations (other time horizons) 
# until we decided to include only these two.
mov_average = pd.DataFrame(data['Date'], columns=['Date'])
mov_average['Date'] = pd.to_datetime(mov_average['Date'], format='%Y-%b-%d')
mov_average['Gold/90SMA'] = (data['Gold']/(data['Gold'].rolling(window=90).mean()))-1
mov_average['Gold/180SMA'] = (data['Gold']/(data['Gold'].rolling(window=180).mean()))-1
rets['Date'] = pd.to_datetime(rets['Date'], format='%Y-%b-%d')
rets = pd.merge(left=rets, right=mov_average, how='left', on='Date')


# In[14]:


# Now, let's create our variable of interest for prediction. In our regression analysis, we decided to predict the 30 day
# return of the gold, and hence its price. 
y_var = pd.DataFrame(data=data['Date'])
y_var['Gold-t+30'] = data['Gold'].pct_change(periods = -30)


# In[15]:


# Dropping NA values  
rets = rets[rets['Gold-t-250'].notna()]
y_var = y_var[y_var['Gold-t+30'].notna()]


# In[16]:


# Merging the dependent variable with all our independent variables in one dataframe
rets = pd.merge(left=rets, right=y_var, how='inner', on='Date', suffixes=(False,False))


# In[17]:


rets.isna().sum()    # Making sure that there are not missing values in the final dataset


# In[18]:


# After finalizing the dataset, it is time to do a regression analysis and prediction. For Regression analysis we will stick 
# with a pycaret library. It is worth mentioning that pycaret is a powerful library, which allows to run numerious ML algorithms 
# in a coherent and practical manner. In addition, feature scaling of the variables are done on the background of Pycaret, 
# therefore there is no need to do it separately. 


# In[19]:


# Pycaret: The first step in pycaret is to initialize the regression function. 
# About the parameters of initialize function - note that pycaret will do 10 cross validation and produce the mean results. 
# Another key parameter here to notice is the fold strategy, which should be set to timeseries, since we are dealing with timeseries data.
# In addition, in case the profile is set to True, data profile for Explanetory Data Analysis will be 
# displayed in an interactive HTML report. At this point, we don't need it.
from pycaret.regression import *
initialize = setup(rets, target='Gold-t+30',
        ignore_features=['Date'], session_id=11,          
        silent=True, profile=False, train_size = 0.75,
        remove_outliers=True, outliers_threshold = 0.05, fold_strategy='timeseries', fold=10, ignore_low_variance = False, 
        remove_perfect_collinearity = True, multicollinearity_threshold = 0.85)

# In the comparision, one can specify other ML algorithms as well. I have mentioned the ones which after several trials turn out 
# to perform good at predicting the future gold price. However, one can experiment with others as well. 
comparison = compare_models(turbo=True, include = ['svm', 'catboost', 'lightgbm'])


# In[20]:


# Since our project requires from us to experiment the SVM, which we do later on in classification part, we decided to observe 
# its predicting power in forecasting the future gold price. Therefore, we create SVM model here. 
# Below you can find the results of our different regression models after 10 cross validations
svm = create_model('svm')


# In[21]:


catboost = create_model('catboost')


# In[22]:


lightgbm = create_model('lightgbm')


# In[23]:


# Now, let's do something interactive. We will use the above created model to predict the future gold price on a new dataset 
# which was neither a part of our training nor testing data
# It is worth to mention that for a regression we decided to predict the return of a gold 30 days ahead,
# hence its price. So let's save the created models and implement our predictions on new data w
reg2_30_days = save_model(svm, model_name='Pred_30_days_svm')
reg3_30_days = save_model(lightgbm, model_name='Pred_30_days_lightgbm')
reg4_30_days = save_model(catboost, model_name = 'Pred_30_days_catboost')


# In[24]:


# We will import the new data from another sheet of the excel file, and the data spans from 2020-01-02 to 2021-06-04, however some observations
# will be dropped out later, since when computing the long-run returns of our explanatory variables, various NA values come out as before.
new_data = pd.read_excel('data.xlsx', sheet_name='Validation')


# In[25]:


new_data.head().round(decimals=2)


# In[26]:


new_data.columns = names  # Renaming the column names for the ease of understanding the explanatory variables 


# In[27]:


# Filling in the missing values
new_data = new_data.fillna(method="ffill", axis=0)
new_data = new_data.fillna(method="bfill", axis=0)
cols = new_data.columns.drop('Date')
new_data[cols] = new_data[cols].apply(pd.to_numeric, errors='coerce').round(decimals=1)
new_data.head()


# In[28]:


# calculating short-term returns of our explanatory variables
period = [21]
rets_new = pd.DataFrame(data=new_data['Date'])
x = new_data[cols].pct_change(periods=21).add_suffix('-t-21')
rets_new = pd.concat(objs=(rets_new, x), axis=1)


# In[29]:


#calculating long term historical returns of selected variables
period = [60, 90, 180, 250]
sel_var = ['Gold','Silver', 'Crude Oil', 'S&P500','MSCI EM ETF', 'Soybean Futures', 'NYSE']
for i in period:
    x = new_data[sel_var].pct_change(periods=i).add_suffix('-t-'+str(i))
    rets_new = pd.concat(objs=(rets_new, x), axis=1)


# In[30]:


#Calculating Moving averages for 90 and 180 days horizon and merging with rets_new variable. 
mov_average = pd.DataFrame(new_data['Date'], columns=['Date'])
mov_average['Date'] = pd.to_datetime(mov_average['Date'], format='%Y-%b-%d')
mov_average['Gold/90SMA'] = (new_data['Gold']/(new_data['Gold'].rolling(window=90).mean()))-1
mov_average['Gold/180SMA'] = (new_data['Gold']/(new_data['Gold'].rolling(window=180).mean()))-1 
rets_new['Date'] = pd.to_datetime(rets_new['Date'], format='%Y-%b-%d')
rets_new = pd.merge(left=rets_new, right=mov_average, how='left', on='Date')
rets_new = rets_new[rets_new['Gold-t-250'].notna()]             
pred_data = rets_new.copy()   # making a copy of our dataframes before proceeding to the forecasting


# In[31]:


# Kindly note that all the data we manipulated now have been aggregated in rets_new variable 
pred_data.head()


# In[32]:


# After preparing the data, let's load the models we created above and predict the gold price for 30 days horizon.


# In[33]:


reg_30_svm = load_model('pred_30_days_svm');            # Support Vector Regressor 
reg_30_catboost = load_model('pred_30_days_catboost');  # Catboost Regressor
reg_30_lightgbm = load_model('pred_30_days_lightgbm');  # Light gradient boosting machine


# In[34]:


# One interpretation for the first observation - On 2020-12-17 the gold price was 1887.2 and according to one of our created models,
# in this case SVR, its predicted return over the next 30 days is expected to be -0.2302%, hence its price 
# is expected to be 1882.9 on 2021-01-16. Feel free to change the model in the above to experiment with other algorithms.


# In[35]:


# Now, let's predict the gold price over 30 day's horizon using our models
pred_ret_30_svm = predict_model(reg_30_svm, data=pred_data)  
pred_ret_30_catboost = predict_model(reg_30_catboost, data=pred_data)
pred_ret_30_lightgbm = predict_model(reg_30_lightgbm, data=pred_data)
pred_ret_30_svm = pred_ret_30_svm[['Date','Label']]
pred_ret_30_svm.columns = ['Date','Return_30_svm']               
pred_ret_30_catboost = pred_ret_30_catboost[['Date','Label']]
pred_ret_30_catboost.columns = ['Date','Return_30_catboost']  
pred_ret_30_lightgbm = pred_ret_30_lightgbm[['Date','Label']]
pred_ret_30_lightgbm.columns = ['Date','Return_30_lightgbm']


# In[36]:


# Here we merge all of our 3 models together to be able to compare their performance on the data 
# Let's interpret our first observation of our pred_values dataframe. On 2020-12-17 the gold price was 1887.2 and its predicted 
# return in 30 days (2021-01-16) is -0.23%, 0.3838%, -0.6867% according to SVM, Catboost and LGBM respectively. 
pred_values = new_data[['Date','Gold']]
pred_values = pred_values.tail(len(pred_ret_30_svm))                                              
pred_values = pd.merge(left=pred_values,right=pred_ret_30_svm,on=['Date'],how='inner')             
pred_values['Gold-t+30_svm'] = (pred_values['Gold']*(1+pred_values['Return_30_svm'])).round(decimals =1) 
pred_values['Date-t+30'] = pred_values['Date']+timedelta(days = 30) 
pred_values = pd.merge(left=pred_values,right=pred_ret_30_lightgbm,on=['Date'],how='inner')
pred_values = pd.merge(left=pred_values,right=pred_ret_30_catboost,on=['Date'],how='inner')
pred_values['Gold-t+30_catboost'] = (pred_values['Gold']*(1+pred_values['Return_30_catboost'])).round(decimals =1)
pred_values['Gold-t+30_lightgbm'] = (pred_values['Gold']*(1+pred_values['Return_30_lightgbm'])).round(decimals =1)
pred_values.head()


# In[37]:


# Combining the actual price of Gold and the predicted ones in one dataframe to better observe the results 
pred_values_final = pred_values.iloc[:, [0,1,3,7,8,4]]
pred_values_real = pred_values.filter(['Date','Gold']).iloc[22:, :]
pred_values_prediction = pred_values.filter(['Date-t+30', 'Gold-t+30_svm', 'Gold-t+30_catboost', 'Gold-t+30_lightgbm']).iloc[:-22, :]
combined = pd.DataFrame()
combined = pred_values_real
combined = combined.reset_index().drop('index', axis = 1)
combined['SVM'] = pred_values_prediction.loc[:, 'Gold-t+30_svm']
combined['Catboost'] = pred_values_prediction.loc[:, 'Gold-t+30_catboost']
combined['LGBM'] = pred_values_prediction.loc[:, 'Gold-t+30_lightgbm']
combined.head()


# In[38]:


#### Part 2 - Classification with SVM and Artifical Neural Network
# In this second part, we will implement classification using solely SVM with its linear and RBF kernels. 
# Each classification problem requires labeling our dependent variable. For our classification problem 
# we have chosen 1 month and 12 months horizons. So the intuition is that if the 30th day gold price is higher 
# than the current price, it will assign 1 to our dependent variable and 0 otherwise. The same logic 
# applies for our 365th day horizon. If the price in a year is greater than the price today, it will assign 1,
# and 0 otherwise. Before proceeding to running the algorithms, we have to do some data manipulation and modification 
# to prepare our data 


# In[39]:


# calculating short-term returns of our explanatory variables 
period = [7, 14, 21]
rets_cl = pd.DataFrame(data=data['Date'])
for i in period:
    x = data[cols].pct_change(periods=i).add_suffix('-t-'+str(i))
    rets_cl = pd.concat(objs=(rets_cl, x), axis=1)


# In[40]:


# calculating long term historical returns of selected variables - for our classification we have added a few more after 
# experimenting a number of times 
period = [60, 90, 180, 250]
selected_var = ['Gold','Silver', 'Crude Oil', 'S&P500','MSCI EM ETF', 'Nasdaq', 'Russel 2000', 'Euro USD', 'Platinum', 'NYSE',
      '10 T-Note', 'Copper', 'Soybean Futures']
for i in period:
    x = data[selected_var].pct_change(periods=i).add_suffix('-t-'+str(i))
    rets_cl = pd.concat(objs=(rets_cl, x), axis=1)


# In[41]:


#Calculating Moving averages
mov_average_cl = pd.DataFrame(data['Date'], columns=['Date'])
mov_average_cl['Date']=pd.to_datetime(mov_average_cl['Date'], format='%Y-%b-%d')
#Adding Simple Moving Averages
mov_average_cl['Gold-90MA'] = (data['Gold']/(data['Gold'].rolling(window=90).mean()))-1
mov_average_cl['Gold-180MA'] = (data['Gold']/(data['Gold'].rolling(window=180).mean()))-1
#Adding Exponential Moving Averages
mov_average_cl['Gold-90EMA'] = (data['Gold']/(data['Gold'].ewm(span=90,adjust=True,ignore_na=True).mean()))-1
mov_average_cl['Gold-180EMA'] = (data['Gold']/(data['Gold'].ewm(span=180,adjust=True,ignore_na=True).mean()))-1
mov_average_cl = mov_average_cl.dropna(axis=0)
# Merging  with our main dataframe
rets_cl['Date'] = pd.to_datetime(rets_cl['Date'], format='%Y-%b-%d')
rets_cl = pd.merge(left=rets_cl, right=mov_average_cl, how='left', on='Date')


# In[42]:


# At this point it is worth to make a copy of our rets_cl dataframe, for further using it when dealing with the classification 
# problem for the 365 days horizon
rets_cl_365 = rets_cl.copy()


# In[43]:


# Adding a new variable Y_30 which is the same as gold prices. We will convert then to 1 and 0 as discussed above.
gold_prices = data.loc[:, 'Gold']
data['Y_30'] = gold_prices.copy() 


# In[44]:


# Preparing the label Y for SVM algorithm 
data.loc[data['Gold'] > data['Gold'].shift(periods=-30, axis='rows'), 'Y_30'] = 1 
data.loc[data['Gold'] <= data['Gold'].shift(periods=-30, axis='rows'), 'Y_30'] = 0 
Y_30 = data.filter(['Date', 'Y_30'])
rets_cl = pd.merge(left = rets_cl, right = Y_30, how = 'inner', on = 'Date', suffixes=(False, False))
rets_cl = rets_cl[rets_cl['Gold-t-250'].notna()]
rets_cl = rets_cl.drop(np.arange(start = 2579, stop = 2609, step = 1), axis = 0) 


# In[45]:


# let's have a look at our dataframe to see if all explanatory and dependent variables seat there on.
rets_cl.head()


# In[46]:


# Pycaret - Now, let's run the SVM classification. Note that we specified in the setup stage 
# of the algorithm that we want to see only linear SVM and Radial SVM results. However, one is free to 
# experiment with the other classification algorithms. 


# In[129]:


from pycaret.classification import *
initialize = setup(rets_cl, target ='Y_30', train_size = 0.70, ignore_features=['Date'], session_id=11, silent=True, 
                   remove_outliers=True, outliers_threshold=0.02, remove_multicollinearity=False, 
                   multicollinearity_threshold=0.85, fold_strategy='timeseries', fold=10)  

compare = compare_models(turbo=True, include=['rbfsvm', 'svm'], cross_validation=True)


# In[130]:


# Now, let's create the two models of interest and tune the hyperparameters of the radial SVM.
svm_linear = create_model('svm', fold=5, cross_validation=True)


# In[131]:


rbf_svm = create_model('rbfsvm', fold = 5, cross_validation=True)


# In[132]:


rbf_svm_tuned = tune_model(rbf_svm, fold=5)


# In[51]:


# Now, let's observe the results according to the metric 'Accuracy'. One can immediately notice that after tuning the 
# hyperparameters of rbfSVM, the accuracy went from 54.76% to 77.1%, which is a significant improvement. It is worth mentioning 
# that the standard deviation also decreased from 0.0605 to 0.0391 by 0.0214. At this point, we would like to remind 
# that we are predicting the returns. In addition, the linear SVM performed good as well with an accuracy of 76%, 
# which is almost the same result as with the radial SVM. It is also important to highlight that the results are shown 
# after the 10 cross validations. Each validation result can be seen separetely above, when we created the models.


# In[52]:


# Now, let's do the same predictions, but this time for 365 day (12 month) horizon.


# In[53]:


data['Y_365'] = gold_prices.copy()


# In[54]:


# Here we just create a variable Y_365 in which 0 and 1 values are being stored according to the logic explanied above
data.loc[data['Gold'] > data['Gold'].shift(periods=-365, axis='rows'), 'Y_365'] = 1 
data.loc[data['Gold'] <= data['Gold'].shift(periods=-365, axis='rows'), 'Y_365'] = 0 
Y_365 = data.filter(['Date', 'Y_365'])


# In[55]:


# Remember that we have copied above the main dataframe, so now we use it and merge with our explanatory variable.
rets_cl_365 = pd.merge(left = rets_cl_365, right = Y_365, how = 'inner', on = 'Date', suffixes=(False, False))
rets_cl_365 = rets_cl_365[rets_cl_365['Gold-t-250'].notna()]
rets_cl_365 = rets_cl_365.drop(np.arange(start = 2244, stop = 2609, step = 1), axis = 0)


# In[56]:


rets_cl_365.tail()


# In[133]:


# Pycaret - Time to run the SVM classification after data preperation
from pycaret.classification import *
initialize = setup(rets_cl_365, target ='Y_365', train_size = 0.70, 
                   ignore_features=['Date'], session_id=11, silent=True, 
                   remove_outliers=True, outliers_threshold=0.02, 
                   remove_multicollinearity=False, multicollinearity_threshold=0.85)  

compare = compare_models(turbo=False, include=['rbfsvm', 'svm'])


# In[134]:


# Let's create the two models for later evaluation. In addition, one can clearly see the output of 10 cross validations.
svm_lin_365 = create_model('svm', fold=5)


# In[135]:


svm_rbf_365 = create_model('rbfsvm', fold=5)


# In[136]:


svm_rbf_365_tuned = tune_model(svm_rbf_365, fold=5)


# In[61]:


# Note that the Accuracy of radial SVM has increased from 68% to 91.5% after tuning the hyperparameters of the model. 
# In addition, the accuracy of linear SVM is almost the same as for the radial SVM, which is 92%. One can play with 
# other classification algorithms as well, since Pycaret offers a bunch of them.


# In[62]:


####### Artificial Neural Network for 30 day horizon. 
# For ANN, we decided to do a categorical prediction, which is that we assigned the Y variable to 0s and 1s according to the 
# same logic as explained above. The target horizon is again 30 days, so we would like to know whether the price in 30 days 
# will be higher or lower in relative to the current price.


# In[63]:


# Take a look at our original dataframe that we created for classification.
rets_cl.head()


# In[109]:


x_var = rets_cl.iloc[:, 1:-1].values # defining X variables
y_var = rets_cl.iloc[:, -1].values   # defining Y variable which is the last column of our dataframe


# In[110]:


# Dividing the dataset into training and testing sets
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.30, random_state = 11)


# In[111]:


# Feature scaling is of utmost importance in ANN. In Pycaret we have not done it, since it is automatically done on the background.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[112]:


# Importing the tensorflow library 
import tensorflow as tf


# In[113]:


# Initializing the ANN model 
ann = tf.keras.models.Sequential()


# In[114]:


# Adding the input layer and the first hiddel layer 
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


# In[115]:


# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[116]:


# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[117]:


# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[118]:


# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 150)


# In[119]:


# Making the prediction with our established ANN
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[120]:


# Compiling the Confusion Matrix
# The ANN performed better than SVM according to the accuracy metric for 30 day horizon. The Accuracy is on average 90%. One can 
# also observe the confusion matrix created below. For more comments please see the paper.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

