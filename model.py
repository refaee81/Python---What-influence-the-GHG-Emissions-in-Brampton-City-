# Spyder project settings
.spyderproject
.spyproject

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:28:23 2018

@author: ramsey
"""
###Target: ONTARIO REGULATION 397/11
""" On or before July 1, 2019 and on or before every fifth anniversary thereafter, every public agency shall publish on its website and intranet site, if it has either or both, and make available to the public in printed form at its head office all of the information that is required to be published and made available under subsection (1), the Energy Consumption and Greenhouse Gas Emission Template that is required to be submitted and published on or before July 1 of that year and the following information:
	1.	A description of current and proposed measures for conserving and otherwise reducing energy consumption and managing its demand for energy.
	2.	A revised forecast of the expected results of the current and proposed measures. 
	3.	A report of the actual results achieved.
	4.	A description of any proposed changes to be made to assist the public agency in reaching any targets it has established or forecasts it has made. """

### Data preprocessing & Cleaning 
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\Sr. Advisor') 


check=pd.read_excel("Energy Consumption.xlsx", na_values='')

list(check.columns.values)

consumption = pd.read_excel("Energy Consumption.xlsx", na_values='', usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
                           header=0, names=['INTERNAL_GROSS_AREA_SQ_FT', 'cWEEKLY_HOURS_OF_OPERATION', 'ELECTRICITY_KWH', 
                                            'NATURAL_GAS_M3', 'TOTAL_ENERGY_EKWH','ENERGY_INTENSITY_EKWH_SQ_FT', 
                                            'GHG_EMISSIONS_KG', 'GHG_INTENSITY_KG_SQ_FT', 'REPORT_YEAR', 'SITE_NAME', 'SITE_TYPE', 'ADDRESS'])



###################################################### Preprocssing & Cleaning Data 

len(consumption['SITE_TYPE'].value_counts())

consumption['SITE_TYPE'].unique()### Dependent Variable: Forcasting the GHG_EMISSIONS_KG 

consumption= consumption.fillna({'INTERNAL_GROSS_AREA_SQ_FT': consumption.INTERNAL_GROSS_AREA_SQ_FT.mean(),
                                 'cWEEKLY_HOURS_OF_OPERATION': consumption.cWEEKLY_HOURS_OF_OPERATION.mean(),
                                 'ELECTRICITY_KWH': consumption.ELECTRICITY_KWH.mean(),
                                 'NATURAL_GAS_M3': consumption.NATURAL_GAS_M3.mean(),
                                 'TOTAL_ENERGY_EKWH': consumption.TOTAL_ENERGY_EKWH.mean(),
                                 'ENERGY_INTENSITY_EKWH_SQ_FT': consumption.ENERGY_INTENSITY_EKWH_SQ_FT.mean(),
                                 'GHG_EMISSIONS_KG': consumption.GHG_EMISSIONS_KG.mean(), 
                                 'GHG_INTENSITY_KG_SQ_FT': consumption.GHG_INTENSITY_KG_SQ_FT.mean()})
    
                        
from collections import Counter

data = Counter(consumption['ADDRESS'])
data.most_common()   # Returns all unique items and their counts
data.most_common(1) 

consumption['REPORT_YEAR'] = consumption['REPORT_YEAR'].fillna(2015) 
consumption['SITE_NAME'] = consumption['SITE_NAME'].fillna('Chris Gibson Recreation Centre') 
consumption['SITE_TYPE'] = consumption['SITE_TYPE'].fillna('Indoor recreational facilities') 
consumption['ADDRESS'] = consumption['ADDRESS'].fillna('9050 BRAMALEA RD') 

consumption.isnull().values.any()

consumption.isnull().values.sum()

### Checking descriptive statistics by visualizing distribution and other statistics 

from matplotlib import pyplot
pyplot.hist(consumption.GHG_EMISSIONS_KG)
pyplot.ylabel('Frequency')
pyplot.xlabel('GHG_EMISSIONS_KG')
pyplot.title('GHG_EMISSIONS_KG')
pyplot.show()


import matplotlib.pyplot as plt

import seaborn as sns
corr = consumption.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

pd.scatter_matrix(consumption, alpha = 0.3, figsize = (14,8), diagonal = 'kde')


### Checking descriptive statistics by visualizing distribution and other statistics 
from matplotlib import pyplot
pyplot.hist(consumption.ELECTRICITY_KWH)
pyplot.ylabel('Frequency')
pyplot.xlabel('Consumption')
pyplot.title('ELECTRICITY_KWH Consumption')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.INTERNAL_GROSS_AREA_SQ_FT)
pyplot.ylabel('Frequency')
pyplot.xlabel('INTERNAL_GROSS_AREA_SQ_FT')
pyplot.title('INTERNAL_GROSS_AREA_SQ_FT')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.cWEEKLY_HOURS_OF_OPERATION, line='s')
pyplot.title('cWEEKLY_HOURS_OF_OPERATION')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.cWEEKLY_HOURS_OF_OPERATION)
pyplot.ylabel('Frequency')
pyplot.xlabel('cWEEKLY_HOURS_OF_OPERATION')
pyplot.title('cWEEKLY_HOURS_OF_OPERATION')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.TOTAL_ENERGY_EKWH, line='s')
pyplot.title('TOTAL_ENERGY_EKWH')
pyplot.show()


from matplotlib import pyplot
pyplot.hist(consumption.TOTAL_ENERGY_EKWH)
pyplot.ylabel('Frequency')
pyplot.xlabel('TOTAL_ENERGY_EKWH')
pyplot.title('TOTAL_ENERGY_EKWH')
pyplot.show()


import seaborn as sns
sns.pairplot(consumption, x_vars=['ELECTRICITY_KWH','NATURAL_GAS_M3','TOTAL_ENERGY_EKWH'], y_vars='GHG_EMISSIONS_KG', size=7, aspect=0.7, kind='reg')

#################################################### OLS Models 

Y2017 = consumption.loc[consumption['REPORT_YEAR'] == 2017]
Y2016 = consumption.loc[consumption['REPORT_YEAR'] == 2016]
Y2015 = consumption.loc[consumption['REPORT_YEAR'] == 2015]
Y2014 = consumption.loc[consumption['REPORT_YEAR'] == 2014]


def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded


SITE_TYPE_bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
SITE_TYPE_labels = {"Indoor recreational facilities",
                    "Administrative offices and related facilities, including municipal council chambers",
                    "Community centres",
                    "Storage facilities where equipment or vehicles are maintained, repaired, or stored",
                    "Public libraries",
                    "Indoor sports arenas",
                    "Other",
                    "Parking garages",
                    "Fire stations and associated offices and facilities",
                    "Performing arts facilities",
                    "Indoor ice rinks",
                    "Cultural facilities",
                    "Gyms and indoor courts for playing tennis, basketball or other sports"}              


consumption['SITE_TYPE_Labels'] = coding(consumption['SITE_TYPE'], {"Indoor recreational facilities": 0,
                    "Administrative offices and related facilities, including municipal council chambers": 1,
                    "Community centres": 2,
                    "Storage facilities where equipment or vehicles are maintained, repaired, or stored": 3,
                    "Public libraries": 4,
                    "Indoor sports arenas": 5,
                    "Other": 6,
                    "Parking garages": 7,
                    "Fire stations and associated offices and facilities": 8,
                    "Performing arts facilities": 9,
                    "Indoor ice rinks": 10,
                    "Cultural facilities": 11,
                    "Gyms and indoor courts for playing tennis, basketball or other sports": 12})
                     
consumption['SITE_TYPE_CAT'] = pd.cut(consumption.SITE_TYPE_Labels, SITE_TYPE_bins, labels = SITE_TYPE_labels, right=False)

### Steps for Implementing VIF: multicolinearity testing & handling 
"""Run a multiple regression.
Calculate the VIF factors.
Inspect the factors for each predictor variable, 
if the VIF is between 5-10, multicolinearity is likely present and you should consider dropping the variable."""

X = consumption.iloc[:, [4,5,6,7,8,11]].values

y = consumption.iloc[:,[9]].values 

from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

y, X = dmatrices('GHG_EMISSIONS_KG  ~  ELECTRICITY_KWH + NATURAL_GAS_M3 + TOTAL_ENERGY_EKWH +  REPORT_YEAR', data = consumption)

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vif["VIF Factor"])


import statsmodels.formula.api as smf

#### Best model without collinearity ## only 4 ID 
Consumption_OLS = smf.ols('GHG_EMISSIONS_KG  ~  ELECTRICITY_KWH + NATURAL_GAS_M3 + TOTAL_ENERGY_EKWH +  REPORT_YEAR', data=consumption).fit()
print(Consumption_OLS.summary())
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(Consumption_OLS.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.savefig('output.png')
plt.show()


##### Regularization of OLS
list(consumption.columns.values)


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0) 

X = consumption.iloc[:, [0, 2,3,4,8]].values

y = consumption.iloc[:,[6]].values 


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import math

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))


### Cross-v

X = consumption.iloc[:, [0, 2,3,4,8]].values

y = consumption.iloc[:,[6]].values 


from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X,y)
y_pred = classifier.predict(X_test)
y_pred2= classifier.predict(X)
print("y_pred = classifier.predict(X_test): ", y_pred)
classifier.score(X_test, y_pred)


### K Fold Cross Validation 

from sklearn.model_selection import KFold

kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator = classifier, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies_mean = accuricies.mean()

### Conduct Grid Search To Find Parameters Producing Highest Score

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

model = Ridge()
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X, y)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

##########################################################PolyNomial 

from math import sqrt
from sklearn.preprocessing import PolynomialFeatures


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Transformation and Regressioin with Degree-3

poly = PolynomialFeatures(degree = 3)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred1 = lin_reg.predict(X_test_poly)
print("Polynomial Regression Score with Degree-3 : ",lin_reg.score(X_test_poly, y_test))
print("Polynomial Regression MSE with Degree-3 : ",mean_squared_error(y_test,y_pred1))
print("Polynomial Regression RMSE with Degree-3: ", sqrt(mean_squared_error(y_test,y_pred1)))


#### polynomial gridsearch 

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm, grid_search

import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline 

def PolynomialRegression(degree=2, **kwargs): 
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs)) 

param_grid = {'polynomialfeatures__degree': np.arange(10), 'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]} 


poly_grid = GridSearchCV(PolynomialRegression(), param_grid, 
                         cv=5, 
                         scoring='r2', 
                         verbose=3) 

poly_grid.fit(X, y)## wait little bit 
print(poly_grid)
print(poly_grid.best_score_)

##############################################################random forest regression 

from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error 

X = consumption.iloc[:, [0, 2,3,4,8]].values

y = consumption.iloc[:,[6]].values 



reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred_treeforet = reg.predict(X_test)
print("Random Forest Regressor Score : ", reg.score(X_test, y_test))
print("Random Forest RMSE : ", sqrt(mean_squared_error(y_test,y_pred_treeforet)))
print("Random Forest MSE : ",mean_squared_error(y_test,y_pred_treeforet))

#K fold corss validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(reg, X,y, cv=10)
print(accuracies)
accuracies.mean()### K-fold of Random Forest 
accuracies.std()

# Random forest Gridsearch 

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rfc=RandomForestClassifier(random_state=42)


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 2)
CV_rfc.fit(X_train, y_train)## wait little bit 

CV_rfc.best_params_
print(CV_rfc.best_score_)

######################################## SVR Model 

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
import csv 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split 

X = consumption.iloc[:, [0, 2,3,4,8]].values

y = consumption.iloc[:,[6]].values 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0) 


svr = SVR(kernel="rbf", gamma=0.01)
log = LinearRegression()
svr.fit(X_train,y_train)
log.fit(X_train, y_train)

predSVR = svr.predict(X_test)
predLog = log.predict(X_test)

svr.score(X_test, predSVR)
log.score(X_test, predLog)


plt.plot(X_test, y_test, label='true data')
plt.plot(X_test, predSVR, 'co', label='SVR')
plt.plot(X_test, predLog, 'mo', label='LogReg')
plt.legend()
plt.show()#### svr uses kernel rbf which is distant far from true values, and reflect bad predicition despite its accuracy 

## SVR gridsearch 

from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0) 
parameters = {'kernel':['linear'], 'C':np.logspace(np.log10(0.001), np.log10(200), num=20), 'gamma':np.logspace(np.log10(0.01), np.log10(2), num=30)}
svr_red = svm.SVR()
grid_searcher_red = GridSearchCV(svr_red, parameters, n_jobs=6, verbose=3)
     SVR.best_params_
print(SVR.best_score_)



##########################################
