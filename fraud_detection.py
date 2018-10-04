#Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Reading the dataset
dataset = pd.read_csv(r'C:\Users\admin\Desktop\ML\fraud detection\creditcard.csv')
dataset.describe()
dataset.columns.values

# Visualizing the all feature distributions (Distplots) of Genuine and Fraud class 
import seaborn as sns
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12,31*4))
gs = gridspec.GridSpec(31,1)
for i, col in enumerate(dataset.columns):
    ax = plt.subplot(gs[i])
    sns.distplot(dataset[col][dataset['Class']==0],color='g',label='Genuine Class')
    sns.distplot(dataset[col][dataset['Class']==1],color='r',label='Fraud Class')
    ax.legend()
plt.show()

'''
 I have shown the distplot of all the features with labels 1 and 0 in different color. 
 Its not necessary to have clear distinct distribution of numbers not overlapping each other. 
 but V14 and V11 it is one of the most important features where we see two distribution with 2 peaks,
 if you see distplot of other features,you will find 2 different distributions with their
 peaks overlapping each other for label 1 and 0 and hence i dropped them.
v_features = dataset.columns
plt.figure(figsize=(12,31*4))
gs = gridspec.GridSpec(31,1)
X1=(1,1,1,1,1,1) 
X2=(1,0,1,0,1,0)
for i, col in enumerate(dataset.columns):
    ax = plt.subplot(gs[i])
    sns.distplot(X1,color='g',label='Genuine Class')
    sns.distplot(X2,color='r',label='Fraud Class')
    ax.legend()
plt.show()
'''
#Selecting features that add value in calculating gaussian distribution based on Feature Importance score and plotting the feature score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
model_rf = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy',random_state = 0)
model_rf.fit(dataset.iloc[:,1:29], dataset.iloc[:,30]);
feature_importance = model_rf.feature_importances_
y, x = [list(x) for x in zip(*sorted(zip(feature_importance, dataset.iloc[:,1:29].columns), reverse = True))]
plt.figure(figsize=(16, 6))
plt.bar(range(len(feature_importance)), y, align='center')
plt.xticks(range(len(feature_importance)), x, rotation='vertical')
plt.title('Feature importance')
plt.ylabel('Feature Importance Score')
plt.xlabel('Features')
plt.show()
unimportant_features=[i for i,j in zip(x,y) if j<0.020] #features below threshold value -> ['V19','V8','V21','V26','V20','V1','V27','V2','V6','V28','V15','V5','V13','V22','V25','V23','V24']
dataset.drop(unimportant_features, axis=1, inplace=True)
dataset.drop(labels = ["Amount","Time"], axis = 1, inplace = True)#remoing Amount and Time features since they wont add much value in calculating gaussian distribution.

'''final features in dataset ['V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18','Class']'''

'''We can also do Feature selection using SelectFromModel(SelectFromModel is a meta-transformer that can be used along with any estimator 
that has a coef_ or feature_importances_ attribute after fitting. The features are considered unimportant and removed given any threshold)

#Feature scores using XGBClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
model_xg = XGBClassifier(n_estimators=100)
model_xg.fit(dataset.iloc[:,1:29], dataset.iloc[:,30])
plot_importance(model_xg)
plt.figure(figsize=(20, 6))
plt.show()
model = SelectFromModel(model_xg, prefit=True)
X = model.transform(dataset.iloc[:,1:29],threshold=0.020)
  '''
  
''' Spliting the dataset into 
            training set: 60% of the Genuine records (y=0),
            cross validation set:  20% of the Genuine records (y=0), 50% of the Fraud records(y=1) 
            test set: Remaining 20% of the Genuine records(y=0), Remaining 50% of the Fraud records(y=1)'''
genuine_set = dataset[dataset["Class"] == 0]
anamoly_set = dataset[dataset["Class"] == 1]

genuine_len = len(genuine_set)#284315
anamoly_len = len(anamoly_set)#492

anamoly_cv  = anamoly_set [: (anamoly_len//2)] #(246, 12)
anamoly_test = anamoly_set [(anamoly_len//2):anamoly_len]#246, 12)

start_mid = (genuine_len * 60) // 100 

genuine_cv_test_len=(genuine_len-start_mid)//2 
start_midway = start_mid + genuine_cv_test_len #227452
genuine_train = genuine_set [:start_mid] #60% of Genuine records -> (170589, 12)
genuine_cv = genuine_set [start_mid:start_midway]  #20% of the Genuine records+50% of the Fraud records(y=1) -> #(56863, 12)
genuine_test = genuine_set [start_midway:genuine_len]  #(56863, 12)

X_cv = pd.concat([genuine_cv,anamoly_cv],axis=0) #(57109, 12) features-['V3' 'V4' 'V7' 'V9' 'V10' 'V11' 'V12' 'V14' 'V16' 'V17' 'V18']
X_test = pd.concat([genuine_test,anamoly_test],axis=0) #(57109, 12)

y_cv = X_cv["Class"] #57109
y_test = X_test["Class"]  #57109
genuine_train.drop(labels = ["Class"], axis = 1, inplace = True) #Traing set-(170589, 11))
X_cv.drop(labels = ["Class"], axis = 1, inplace = True) #Cross validation set-(57109, 11)
X_test.drop(labels = ["Class"], axis = 1, inplace = True) #Test set-(57109, 11)

#Fitting Multivaritae Gaussian distribution
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, recall_score, precision_score
mu = np.mean(genuine_train, axis=0) #Mean vector
sigma = np.cov(genuine_train.T) #Covariance vector

def multivariateGaussian(X,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(X)
p = multivariateGaussian(genuine_train,mu,sigma)
p_cv = multivariateGaussian(X_cv,mu,sigma)
p_test = multivariateGaussian(X_test,mu,sigma)
#Finding threshold value(epsilon) using cross validation set to flag anamolaies
#epsilons = np.arange(min(p_cv),max(p_cv), 0.00105828e-4) #Started with this and tried different values to find threshold
epsilons = np.arange(min(p_cv),1.05828e-41,0.0000000000000000000000000000000000105828e-10)
best_epsilon = 0
best_f1 = 0
f = 0
farray = []
Recallarray = []
Precisionarray = []
for epsilon in epsilons:
    predictions = (p_cv < epsilon)
    f = f1_score(y_cv, predictions, average = "binary")
    Recall = recall_score(y_cv, predictions, average = "binary")
    Precision = precision_score(y_cv, predictions, average = "binary")
    farray.append(f)
    Recallarray.append(Recall)
    Precisionarray.append(Precision)
    if f > best_f1:
        best_f1 = f #0.7540983606557378
        best_recall = Recall#0.8414634146341463
        best_precision = Precision#0.6831683168316832
        best_epsilon = epsilon    #1.05828e-45
fscore, ep= best_f1, best_epsilon #best f1score->0.7540983606557378 ,best threshold value ->1.05828e-45

#Predicting anaomalies if P(x)<threshold in test set 
predictions = (p_test < ep)
Recall = recall_score(y_test, predictions, average = "binary")    #0.768292682926829
Precision = precision_score(y_test, predictions, average = "binary")#0.6494845360824743
F1score = f1_score(y_test, predictions, average = "binary")   #0.7039106145251396

#Predicting anaomalies if P(x)<threshold in cross validation set
predictions = (p_cv < ep)
Recall = recall_score(y_cv, predictions, average = "binary")    #0.8414634146341463
Precision = precision_score(y_cv, predictions, average = "binary")#0.6831683168316832
F1score = f1_score(y_cv, predictions, average = "binary")#0.7540983606557378
