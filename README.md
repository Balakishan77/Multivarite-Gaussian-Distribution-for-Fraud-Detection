# Fraud-Detection-using-Gaussian-Normal-Distribution

This dataset contains transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly skewd, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. We do not have the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are ‘Time’ and ‘Amount’.

Feature ‘Time’ is the number of seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction Amount. Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Anomaly Detection Alogorithm:

## Algorithm Selection:
1) For this dataset, we are going to use multivariate normal probability density function, since it automatically generates the relationships (correlation) between variables to calculate the probabilities. So we don’t need to derive new features. As the features are outcome of PCA, it is difficult for us to understand the relationship between these features.
2) However multivariate normal probability density function is computationally expensive compared to normal Gaussian probability density function. On very large datasets, we might have to prefer Gaussian probability density function instead of multivariate normal probability density function to speed up the process and do feature engineering based on the subject matter expertise.
## Data Preprocessing:
For training and evaluating Gaussian distribution algorithms, we are going to split the train, cross validation and test data sets using below ratios.
1) Train: 60% of the Genuine records (y=0), no Fraud records(y=1). So the training set will not have a label as well.
2) CV: 20% of the Genuine records (y=0), 50% of the Fraud records(y=1)
3) Test: Remaining 20% of the Genuine records(y=0), Remaining 50% of the Fraud records(y=1)

1) Fit the model p(x) on training set.

2)We use cross validation to choose the threshold parameter epsilon using the evaluation metrics Preceion/Recall, F1-score.

3) On cross validation/test data, predict
     y = 1 if p(x) < epsilon (anomaly)
     y = 0 if p(x) >= epsilon (normal)

### Feature Selection:
1) Features that we choose for these algorithms have to be normally distributed. Otherwise we need to transform the features to normal distribution using log, sqrt etc.
2) Choose features that might take on unusually large or small values in the event of an anomaly. We looked at the distribution in the beginning using distplot. So it is wise to choose features which have completely different distribution for fraud records compared to genuine records.
          ![feature_distribution](https://user-images.githubusercontent.com/40944675/46391148-f9352980-c6f8-11e8-809b-71347e6a0885.png)
##### Feature Importance:
Lets use Feature importance to get rid of unwanted features whose existance will not improve our prediction model. 
I have used random forest classifier to identify the influential fetures. 
     ![feature_importance](https://user-images.githubusercontent.com/40944675/46391523-0f43e980-c6fb-11e8-9d72-8b88371fc15c.png)
1) From the Distplot we can see Normal Distribution of anomalous transactions (class = 1) is matching with Normal Distribution of genuine transactions (class = 0) for V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8' features. It is better to delete these features as they may not be useful in finding anomalous records. 
2) Time is also not useful variable since it contains the seconds elapsed between the transaction for that record and the first transaction in the dataset. So the data is in increasing order always.
