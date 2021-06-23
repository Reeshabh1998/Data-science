#Problem Statement: -

#A logistics company recorded the time taken for delivery and the time taken for the sorting of the items for delivery. Build a Simple Linear Regression model to find the relationship between delivery time and sorting time with delivery time as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient values for different models.


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\360Digi\Simple Resgression Ass\\delivery_time.csv")

df.describe()

df.columns.values[0] = "DT"
df.columns.values[1] = "ST"
df.columns


# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)




EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}


plt.figure(figsize=(30, 30))
sns.pairplot(df, hue='DT', height=3, diag_kind='hist')

#yes or no count
sns.catplot('DT', data=df, kind='count')

import matplotlib.pyplot as plt

plt.figure(figsize = (12, 8))
ax = sns.boxplot(data = df, orient = 'h', palette = 'Set2')
plt.title('Boxplot overview dataset')
plt.xlabel('values')
plt.xlim(-3, 50)
plt.show()

plt.figure(figsize = (12, 8))
sns.heatmap(df.corr(), annot = True)
plt.title('Correlation matrix')
plt.show()

sns.pairplot(df) 

#Bar plot
df['ST'].value_counts().plot.bar()
'''
# Normalization function using z std. all are continuous data.
def std_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
cal = std_func(df)
cal.describe()
'''
cal = df

#Data Modeling

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#plt.bar(height = cal.ST, x = np.arange(1, 110, 1))
plt.hist(cal.ST) #histogram
plt.boxplot(cal.ST) #boxplot

#plt.bar(height = cal.CC, x = np.arange(1, 110, 1))
plt.hist(cal.DT) #histogram
plt.boxplot(cal.DT) #boxplot

# Scatter plot
plt.scatter(x = cal.ST, y = cal.DT, color = 'green') 

# correlation
np.corrcoef(cal.ST, cal.DT) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(cal.ST, cal.DT)[0, 1]
cov_output



# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('DT ~ ST', data = cal).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(cal.ST))

# Regression Line
plt.scatter(cal.ST, cal.DT)
plt.plot(cal.ST, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = cal.DT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(cal.ST), y = cal.DT, color = 'brown')
np.corrcoef(np.log(cal.ST), cal.DT) #correlation

model2 = smf.ols('DT ~ np.log(ST)', data = cal).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(cal.ST))

# Regression Line
plt.scatter(np.log(cal.ST), cal.DT)
plt.plot(np.log(cal.ST), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = cal.DT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
'''
'''
#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = cal.ST, y = np.log(cal.DT), color = 'orange')
np.corrcoef(cal.ST, np.log(cal.DT)) #correlation

model3 = smf.ols('np.log(DT) ~ ST', data = cal).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(cal.ST))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(cal.ST, np.log(cal.DT))
plt.plot(cal.ST, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = cal.DT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = cal).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(cal.ST))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cal.iloc[:, 1:].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(cal.ST, np.log(cal.DT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = cal.DT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4



# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(cal, test_size = 0.2)

finalmodel = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.DT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT


# Model Evaluation on train data
train_res = train.DT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse




# Model having highest R-Squared value is better i.e. (model=0.782 is better than model1=0.682). There has good relationship>0.85




















