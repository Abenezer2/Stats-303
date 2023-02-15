# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:26:24 2023

@author: Emre
"""

# Last part regarding Linear Regression

# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import statsmodels.formula.api as smf

# Training data
trainf = pd.read_csv('Car_features_train.csv')
trainp = pd.read_csv('Car_prices_train.csv')
train = pd.merge(trainf, trainp)
# Test data
testf = pd.read_csv('Car_features_test.csv')
testp = pd.read_csv('Car_prices_test.csv')


# One main thing left to do with Lin Reg.: The Potential Underlying Issues
# 6 of them - 3.3 in the textbook
# What we will discuss the rest of today and Monday

# Let's train with a model with more data engineering

# **2 without I only means interactions
ols_object = smf.ols(formula='price~(year+engineSize+mileage+mpg)**2 + I(mileage**2)', data=train)
model = ols_object.fit()
print(model.summary())

# **2 gives all possible interactions - 6 different permutations
# We also have a quadratic term

# Highly non-linear, check for overfitting

print(np.sqrt(model.mse_resid))

pred_price = model.predict(testf)
print(np.sqrt(((testp['price']-pred_price)**2).mean()))

# Comparable RSE and RMSE, good


# 1) Captured non-linearity between the predictors and the response

    # After creating new terms with transformations/interactions,
    # we assume there is a linear assoc. between the response and 
    # all the predictors we use (original ones, interaction/transf. ones)
    
# The linearity assumption - does not always hold
# There might be uncaptured non-linearity in the relation
# How to check this? Best way is a residual plot.

# res = true responses - predicted responses

sns.scatterplot(x = model.fittedvalues, y = model.resid, color='orange')
sns.lineplot(x = [pred_price.min(),pred_price.max()],y = [0,0],color = 'blue')
plt.xlabel('Predicted price')
plt.ylabel('Residual')
plt.show()

# What we need to look for is patterns:
    # If there is no clear pattern - like a quadratic cluster in the MLR code
        # that means non-linearity is captured.
        
# If you use a model without inter./transf. and no pattern --> Linearity assumption is held
    # If a pattern - use interactions/transformation --> The res plot should look more random
    # While checking, you need to focus on the main cluster

# Just like coming up with good inter./transf. predictors, detecting and dealing with
# there underlying issues, like the linearity assumption, is based on visual inspection,
# it is partially subjective. In other words, you need to get back to the
# "engineering" part of data engineering


# Looking at the res plot, another thing we can find out is if the linearity assumption
# is PERFECTLY met - ideal scenario, very very uncommon to find with data from real life.

# If the res plot is symmetric across the x-axis, that means the assumption is perfectly met.

# 2) Non-constant variance of error terms - heteroscedasticity

# When we wrote the equation for a lin reg model: Y = f(X) + epsilon 
    # We made an assumption about epsilon - a RV with constant var.: Var(epsilon) = sigma**2
    # Many calculations for inference such as st. err., CI, PI, stat. sig. depend on this assumption
    # Most of the time, it does not hold perfetly - how severe, we need to check
    
# To check, again the res plot

# We can see that as the pred. value increases, the var of residue increases
# The model is heteroscasdic

# The common way to address: Transform the response
    # Y --> np.log(Y) to shrink the larger values more
    
# Train a model with the log responses

ols_object = smf.ols(formula='np.log(price)~(year+engineSize+mileage+mpg)**2 + I(mileage**2)', data=train)
model_log = ols_object.fit()
print(model_log.summary())


# R2 0.73 --> 0.8, better inference

# Let's see the res plot again
sns.scatterplot(x = model_log.fittedvalues, y = model_log.resid, color='orange')
sns.lineplot(x = [model_log.fittedvalues.min(),model_log.fittedvalues.max()],y = [0,0],color = 'blue')
plt.xlabel('Predicted price')
plt.ylabel('Residual')
plt.show()

# Res var looks similar with random fluctuations.

# Note: Maintaining homoscedasticity is important for inference
# It is not expected to contribute to prediction

# Let's see the test RMSE - converting the pred values back with np.exp

pred_price_log = model_log.predict(testf)
print(np.sqrt(((testp['price']-np.exp(pred_price_log))**2).mean()))


# Above 9k - Higher RMSE compared to the non log model
    # We sacrificed some prediction accuracy to guarantee better inference
    # Remember that RMSE is OVERALL metric - does it mean that the pred got worse for
        # most test obs?
    # Let's make a point-by-point comparison
    
# Error for each test obs in the non-log model
err = np.abs(testp.price - pred_price)

# Error for each test obs in the log model
err_log = np.abs(testp.price - np.exp(pred_price_log))
    
# compare the test data errors visually and numerically

sns.scatterplot(x = err, y = err_log, color='orange')
sns.lineplot(x = [0, 100000], y = [0, 100000], color= 'blue')
plt.plot()

# The percentage of obs or which the log model is predicting better

print(np.sum(err_log < err)/len(err)) # 0.557
    
# 56% of the test obs, the log model predicts better
# For the obs that the non-log model predicts better, the prediction is a lot better than the log model
    
# This demonstrates that sometimes, you need to look beyond the overall metrics, like RMSE


# Wrapping up with a visual comparison between price (Y) and log price (log(Y))

#Visualizing the distribution of price and log(price)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.2)
sns.set(rc = {'figure.figsize':(20,12)})
sns.set(font_scale = 2)
ax = fig.add_subplot(2, 2, 1)
sns.histplot(train.price,kde=True)
ax.set(xlabel='price', ylabel='Count')
ax = fig.add_subplot(2, 2, 2)
sns.histplot(np.log(train.price),kde=True)
ax.set(xlabel='log price', ylabel='Count')
plt.show()    
    
    
# Visualizes how the log changes the variation in Y
# It offsets the effect of very high prices - making the dist.
                            # closer to Normal (Gaussian)
# The skewed tail of the price dist is gone - easier to be sure for inference

# Another way to improve inference could have been adding more predictors
    # Like car model (never used so far)
    # If you do not have such a predictor - just transform your response
    
################ Midterm up to here #########################################



    
    









