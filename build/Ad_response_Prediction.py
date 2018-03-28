
#%%==============================================================================
# ################# Ad response PREDICTION #################
# ################# Ad response PREDICTION #################
# ################# Ad response PREDICTION #################
#==============================================================================

#%% ==============================================================================
# # Importing the libraries
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


#%%==============================================================================
# # Importing the dataset
#==============================================================================
import os
os.chdir("D:/Knowledge/github/Ad_response-prediction/data")
ad_data  = pd.read_csv("advertising.csv")
ad_data.head()
ad_data.info()
ad_data.describe()

#%%==============================================================================
# # Missing Value Check ##
#==============================================================================
plt.figure(figsize=(22,14))
sns.heatmap(ad_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


#%%==============================================================================
# # Visualizing the Data ##
#==============================================================================

# 
# ** Create a histogram of the Age**
sns.distplot(ad_data['Age'], kde=False)


# **Create a jointplot showing Area Income versus Age.**
sns.jointplot('Area Income', 'Age', data=ad_data, kind='hex')


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**
sns.jointplot("Daily Time Spent on Site", 'Age', kind='kde', data=ad_data)


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**
sns.jointplot("Daily Time Spent on Site", 'Daily Internet Usage', kind='hex', data=ad_data)


# ** create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**
sns.pairplot(ad_data, hue='Clicked on Ad')


# ** create a heat map for correlation matrix
sns.heatmap(ad_data.corr(), cmap='coolwarm', annot=ad_data.corr())



#%%==============================================================================
# # Identify X and y ##
#==============================================================================


X = pd.get_dummies(ad_data.drop(['Clicked on Ad', 'Timestamp', 'Ad Topic Line', 'City'], axis=1),prefix=None, prefix_sep='_', drop_first=True)
X.head()


y = ad_data['Clicked on Ad']
y.head()



#%%==============================================================================
# # Splitting the dataset into the Training set and Test set
#==============================================================================


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.head()


#%%==============================================================================
# # Fitting Logistic Regression to the Training set
#==============================================================================


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred


#%%==============================================================================
# # MODEL EVALUATION ##
#==============================================================================


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#%%########## classification report ##################


print(classification_report(y_test, y_pred))

#%%########## confision matrix ##################

confusion_matrix(y_test, y_pred)

#%%########## accuracy score ##################

accuracy_score(y_test, y_pred)
