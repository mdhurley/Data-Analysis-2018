# Data-Analysis-2018-

 
# coding: utf-8

# # Hypothesis

# Going off of the GDP Equation:
#     
# \begin{align}
# GDP = C + G + I + NX,
# \end{align}
# 
# where C represents personal consumption, G for government expenditures, I for investments, and NX for net exports. Using this equation as a proxy for overall economic health on a national level we can see how certain effects in the real world have on GDP.

# ### Multiple Linear Regression

# In[2]:


## Importing the libraries

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt


# In[3]:


## Importing the dataset

df = pd.read_csv('econometrics_gdp_quarterly.csv', delimiter=',')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, 7].values

irate = X[:, [1]]
sp500 = X[:, [2]]


# In[4]:


## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[5]:


## Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Building the optimal model using Backward Elimination

# In[6]:


import statsmodels.formula.api as sm
X =  np.append(arr = np.ones((39, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Looking at the T & P values we can see that the first two independent variables, the S&P 500 and the 10-year Treasury maturity rate, are statistically significant. This is shown in their P values being far below 5% thus allowing us to reject the null hypothesis. On the other hand the last two exogenous variables, government expenditures and the trade balance, seem to not be statistically significant due to both having a P-value greater than 5%.
# 
# Proceeding forward we remove the trade balance variable to conduct our initial backwards elimination process:

# In[7]:


import statsmodels.formula.api as sm
X =  np.append(arr = np.ones((39, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# By taking out net exports 

# In[8]:


from numpy import genfromtxt
data = genfromtxt('econometrics_gdp_quarterly.csv', delimiter=',')
plt.plot(data[:, 1], data[:, 2], linestyle='dashed')
plt.title('Interest Rates Vs S&P 500')
plt.ylabel('Interest Rates')
plt.xlabel('S&P 500')


# In[9]:


from pandas.plotting import scatter_matrix
scatter_matrix(df)

plt.show()


# In[10]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

