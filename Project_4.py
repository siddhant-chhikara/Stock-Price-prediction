import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Project_4_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(mdates.date2num)

y = df.iloc[:, 1].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(df['Date'].values.reshape(-1,1), y, test_size=0.01, random_state=0)
model = LinearRegression()  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

stocks = ['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']
df_melt = df.melt(id_vars='Date', value_vars=stocks, var_name='Stock', value_name='Price')

plt.figure(figsize=(12, 6))
sns.boxplot(x='Stock', y='Price', data=df_melt)
plt.title('Box plot of Stocks and Prices')
plt.show()