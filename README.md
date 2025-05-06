# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING



### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")
data = pd.read_csv('/content/drive/MyDrive/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
passengers_data = data[['#Passengers']]
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original #Passengers Data')
plt.title('Original Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
rolling_mean_5 = passengers_data['#Passengers'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['#Passengers'].rolling(window=10).mean()
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
data_monthly = data.resample('MS').sum()
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data = scaled_data + 1
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
plt.figure(figsize=(12, 6))
ax = train_data.plot(label='Train')
test_data.plot(ax=ax, label='Test')
test_predictions_add.plot(ax=ax, label='Forecast')
ax.legend()
ax.set_title('Visual Evaluation - Train vs Test vs Forecast')
plt.grid()
plt.show()
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Squared Error (RMSE):", rmse)
print("Standard Deviation (approx):", np.sqrt(scaled_data.var()))
print("Mean:", scaled_data.mean())
model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=12)
plt.figure(figsize=(12, 6))
ax = data_monthly.plot(label='Historical Data')
predictions.plot(ax=ax, label='Future Predictions', linestyle='--')
ax.set_xlabel('Months')
ax.set_ylabel('Number of Monthly Passengers')
ax.set_title('Passenger Forecast for Next Year')
ax.legend()
plt.grid()
plt.show()
```
### OUTPUT:

![Screenshot 2025-05-06 091346](https://github.com/user-attachments/assets/5b3fb186-a461-4cc0-a5ff-b9ece321fac2)


![Screenshot 2025-05-06 091935](https://github.com/user-attachments/assets/42215432-187f-4f9c-9dde-7f17a5bc672d)




**Moving Average**


![Screenshot 2025-05-06 091645](https://github.com/user-attachments/assets/327de7bf-4e52-4f84-ae93-2a6f3971e9a9)



**Plot Transform Dataset**


![Screenshot 2025-05-06 091717](https://github.com/user-attachments/assets/41b28722-185c-4e21-a1a3-36e9b3b4f866)


**Exponential Smoothing**


![Screenshot 2025-05-06 093317](https://github.com/user-attachments/assets/03e58781-d19a-4d5d-ab19-d5c43b5a0f09)


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
