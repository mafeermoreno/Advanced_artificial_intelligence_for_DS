import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define variable to store errors
__errors__= [];

df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')

print(df.head())
df.info()
print()
# Check features' stadistics
print(df.describe())
print()

# Verify duplicated rows
duplicated_rows = df.duplicated()
print(f"Number of duplicated rows: {duplicated_rows.sum()}")
print()

# Verify missing values
missing_values = df.isnull()
print(f"Missing values:\n{missing_values.sum()}")
print()

# Count the number of 0's in the 'Solar radiation' column
zero_solar = df['Solar Radiation (MJ/m2)'] == 0
print(f"Number of 0's in the 'Solar radiation' column: {zero_solar.sum()}/{len(df)}")
# Count the number of 0's in the 'Rainfall' column
zero_rainfall = df['Rainfall(mm)'] == 0
print(f"Number of 0's in the 'Rainfall' column: {zero_rainfall.sum()}/{len(df)}")
# Count the number of 0's in the 'Snowfall' column
zero_snowfall = df['Snowfall (cm)'] == 0
print(f"Number of 0's in the 'Snowfall' column: {zero_snowfall.sum()}/{len(df)}")
print() 

# Create a copy of the dataset in csv format
df.to_csv('SeoulBikeData_clean.csv', index = False)

# Read the new dataset
df_clean = pd.read_csv('SeoulBikeData_clean.csv', encoding ='ISO-8859-1')

# Delete from the copy the features 'Solar radiation(MJ/m2)', 'Rainfall(mm)' and 'Snowfall (cm)' (Fully missing II)
df_clean.drop(['Solar Radiation (MJ/m2)','Rainfall(mm)', 'Snowfall (cm)'], axis=1, inplace = True)

# Apply one-hot encoding to the 'Seasons' and 'Holiday' features
df_encoded = pd.get_dummies(df_clean, columns=['Seasons', 'Holiday'], prefix ='', prefix_sep ='')
df_encoded['Spring'] = df_encoded['Spring'].astype(int)
df_encoded['Summer'] = df_encoded['Summer'].astype(int)
df_encoded['Autumn'] = df_encoded['Autumn'].astype(int)
df_encoded['Winter'] = df_encoded['Winter'].astype(int)
df_encoded['Holiday'] = df_encoded['Holiday'].astype(int)
df_encoded['No Holiday'] = df_encoded['No Holiday'].astype(int)
df_encoded.columns = df_encoded.columns.str.strip()

# Scalarize the dataset
scaled_features = ['Rented Bike Count', 'Hour', 'Temperature(Â°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(Â°C)']
df_encoded[scaled_features] = (df_encoded[scaled_features] - df_encoded[scaled_features].mean()) / df_encoded[scaled_features].std()

# Hyphotesis function
def h(params, sample, bias):
    acum = bias
    for i in range(len(params)):
        acum += params[i] * sample[i]
    
    # Apply the sigmoid function to the linear combination
    return 1 / (1 + np.exp(-acum))

# Percentage of error
def error_perc (params, sample, y, bias):
    N = len(y) 
    total_error = 0
    
    for i in range(N):
        # Predict the probability using the hypothesis function
        prediction = h(params, sample[i], bias)
        # Calculate the cross-entropy for the sample
        total_error += -y[i] * np.log(prediction) - (1 - y[i]) * np.log(1 - prediction)
    
    # Average error
    return total_error / N

# Gradient descent function
def desc_gradient(params, samples, y, bias, alfa):
    # Calculate the hypothesis to be compared with the real value
    hypothesis = h(params, samples, bias)
    # Calculate the error (difference between the hypothesis and the real value)
    error = hypothesis - y
    N = len(y)
    
    # Calculate the gradient
    grad = np.dot(samples.T, error) / N
    params = params - alfa * grad
    bias = bias - alfa * np.sum(error) / N
    
    return params, bias