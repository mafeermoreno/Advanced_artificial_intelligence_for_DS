import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read original csv file
df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')
df = df[df['Rented Bike Count'] < 2500]
print(df.head())
print(df.info())

# Normalize data
columns_to_normalize = [
    'Rented Bike Count', 'Hour', 'Temperature(C)', 
    'Visibility (10m)', 'Wind speed (m/s)', 
    'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 
    'Rainfall(mm)'
]

for column in columns_to_normalize:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df = pd.get_dummies(df, columns=['Seasons'], prefix ='', prefix_sep ='')
df['Spring'] = df['Spring'].astype(int)
df['Summer'] = df['Summer'].astype(int)
df['Autumn'] = df['Autumn'].astype(int)
df['Winter'] = df['Winter'].astype(int)
df.columns = df.columns.str.strip()

df = df.drop(['Date', 'Holiday', 'Humidity(%)', 'Snowfall (cm)', 'Functioning Day'], axis=1)
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
df.to_csv('SeoulBikeData_clean.csv', index=False)

# Divide the dataset into training and test sets
df = df.sample(frac=1).reset_index(drop=True)
df_train = df[:int(0.8*len(df))]
test = df[int(0.8*len(df)):]

# Select features to use as input
features = ['Hour', 'Temperature(C)', 'Visibility (10m)', 'Wind speed (m/s)', 'Dew point temperature(C)','Spring', 'Summer', 'Autumn', 'Winter','Solar Radiation (MJ/m2)', 'Rainfall(mm)']
x_train = df_train[features].values
# Select the target variable (the value we want to predict)
y_train = df_train['Rented Bike Count'].values

# Select features and target variable for the test set
x_test = test[features].values
y_test = test['Rented Bike Count'].values

# Add bias to the input features
train_with_bias = np.c_[np.ones(x_train.shape[0]), x_train]
test_with_bias = np.c_[np.ones(x_test.shape[0]), x_test]

# Initialize the parameters randomly
params = np.random.randn(train_with_bias.shape[1]) 
alfa = 0.1 #Learning rate
epochs = 0
_errors_ = []

# Hypothesis function (linear regression)
def h(params, sample):
    return np.dot(params, sample)

# Mean squared error function
def mse(params, samples, y):
    global _errors_
    predictions = np.dot(samples, params)
    errors = predictions - y
    total_error = np.sum(errors ** 2)
    mean_error_param = total_error / (2 * len(samples))
    _errors_.append(mean_error_param)
    return mean_error_param

# Gradient descent function
def gd(params, samples, y, alfa):
    predictions = np.dot(samples, params)
    errors = predictions - y
    gradients = np.dot(samples.T, errors) / len(samples)
    params -= alfa * gradients
    return params

# Training loop
while True:
    initial_params = params.copy()
    params = gd(params, train_with_bias, y_train, alfa)
    error = mse(params, train_with_bias, y_train)
    print(f"Epoch #{epochs}, Error: {error}")
    epochs += 1
    if np.allclose(initial_params, params) or epochs == 10000:
        print(params)
        break

# Calculate R^2 to evaluate the model performance 
def r_2(y_real, y_pred):
    total_variance = np.sum((y_real - np.mean(y_real)) ** 2)
    residual_variance = np.sum((y_real - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)

# Predicciones y cálculo de R^2
predictions = np.dot(test_with_bias, params)
predictions = np.maximum(predictions, 0) 
r2_test = r_2(y_test, predictions)
print(f"R^2 en el conjunto de prueba: {r2_test:.4f}")

# Plot the error vs epochs
plt.plot(_errors_)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Epochs')
plt.show()

# Plot real values vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='green', label='True Values', alpha=0.6)
plt.scatter(range(len(predictions)), predictions, color='blue', label='Predictions', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Rented Bike Count')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()