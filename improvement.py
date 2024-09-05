import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read original csv file
df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')
df = df[df['Rented Bike Count'] < 2500]

# Normalize data
columns_to_normalize = [
    'Rented Bike Count', 'Hour', 'Temperature(C)', 
    'Visibility (10m)', 
    'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 
    'Rainfall(mm)'
]

for column in columns_to_normalize:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df = pd.get_dummies(df, columns=['Seasons'], prefix ='', prefix_sep ='')
df.columns = df.columns.str.strip()

df = df.drop(['Date', 'Holiday', 'Humidity(%)', 'Visibility (10m)', 'Wind speed (m/s)', 'Snowfall (cm)', 'Functioning Day'], axis=1)

# Divide the dataset into training, validation, and test sets
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(0.7 * len(df))  # 70% for training
val_size = int(0.15 * len(df))   # 15% for validation
test_size = len(df) - train_size - val_size  # 15% for test

df_train = df[:train_size]
df_val = df[train_size:train_size+val_size]
df_test = df[train_size+val_size:]

# Select features for training, validation, and test sets
features = ['Hour', 'Temperature(C)', 'Dew point temperature(C)', 'Spring', 'Summer', 'Autumn', 'Winter', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)']

x_train = df_train[features].values.astype(np.float64)
y_train = df_train['Rented Bike Count'].values.astype(np.float64)

x_val = df_val[features].values.astype(np.float64)
y_val = df_val['Rented Bike Count'].values.astype(np.float64)

x_test = df_test[features].values.astype(np.float64)
y_test = df_test['Rented Bike Count'].values.astype(np.float64)

# Add bias to the input features
train_with_bias = np.c_[np.ones(x_train.shape[0]), x_train].astype(np.float64)
val_with_bias = np.c_[np.ones(x_val.shape[0]), x_val].astype(np.float64)
test_with_bias = np.c_[np.ones(x_test.shape[0]), x_test].astype(np.float64)

# Initialize the parameters randomly
params = np.random.randn(train_with_bias.shape[1]) 
alfa = 0.1  # Learning rate
epochs = 0
train_errors = []
val_errors = []
train_r2 = []
val_r2 = []

# Hypothesis function (linear regression)
def h(params, sample):
    return np.dot(params, sample)

# Mean squared error function
def mse(params, samples, y):
    predictions = np.dot(samples, params)
    errors = predictions - y
    total_error = np.sum(errors ** 2)
    mean_error_param = total_error / (2 * len(samples))
    return mean_error_param

# R² function to calculate accuracy
def r_2(y_real, y_pred):
    total_variance = np.sum((y_real - np.mean(y_real)) ** 2)
    residual_variance = np.sum((y_real - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)

# Gradient descent function
def gd(params, samples, y, alfa):
    predictions = np.dot(samples, params)
    errors = predictions - y
    gradients = np.dot(samples.T, errors) / len(samples)
    params -= alfa * gradients
    return params

# Training loop with validation
while True:
    initial_params = params.copy()
    params = gd(params, train_with_bias, y_train, alfa)
    
    # Calculate train and validation errors
    train_error = mse(params, train_with_bias, y_train)
    val_error = mse(params, val_with_bias, y_val)
    train_errors.append(train_error)
    val_errors.append(val_error)
    
    # Calculate train and validation R² (accuracy)
    train_r2_value = r_2(y_train, np.dot(train_with_bias, params))
    val_r2_value = r_2(y_val, np.dot(val_with_bias, params))
    train_r2.append(train_r2_value)
    val_r2.append(val_r2_value)
    
    print(f"Epoch #{epochs}, Train Error: {train_error}, Val Error: {val_error}, Train R²: {train_r2_value:.4f}, Val R²: {val_r2_value:.4f}")
    epochs += 1
    
    if np.allclose(initial_params, params) or epochs == 10000:
        print(params)
        break

# Plot training and validation R² (accuracy)
plt.plot(train_r2, label='Train R² (Accuracy)')
plt.plot(val_r2, label='Validation R² (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('R² (Accuracy)')
plt.title('Train vs Validation R² (Accuracy)')
plt.legend()
plt.show()

# Calculate R² for test set
predictions = np.dot(test_with_bias, params)
predictions = np.maximum(predictions, 0) 
r2_test = r_2(y_test, predictions)
print(f"R² en el conjunto de prueba: {r2_test:.4f}")

# Plot real values vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='#34a2eb', label='True Values', alpha=0.6)
plt.scatter(range(len(predictions)), predictions, color='#eb3471', label='Predictions', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Rented Bike Count')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()
