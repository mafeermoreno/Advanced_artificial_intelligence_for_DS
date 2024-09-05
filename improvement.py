import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Read original csv file
df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')
df = df[df['Rented Bike Count'] < 2500]

# Normalize data
columns_to_normalize = [
    'Rented Bike Count', 'Hour', 'Temperature(C)', 
    'Visibility (10m)', 'Dew point temperature(C)', 
    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Humidity(%)'
]

for column in columns_to_normalize:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# One-hot encoding for column 'Seasons'
df = pd.get_dummies(df, columns=['Seasons'], prefix='', prefix_sep='')
df.columns = df.columns.str.strip()

# Delete irrelevant columns
df = df.drop(['Date', 'Holiday', 'Visibility (10m)', 
              'Wind speed (m/s)', 'Snowfall (cm)', 'Functioning Day'], axis=1)

# Split the dataset into training, validation, and test sets
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(0.6 * len(df))  # 60% for training
val_size = int(0.2 * len(df))    # 20% for validation
test_size = len(df) - train_size - val_size  # 20% for testing

df_train = df[:train_size]
df_val = df[train_size:train_size+val_size]
df_test = df[train_size+val_size:]

# Select features
features = ['Hour', 'Temperature(C)', 'Dew point temperature(C)', 
            'Spring', 'Summer', 'Autumn', 'Winter', 
            'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Humidity(%)']

x_train = df_train[features].values.astype(np.float64)
y_train = df_train['Rented Bike Count'].values.astype(np.float64)

x_val = df_val[features].values.astype(np.float64)
y_val = df_val['Rented Bike Count'].values.astype(np.float64)

x_test = df_test[features].values.astype(np.float64)
y_test = df_test['Rented Bike Count'].values.astype(np.float64)

# Create Random Forest model
rf = RandomForestRegressor(
    n_estimators=123,          # Number of trees in the forest
    max_depth=7,               # Maximum depth of the tree
    min_samples_split=10,      # Minimum number of samples required to split an internal node
    min_samples_leaf=5,        # Minimum number of samples required to be at a leaf node
    random_state=42            # Random seed
)

# Train the model with the training data
rf.fit(x_train, y_train)

# Predictions on the training and validation sets
y_train_pred = rf.predict(x_train)
y_val_pred = rf.predict(x_val)

# Calculate MSE and R² on the training and validation sets
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
print(f"Validation MSE: {val_mse:.4f}, Validation R²: {val_r2:.4f}")

# Predictions on the test set
y_pred = rf.predict(x_test)

# Calculate R² on the test set
r2_test = r2_score(y_test, y_pred)
print(f"R² en el conjunto de prueba: {r2_test:.4f}")

# Calculate Mean Squared Error (MSE) on the test set
mse_test = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error en el conjunto de prueba: {mse_test:.4f}")

# Plot true values vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='#34a2eb', label='True Values', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='#eb3471', label='Predictions', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Rented Bike Count')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()

# Calculate the residuals
residuals = y_test - y_pred

# Plot the distribution of the residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='#34ebc9')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Show feature importances
importances = rf.feature_importances_
for feature, importance in zip(features, importances):
    print(f'{feature}: {importance:.4f}')
