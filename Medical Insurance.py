import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load Dataset
df = pd.read_csv(r"D:\Medical Insurance\insurance.csv")

# Data Preprocessing
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_features = ['sex', 'smoker', 'region']
encoded_data = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
df = df.drop(columns=categorical_features).join(encoded_df)

# Splitting Data
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr))}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr)}")

# Train Neural Network Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# Evaluate Neural Network
y_pred_nn = nn_model.predict(X_test).flatten()
print("\nNeural Network Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_nn))}")
print(f"R2 Score: {r2_score(y_test, y_pred_nn)}")

# Data Visualization
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Scatter Plot for Age vs Charges
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['age'], y=df['charges'], hue=df['smoker_yes'])
plt.xlabel('Age')
plt.ylabel('Medical Charges')
plt.title('Age vs Medical Charges')
plt.show()

# Save Models
joblib.dump(lr_model, r"D:\Medical Insurance\linear_regression_model.pkl")
nn_model.save(r"D:\Medical Insurance\neural_network_model.h5")
