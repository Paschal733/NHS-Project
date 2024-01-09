#!/usr/bin/env python
# coding: utf-8

# # Random Forest Model Implementation

# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the updated data
updated_data = pd.read_csv('C:/Users/ELITEBOOK 840 G6/Desktop/main_dataset_dissertation.csv')


# In[12]:


# Extract month and year from 'Period' column
updated_data['Year'] = updated_data['Period'].str.extract(r'(\d{4})').astype(float)
updated_data['Month'] = updated_data['Period'].str.extract(r'-([A-Za-z]+)-')

# Handling NaN values in 'Year' and 'Month' columns
updated_data = updated_data.dropna(subset=['Year', 'Month'])

# Now safely convert 'Year' to int
updated_data['Year'] = updated_data['Year'].astype(int)

# Map month names to numbers
months_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
updated_data['Month'] = updated_data['Month'].map(months_mapping)

# Preprocessing: One-Hot Encoding the 'Org Code'
encoder = OneHotEncoder(sparse=False)
org_code_encoded = encoder.fit_transform(updated_data[['Org Code']])
org_code_encoded_df = pd.DataFrame(org_code_encoded, columns=encoder.get_feature_names(['Org Code']))

# Merging the encoded Org Code back with the original data
data_encoded = pd.concat([updated_data, org_code_encoded_df], axis=1)
data_encoded.drop(['Org Code', 'Period'], axis=1, inplace=True)


# In[ ]:


# Define the target column
target = 'Number of adult critical care beds occupied'

# Define features and target for the encoded data
X_encoded = data_encoded.drop(columns=[target])
y_encoded = data_encoded[target]

# Handle NaN values in features and target
X_encoded = X_encoded.dropna()
y_encoded = y_encoded[X_encoded.index]  # Aligning target with features after dropping NaNs


# In[13]:



# Training the Random Forest Model
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)
rf_model_encoded = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_encoded.fit(X_train_encoded, y_train_encoded)


# In[18]:


# Random Forest Performance evaluation
y_pred = rf_model_encoded.predict(X_test_encoded)
mae = mean_absolute_error(y_test_encoded, y_pred)
mse = mean_squared_error(y_test_encoded, y_pred)
r2 = r2_score(y_test_encoded, y_pred)

# Display model performance
print("Random Forest Model Performance:")
print(f"RF Mean Absolute Error: {mae}")
print(f"RF Mean Squared Error: {mse}")
print(f"RF Root Mean Squared Error: {mse**0.5}")
print(f"RF R-squared: {r2}")


# In[23]:


import pandas as pd

# Assuming 'data_encoded', 'rf_model_encoded', and 'X_encoded' are already defined

# Initialize an empty DataFrame for predictions
predictions_df = pd.DataFrame(columns=['Org Code', 'Month', 'Predicted Adult Critical Care Beds'])

# Iterate over all combinations of 'Org Code' and 'Month'
for org_code in updated_data['Org Code'].unique():
    for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        month_num = months_mapping[month]
        org_code_column = 'Org Code_' + org_code

        if org_code_column in data_encoded.columns:
            trust_data = data_encoded[(data_encoded[org_code_column] == 1) & (data_encoded['Month'] == month_num)]

            if not trust_data.empty:
                # Predict using the model
                prediction = rf_model_encoded.predict(trust_data[X_encoded.columns])[0]
            else:
                prediction = None  # or use a default value like 0 or np.nan

            # Append the prediction to the DataFrame
            predictions_df = predictions_df.append({'Org Code': org_code, 'Month': month, 'Predicted Adult Critical Care Beds': prediction}, ignore_index=True)

# Display the DataFrame with all predictions
print(predictions_df.head(30))


# In[15]:


# Creating an interface for implementation
from ipywidgets import interact, Dropdown

# Define a function for making predictions
def make_predictions(org_code, month):
    month_num = months_mapping[month]
    org_code_column = 'Org Code_' + org_code

    if org_code_column in data_encoded.columns:
        trust_data = data_encoded[(data_encoded[org_code_column] == 1) & (data_encoded['Month'] == month_num)]
    else:
        return f"No data available for Org Code: {org_code}"

    if trust_data.empty:
        return "No data available for the selected trust and month."

    # Predict using the model
    prediction = rf_model_encoded.predict(trust_data[X_encoded.columns])[0]
    return f"Predicted number of adult critical care beds occupied for 2024: {prediction}"

# Create an interactive interface
org_codes = updated_data['Org Code'].unique()
months = Dropdown(options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
interact(make_predictions, org_code=org_codes, month=months)


# In[17]:


import matplotlib.pyplot as plt

# Generate predictions on the test set
y_pred_encoded = rf_model_encoded.predict(X_test_encoded)

# Plotting actual vs predicted values
plt.figure(figsize=(9, 5))
plt.scatter(y_test_encoded, y_pred_encoded, alpha=0.5)
plt.plot([y_test_encoded.min(), y_test_encoded.max()], [y_test_encoded.min(), y_test_encoded.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming predictions_df is your DataFrame containing predictions for each 'Org Code' and 'Month'

# First, convert 'Month' to a categorical type for proper ordering in the plot
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
predictions_df['Month'] = pd.Categorical(predictions_df['Month'], categories=months_order, ordered=True)

# Aggregate the predictions by month to get the average predicted beds for each month
average_beds_per_month = predictions_df.groupby('Month')['Predicted Adult Critical Care Beds'].mean()

# Creating the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=average_beds_per_month.index, y=average_beds_per_month.values)

plt.title('Average Predicted Adult Critical Care Beds per Month')
plt.ylabel('Average Predicted Number of Beds')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming predictions_df is your DataFrame containing predictions for each 'Org Code' and 'Month'

# Calculate the average predicted beds for each 'Org Code'
average_beds_per_org = predictions_df.groupby('Org Code')['Predicted Adult Critical Care Beds'].mean()

# Sort the values for better visualization
average_beds_per_org = average_beds_per_org.sort_values()

# Creating the bar plot
plt.figure(figsize=(50, 40))  # Adjust the figure size as needed
sns.barplot(x=average_beds_per_org.values, y=average_beds_per_org.index)

plt.title('Average Predicted Adult Critical Care Bed Demand for Each NHS Trust (Org Code)')
plt.xlabel('Average Predicted Number of Beds')
plt.ylabel('NHS Trust Org Code')
plt.grid(True)
plt.show()


# In[ ]:





# # XGBoost Model Implementation

# In[35]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[36]:


# Load the updated data
updated_data = pd.read_csv('C:/Users/ELITEBOOK 840 G6/Desktop/main_dataset_dissertation.csv')

# Extract month and year from 'Period' column
updated_data['Year'] = updated_data['Period'].str.extract(r'(\d{4})').astype(float)
updated_data['Month'] = updated_data['Period'].str.extract(r'-([A-Za-z]+)-')

# Handling NaN values in 'Year' and 'Month' columns
updated_data = updated_data.dropna(subset=['Year', 'Month'])

# Now safely convert 'Year' to int
updated_data['Year'] = updated_data['Year'].astype(int)

# Map month names to numbers
months_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
updated_data['Month'] = updated_data['Month'].map(months_mapping)

# Preprocessing: One-Hot Encoding the 'Org Code'
encoder = OneHotEncoder(sparse=False)
org_code_encoded = encoder.fit_transform(updated_data[['Org Code']])
org_code_encoded_df = pd.DataFrame(org_code_encoded, columns=encoder.get_feature_names(['Org Code']))

# Merging the encoded Org Code back with the original data
data_encoded = pd.concat([updated_data, org_code_encoded_df], axis=1)
data_encoded.drop(['Org Code', 'Period'], axis=1, inplace=True)

# Define the target column
target = 'Number of adult critical care beds occupied'

# Define features and target for the encoded data
X_encoded = data_encoded.drop(columns=[target])
y_encoded = data_encoded[target]

# Handle NaN values in features and target
X_encoded = X_encoded.dropna()
y_encoded = y_encoded[X_encoded.index]  # Aligning target with features after dropping NaNs
# Splitting the data into training and testing sets
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)







# Training the XGBoost Model
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)
xgb_model.fit(X_train_encoded, y_train_encoded)


# In[37]:



# Performance evaluation
y_pred = xgb_model.predict(X_test_encoded)
mae = mean_absolute_error(y_test_encoded, y_pred)
mse = mean_squared_error(y_test_encoded, y_pred)
r2 = r2_score(y_test_encoded, y_pred)

# Display model performance
print("XGBoost Model Performance:")
print(f"XGB Mean Absolute Error: {mae}")
print(f"XGB Mean Squared Error: {mse}")
print(f"XGB Root Mean Squared Error: {mse**0.5}")
print(f"XGB R-squared: {r2}")


# In[ ]:




