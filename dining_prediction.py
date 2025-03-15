import glob
from pymongo import MongoClient
import pandas as pd
from together import Together
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import os
print("Current Working Directory:", os.getcwd())

l = [] #use a list and concat later, faster than append in the loop
for f in glob.glob("./*.xlsx"):
   df = pd.read_excel("dining_info.xlsx")
   cuisine_dish = pd.read_excel('cuisine_dish.xlsx')
   cuisine_dish = pd.read_excel('swetha/cuisine_dish.xlsx')

# Load dataset
for f in glob.glob("./*.xlsx"):
    df = pd.read_excel("dining_info.xlsx")
    cuisine_dish = pd.read_excel('cuisine_dish.xlsx')
    cuisine_dish = pd.read_excel('swetha/cuisine_dish.xlsx')
# Create a Together object

# Connect to MongoDB
client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["hotel_guests"]
collection = db["dining_info"]

# Initialize Together AI client (API key should be secured in environment variables)
client = Together (api_key='5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4')

# Load data from MongoDB
df_from_mongo = pd.DataFrame(list(collection.find()))
df = df_from_mongo.copy()

# Convert to datetime
df['check_in_date'] = pd.to_datetime(df['check_in_date'])
df['check_out_date'] = pd.to_datetime(df['check_out_date'])
df['order_time'] = pd.to_datetime(df['order_time'])

# Extract features
df['check_in_day'] = df['check_in_date'].dt.dayofweek
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days

# Split data into training and test sets
features_df = df[df['order_time'] < '2024-01-01']
train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-10-01')]
test_df = df[df['order_time'] > '2024-10-01']

# Customer-Level Aggregations
customer_features = features_df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),
    avg_spend_per_customer=('price_for_1', 'mean')
).reset_index()

# Merge Features
train_df = train_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(customer_features, on='customer_id', how='left')

# Drop unnecessary columns
train_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)
test_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)

# One-Hot Encoding for Categorical Features
categorical_cols = ['Preferred Cusine']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoded_array = encoder.fit_transform(train_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

encoded_test = encoder.transform(test_df[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

# Label Encoding for Target Variable
train_df = train_df.dropna(subset=['dish'])
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])

X_train = train_df.drop(columns=['dish'])
y_train = train_df['dish']

test_df = test_df.dropna(subset=['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])
X_test = test_df.drop(columns=['dish'])
y_test = test_df['dish']

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.05,  # Adjusted learning rate
    max_depth=4,  # Reduced depth to prevent overfitting
    n_estimators=50,  # Reduced estimators for lower complexity
    subsample=0.7,  # Adjusted subsample
    colsample_bytree=0.7,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
y_pred_prob = xgb_model.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_prob)

# Feature Importance
plt.figure(figsize=(10, 5))
xgb.plot_importance(xgb_model, max_num_features=5)
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")
