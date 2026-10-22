import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Dataset Load Karein
df = pd.read_csv('Starbucks_Dataset.csv')

# 2. Target Calories Select Karein
target_column = 'Calories'
X = df.drop(columns=['Product_ID', 'Beverage_Name', target_column])
y = df[target_column]

# 3. Features Preprocessing & One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)
model_columns = X_encoded.columns.tolist()

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)

# 5. Decision Tree Regressor Train Karein
regressor = DecisionTreeRegressor(random_state=42, max_depth=4)
regressor.fit(X_train, y_train)

# 6. Model ko Save Karein
joblib.dump(regressor, 'decision_tree_calories_model.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

print("✅ Starbucks Calories Prediction Model successfully train aur save ho gaya hai!")
