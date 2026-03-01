import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data with the CORRECT filename
df = pd.read_csv('credit_risk_dataset (4).csv')
df = df.dropna(subset=['loan_status'])
df.drop_duplicates(inplace=True)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Features
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

# Build Models
lr = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
rf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))])

# Train Models
print("Training models locally... this will take a few seconds.")
lr.fit(X, y)
rf.fit(X, y)

# Save Models
joblib.dump(lr, 'logistic_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
print("✅ Success! Mac-compatible models generated.")