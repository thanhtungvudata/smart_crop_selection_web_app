import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from .preprocess import preprocess_data

# Load dataset
df = pd.read_csv('data/soil_measures.csv')

# Preprocess data
X_scaled, y_encoded, scaler, label_encoder = preprocess_data(df, fit_scaler=True, fit_encoder=True)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Define parameter grids for XGBoost
xgb_params = {
    'max_depth': [10, 20],
    'learning_rate': [0.001, 0.1, 1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.9]
}

# Perform hyperparameter tuning using cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost tuning
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', verbosity=1)
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=kf, scoring='accuracy', n_jobs=1, verbose=1)
xgb_grid.fit(X_train, y_train)

# Best model from GridSearchCV
best_xgb_model = xgb_grid.best_estimator_

# Evaluate model
y_pred = best_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best XGBoost Model Accuracy: {accuracy:.4f}')


# Save the best model
joblib.dump(best_xgb_model, 'models/crop_model.pkl')
