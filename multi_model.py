import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error

class MultiTaskRailwayModel:
    def __init__(self):
        self.label_encoders = {}  # Stores LabelEncoders for categorical columns
        self.models = {}          # Stores trained models
        self.targets = {}         # Stores target names and types

    # -----------------------------
    # Register targets
    def add_target(self, target_name, model_type='classification'):
        """
        Register a new target for training.
        model_type: 'classification' or 'regression'
        """
        self.targets[target_name] = model_type

    # -----------------------------
    # Train all registered targets
    def fit(self, df):
        """
        Train all targets on the given dataframe.
        """
        # Encode categorical columns
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le

        # Train each model
        for target, model_type in self.targets.items():
            X = df_encoded.drop(target, axis=1)
            y = df_encoded[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Print metrics
            print(f"--- Model for target: {target} ---")
            if model_type == 'classification':
                print(classification_report(y_test, y_pred))
            else:
                print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

            # Store model
            self.models[target] = model

    # -----------------------------
    # Predict all targets
    def predict(self, df):
        """
        Predict all registered targets for new data.
        Returns a dictionary: {target_name: predictions}
        """
        df_encoded = df.copy()
        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = le.transform(df_encoded[col])

        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(df_encoded)
        return predictions

    # -----------------------------
    # Save models and encoders
    def save(self, path='models/'):
        """
        Save all models and encoders for deployment.
        """
        import os
        os.makedirs(path, exist_ok=True)
        # Save models
        for target, model in self.models.items():
            joblib.dump(model, f"{path}{target}_model.pkl")
        # Save encoders
        joblib.dump(self.label_encoders, f"{path}label_encoders.pkl")
        # Save targets info
        joblib.dump(self.targets, f"{path}targets_info.pkl")
        print(f"Models and encoders saved in '{path}'")

    # -----------------------------
    # Load models and encoders
    def load(self, path='models/'):
        """
        Load models and encoders from disk for deployment.
        """
        self.label_encoders = joblib.load(f"{path}label_encoders.pkl")
        self.targets = joblib.load(f"{path}targets_info.pkl")
        self.models = {target: joblib.load(f"{path}{target}_model.pkl") for target in self.targets}
        print(f"Models and encoders loaded from '{path}'")
