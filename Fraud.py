
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class FraudDetector:
    def __init__(self):
        self.model = None
        self.feature_order = [f'V{i}' for i in range(1,29)] + ['Amount']

    def load_and_prepare_data(self, file_path='creditcard.csv'):
        print("Loading dataset...")
        data = pd.read_csv(file_path)
        print(f"Dataset shape: {data.shape}")

        # Features and target
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Log transform Amount to reduce skewness
        X['Amount'] = np.log1p(X['Amount'])

        return X, y

    def train(self, X, y):
        print("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)
        self.model = model

        print("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return model

    def predict_fraud(self, df):
        """
        Predict fraud for given transactions DataFrame.

        Parameters:
        - df: pandas DataFrame with columns matching features V1..V28, Amount

        Returns:
        - numpy array of predictions (0=legit, 1=fraud)
        """
        if self.model is None:
            raise Exception("Model not trained. Call train() first.")

        # Make sure all needed columns are present
        for feature in self.feature_order:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature column: {feature}")

        # Apply log1p transform to Amount column to be consistent
        df = df.copy()
        df['Amount'] = np.log1p(df['Amount'])

        X_input = df[self.feature_order].values
        preds = self.model.predict(X_input)
        return preds

def main():
    detector = FraudDetector()
    X, y = detector.load_and_prepare_data()
    detector.train(X, y)

    # Example usage: predict fraud for first 5 test samples
    sample_df = X.head(5)
    print("\nPredictions for first 5 samples:")
    preds = detector.predict_fraud(sample_df)
    print(preds)

if __name__ == "__main__":
    main()
