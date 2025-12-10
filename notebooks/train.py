import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from preprocess import build_preprocessor

def train_model(df, target="Churn"):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("gb", GradientBoostingClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/model.pkl")

    return model, X_test, y_test
