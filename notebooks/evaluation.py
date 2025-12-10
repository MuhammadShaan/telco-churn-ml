from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

    return y_prob
