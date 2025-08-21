import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('static/creditcard_2023.csv', nrows=10000)

X = data.drop(columns=['Class'], axis=1, errors='ignore')
y = data['Class']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(pd.Series(y_train).value_counts(normalize=True))

rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    min_samples_split=5,
    max_depth=10
)

cv_score = cross_val_score(rf_model, X_train_scaled, y_train, cv=3, scoring='f1')
print("Cross-validation F1 scores:", cv_score)
print('Average F1 score:', np.mean(cv_score))

rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

importance = rf_model.feature_importances_
feature_imp = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importance
}).sort_values('Importance', ascending=False) 

print(feature_imp.head())


# Bar Graph
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_imp, x='Importance', y='Feature')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

#ROC AUC
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred_proba)
roc_auc = sklearn.metrics.auc(fpr, tpr)

#Visualisation of ROC AUC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()