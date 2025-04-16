import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
# Add to imports
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize and train model with GridSearch
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get best model
model = grid_search.best_estimator_

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("\nBest Parameters:", grid_search.best_params_)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
pickle.dump(model, open('crop_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("\nModel and scaler have been saved successfully!")

# After model evaluation, add these sections:
# Calculate and display feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Crop Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Calculate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Save additional model metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance.to_dict()
}
pickle.dump(metrics, open('model_metrics.pkl', 'wb'))

# Add learning curves analysis
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('learning_curves.png')

# Add validation curves for n_estimators
param_range = np.linspace(10, 200, 10).astype(int)
train_scores, val_scores = validation_curve(
    model, X_train_scaled, y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=5, n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(param_range, np.mean(val_scores, axis=1), label='Cross-validation score')
plt.xlabel('Number of trees')
plt.ylabel('Score')
plt.title('Validation Curve')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('validation_curves.png')

# After validation curves, add these visualizations
# ROC Curve with mean ROC
plt.figure(figsize=(15, 10))
y_test_bin = label_binarize(y_test, classes=model.classes_)
y_score = model.predict_proba(X_test_scaled)

# Plot ROC curves for each class
mean_tpr = np.zeros_like(np.linspace(0, 1, 100))
mean_fpr = np.linspace(0, 1, 100)

for i in range(len(model.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, alpha=0.8, label=f'{model.classes_[i]} (AUC = {roc_auc:.2f})')
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_tpr += interp_tpr

# Plot mean ROC
mean_tpr /= len(model.classes_)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (AUC = {mean_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Crop Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')

# Precision-Recall Curve with average precision
plt.figure(figsize=(15, 10))
mean_precision = np.zeros_like(np.linspace(0, 1, 100))
mean_recall = np.linspace(0, 1, 100)

for i in range(len(model.classes_)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, alpha=0.8, label=f'{model.classes_[i]} (AP = {avg_precision:.2f})')
    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
    mean_precision += interp_precision

mean_precision /= len(model.classes_)
plt.plot(mean_recall, mean_precision, 'k--', label=f'Mean PR', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Each Crop Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')

# Add after detailed metrics saving
# Cross-validation scores
# Replace cross-validation section with stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
print(f"\nStratified Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.3f}, {cv_scores.mean() + 1.96*cv_scores.std():.3f}])")

# Enhanced calibration curves with confidence intervals
plt.figure(figsize=(15, 10))
n_bins = 10

for i in range(len(model.classes_)):
    # Calculate initial calibration curve with fixed bin size
    prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_score[:, i], n_bins=n_bins, strategy='uniform')
    
    # Initialize list for collecting bootstrap results
    bootstrapped_probs = []
    n_bootstraps = 1000
    
    # Perform bootstrap sampling
    for j in range(n_bootstraps):
        indices = np.random.randint(0, len(y_test_bin), len(y_test_bin))
        boot_true = y_test_bin[indices, i]
        boot_pred = y_score[indices, i]
        boot_true_prob, _ = calibration_curve(boot_true, boot_pred, n_bins=n_bins, strategy='uniform')
        bootstrapped_probs.append(boot_true_prob)
    
    # Convert to numpy array and calculate confidence intervals
    bootstrapped_probs = np.array(bootstrapped_probs)
    ci_lower = np.percentile(bootstrapped_probs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrapped_probs, 97.5, axis=0)
    
    # Plot calibration curve with confidence intervals
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model.classes_[i]}', linewidth=2)
    plt.fill_between(prob_pred, ci_lower, ci_upper, alpha=0.2)

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('True probability')
plt.title('Calibration Curves with 95% Confidence Intervals')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_curves.png', dpi=300, bbox_inches='tight')

# Update detailed metrics
detailed_metrics.update({
    'cross_validation': {
        'scores': cv_scores.tolist(),
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'ci_lower': cv_scores.mean() - 1.96*cv_scores.std(),
        'ci_upper': cv_scores.mean() + 1.96*cv_scores.std()
    }
})

# Save updated metrics
pickle.dump(detailed_metrics, open('detailed_metrics.pkl', 'wb'))

print("\nAll visualizations, metrics, and cross-validation results have been saved successfully!")