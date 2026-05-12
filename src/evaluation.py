# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=2)
y_true_classes = np.argmax(y_test, axis=2)
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
# Compute per-class accuracy
true_per_class = np.diag(conf_matrix)  # Correct predictions for each class
support = np.sum(conf_matrix, axis=1)  # Total true samples for each class
accuracy_per_class = true_per_class / support

