from EDA import eda
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

def plot_confusion(cm,algo):       
  plt.figure(figsize = (10,10))
  sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
  plt.xlabel("Prediction")
  plt.ylabel("True value")
  plt.title(" Confusion Matrix for " + algo + " algorithm")
  plt.show()
  return sn


X_train, X_test, y_train, y_test = eda("activity_context_tracking_data.csv")

print("---------------------------------------Started Training models---------------------------------------\n ---------------------------------------Be Patient---------------------------------------")
#Random Forest Classifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("---------------------------------------RF Model Trained---------------------------------------")

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
print("Accuracy of Random Forest Classifier Algorith :", rf_accuracy)
print("Precision of Random Forest Classifier Algorith :", rf_precision)
print("Recall of Random Forest Classifier Classifier Algorith :", rf_recall)
print("F1 Score of Random Forest Classifier Algorith :", rf_f1)
print("Printing Confusion Matrix of Random Forest Classifier")
plot_confusion(confusion_matrix(y_test,rf_predictions),"Random Forest Classifier")

#DT Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
print("---------------------------------------DT Model Trained---------------------------------------")
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, average='weighted')
dt_recall = recall_score(y_test, dt_predictions, average='weighted')
dt_f1 = f1_score(y_test, dt_predictions, average='weighted')
print("Accuracy of DecesionTree Classifier Algorith :", dt_accuracy)
print("Precision of DecesionTree Classifier Algorith :", dt_precision)
print("Recall of DecesionTree Classifier Algorith :", dt_recall)
print("F1 Score of DecesionTree Classifier Algorith :", dt_f1)
print("Printing Confusion Matrix of Decision Tree Classifier")
plot_confusion(confusion_matrix(y_test,dt_predictions),"Decision Tree Classifier")

# KNN Model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
print("---------------------------------------KNN Model Trained---------------------------------------")
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')
print("Accuracy of K-Nearest Neighbors Classifier Algorith :", knn_accuracy)
print("Precision of K-Nearest Neighbors Classifier Algorith :", knn_precision)
print("Recall of K-Nearest NeighborsClassifier Classifier Algorith :", knn_recall)
print("F1 Score of K-Nearest Neighbors Classifier Algorith :", knn_f1)
print("Printing Confusion Matrix of K-Nearest Neighbors Classifier")
plot_confusion(confusion_matrix(y_test,knn_predictions),"K-Nearest Neighbors")

print("---------------------------------------Ended training models---------------------------------------")
print("Pickling trained models for future predictions")
filename_rf_model = "rfm_model.pkl"
filename_dt_model = "dt_model.pkl"
filename_knn_model = "knn_model.pkl"
joblib.dump(rf_model,filename_rf_model)
joblib.dump(dt_model,filename_dt_model)
joblib.dump(knn_model,filename_knn_model)
print("---------------------------------------Pickling training models completed---------------------------------------")
