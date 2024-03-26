import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('breast_cancer_survival.csv')

features = data.drop('Patient_Status', axis=1)
target = data['Patient_Status']



print("Target Variable:", target.name)



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)


k_values = range(2, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracies.append(accuracy)

plt.plot(k_values, accuracies)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K in KNN')
plt.show()
