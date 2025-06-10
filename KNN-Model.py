import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#Loading the dataset as a pandas frame
dataframe=pd.read_csv("network-logs.xls")

#Printing 1st five rows using head()
print(dataframe.head())

#Finding Duplicate values
print("number of duplictes = ", dataframe.duplicated().sum())
print("removing duplicates..........")
#Removing duplicates
dataframe=dataframe.drop_duplicates()
print("number of duplicates = ",dataframe.duplicated().sum())





#Seperating the dataframe into X and Y axis
X=dataframe.drop("ANOMALY",axis=1)
Y=dataframe["ANOMALY"]

#Splitting the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.30)

#Defining the KNN Classifier
knnModel=KNeighborsClassifier(n_neighbors=5)

#Fitting the KNN model
knnModel.fit(X_train,y_train)

#Making predictions based on the training data
y_pred=knnModel.predict(X_test)

#Calculating the Performnace Metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

#zero_divsion = 0 to handle divisions with zero as the denominator
precision=metrics.precision_score(y_test,y_pred, average='weighted',zero_division=0)
recall=metrics.recall_score(y_test,y_pred, average='weighted',zero_division=0)
confusionMatrix=metrics.confusion_matrix(y_test,y_pred)

#Converting the Confusion Matrix into a Graph
cmDisplay= metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=[0,1])

#Calculting the True Positives etc. using the Confusion Matrix
trueNegative, falsePositive, falseNegative, truePositive = confusionMatrix.ravel()

#printing the calculated metrics
print("-----------------------------------------------")
print("accuracy = ",accuracy)
print("precision = ",precision)
print("recall = ",recall)

print("")

print(f'True Positives (TP): {truePositive}')
print(f'False Positives (FP): {falsePositive}')
print(f'True Negatives (TN): {trueNegative}')
print(f'False Negatives (FN): {falseNegative}')
print("-----------------------------------------------")

#plotting the confusion Matrix
cmDisplay.plot()

#setting the title
plt.title("Confusion Matrix for KNN")
plt.show()



