import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#Loading the dataset as a pandas frame
dataframe=pd.read_csv("network-logs.xls")

#Printing 1st five rows using head()
print("Head")
print(dataframe.head())
print("--------------------------------------------------------------")

#Finding Duplicate values
print("number of duplictes = ", dataframe.duplicated().sum())
print("removing duplicates..........")

#Removing duplicates
dataframe=dataframe.drop_duplicates()
print("number of duplicates = ",dataframe.duplicated().sum())
print("--------------------------------------------------------------")

#Seperating the dataframe into X and Y axis
X=dataframe.drop("ANOMALY",axis=1)
Y=dataframe["ANOMALY"]

#The list holding the train sizes
trainSize=[0.40,0.60,0.80]
#list to hold the results
results=[]

#This for loop will iterate through the list trainSize
for i in trainSize:
    #Splitting the data into train and test
    X_train,X_test,y_train,y_test=train_test_split(X,Y, train_size=i)

    #defining the KNN Model
    knnModel = KNeighborsClassifier(n_neighbors=5)

    #Fitting the Model
    knnModel.fit(X_train, y_train)
    #Making prediction based on the training data
    y_pred = knnModel.predict(X_test)

    #Calculating the performance metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # zero_divsion = 0 to handle divisions with zero as the denominator
    precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)

    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)

    # Converting the Confusion Matrix into a Graph
    cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1])

    # Calculting the True Positives etc. using the Confusion Matrix
    trueNegative, falsePositive, falseNegative, truePositive = confusionMatrix.ravel()

    #Adding the performnace metrics to results in the form of a dictionary
    results.append({
        'Train Size': i*100,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'True Negative': trueNegative,
        'True Positive':truePositive,
        'False Positive': falsePositive,
        'False Negative': falseNegative,
    })

    # plotting the confusion Matrix
    cmDisplay.plot()
    #Setting the title
    plt.title(f'Confusion Matrix for Train Size : {i}')
    plt.show()

#Converting the results into a pandas frame
resultsDataFrame=pd.DataFrame(results)

#Printing the dataframe containing the results by converting into a string
#The parameters set allow the dataframe to be fully printed
print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format, header=True))

# adding the x values for the accuracy bar
xValues=[40,60,80]
yValues=[]
#getting the accuracies from the results list to act as y values
for i in results:
    yValues.append(i.get("Accuracy"))

#plotting the accurray bar
plt.bar(xValues,yValues)
#setting the x and y labels
plt.xlabel("Train Size")
plt.ylabel("Accuracy")
#setting the title
plt.title("Accuracy Comparsion Between different Train sizes")
plt.show()

