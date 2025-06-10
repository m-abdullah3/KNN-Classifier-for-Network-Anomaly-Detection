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

#Adding the k values to a list
numberOfNeighbors=[1,5,15]

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.70)

# a list to store the performance metrics
results=[]

#This loop will iterate over the list numberOfNeighbors
for i in numberOfNeighbors:
    #Defining the KNN model
    knnModel = KNeighborsClassifier(n_neighbors=i)
    #Fitting the Model
    knnModel.fit(X_train, y_train)
    #Making preditction based on the training data
    y_pred = knnModel.predict(X_test)

    #Calculaing the performance metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # zero_divsion = 0 to handle divisions with zero as the denominator
    precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)

    # Converting the Confusion Matrix into a Graph
    cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1])

    # Calculting the True Positives etc. using the Confusion Matrix
    trueNegative, falsePositive, falseNegative, truePositive = confusionMatrix.ravel()

    # Adding the performnace metrics to the results list in the form of a dictionary
    results.append({
        'No. of K': i,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'True Negative': trueNegative,
        'True Positive': truePositive,
        'False Positive': falsePositive,
        'False Negative': falseNegative,
    })

    #Plotting the confsion matrix
    cmDisplay.plot()

    #setting the title
    plt.title(f'Confusion Matrix for K = {i}')
    plt.show()

#converting the results list into a dataframe
resultsDataFrame = pd.DataFrame(results)

#Printing the dataframe containing the results by converting into a string
#The parameters set allow the dataframe to be fully printed
print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format, header=True))

#setting the x values for the accuracy bar
xValues=[1,5,15]

yValues=[]

#getting the accuracies from the results list to act as y values
for i in results:
    yValues.append(i.get("Accuracy"))


#PLotting the accuracy bar
plt.bar(xValues,yValues)

#setting up x and y labels
plt.xlabel("Number of Ks")
plt.ylabel("Accuracy")

#setting up the title
plt.title("Accuracy Comparsion Between different Ks")
plt.show()