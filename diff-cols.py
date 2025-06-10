import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#a global List to store the results
results = []

"""The TrainModel() function takes the dataset and a string describing its features as input.
 It seperates the dataset into X,Y axis and further splits it into train and test.
 Next, it trains a KNN model on the given dataset and evallates it using various metrics and saves the result into
  the results list in the form of a dictionary. Lastly it plots the confusion matrix of the model in question."""
def TrainModel(dataset,comment):
    dataset=dataset.drop_duplicates()


    # Seperating the dataframe into X and Y axis
    X = dataset.drop("ANOMALY", axis=1)
    Y = dataset["ANOMALY"]



    #splitting the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

    #defining the KNN model
    knnModel = KNeighborsClassifier(n_neighbors=5)

    #Fitting the Model
    knnModel.fit(X_train, y_train)

    #Making predication based on the training data
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



    # Adding the performnace metrics to results in the form of a dictionary
    results.append({
        'Features':comment,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'True Negative': trueNegative,
        'True Positive': truePositive,
        'False Positive': falsePositive,
        'False Negative': falseNegative,
    })

    #plotting the confusion matrix
    cmDisplay.plot()
    #setting the title of the confusion Matrix
    plt.title(f'Confusion Matrix for {comment} Column')
    plt.show()






"""The main() function calls the TrainModel() function to train KNN Models with
different features."""
def main():
    # Loading the dataset as a pandas frame
    dataframe = pd.read_csv("network-logs.xls")

    # Printing 1st five rows using head()
    print("Head")
    print(dataframe.head())
    print("--------------------------------------------------------------")

    #Dislaying dataset with All features
    print("Data with All Features\n",dataframe.head())
    print("--------------------------------------------------------------")

    #calling the TrainModel()
    TrainModel(dataframe,"All")

    #Dropping one key feature using drop()

    modifiedDataframe=dataframe.drop(["LATENCY"],axis=1)
    print("data with one less key coloumn \n",modifiedDataframe.head())
    print("--------------------------------------------------------------")


    #Calling TrainModel()

    TrainModel(modifiedDataframe,"One Less")


    #Generating a feature with random values (0 - 100) using numpy
    randomColumn= np.random.randint(0, 100, dataframe.shape[0])

    #Adding the feature with random values to the dataframe using assign()
    #assign() adds a feature to a dataframe and returns the resulting dataframe
    dataframeRandomCol=dataframe.assign(Random_Values=randomColumn)

    print("Data with a feature with random values \n",dataframeRandomCol.head())
    print("--------------------------------------------------------------")

    #Calling TrainModel()
    TrainModel(dataframeRandomCol,"Random Values")


    #Converting Results to a dataframe
    resultsDataFrame = pd.DataFrame(results)

    # Printing the dataframe containing the results by converting into a string
    # The parameters set allow the dataframe to be fully printed
    print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None,
                                     float_format='{:,.4f}'.format, header=True))
    # adding the x values for the accuracy bars
    xValues = ["All Features","One Less","Random Feature"]
    yValues = []
    # getting the accuracies from the results list to act as y values
    for i in results:
        yValues.append(i.get("Accuracy"))
    # plotting the accurray bar
    plt.bar(xValues, yValues)
    #adding x and y labels
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    #setting the title of the plot
    plt.title("Accuracy Comparsion Between different Features")
    plt.show()

#Calling the main() function
if __name__ == "__main__":
    main()