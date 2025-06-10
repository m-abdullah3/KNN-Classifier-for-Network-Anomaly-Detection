import pandas as pd

#Loading the dataset as a pandas frame
dataFrame= pd.read_csv("network-logs.xls")

#Printing 1st five rows using head()
print("Head: ")
print(dataFrame.head())
print("------------------------------------------------")

#Printing last five rows using tail()
print("Tail")
print(dataFrame.tail())
print("------------------------------------------------")

#Finding Missing Values
print("Missing Values")
print(dataFrame.isnull().sum())
print("------------------------------------------------")

#Finding Duplicate values
print("number of Duplicates = ",dataFrame.duplicated().sum())

#Removing duplicates
newDataframe=dataFrame.drop_duplicates()
print("Removing Duplictes..........")
print("number of Duplicates = ",newDataframe.duplicated().sum())

print("------------------------------------------------")

#getting statistics about the data using describe
print("Describe()")
print(dataFrame.describe())

print("------------------------------------------------")

#Getting information about the dataframe using info()
print("Info()")
print(newDataframe.info())

print("------------------------------------------------")

#Checking count of unique labels
print("Count of Labels")
labelCount=dataFrame["ANOMALY"].value_counts()
#printing the result
print(labelCount)

