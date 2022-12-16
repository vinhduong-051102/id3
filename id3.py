import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv('dataset.csv')
    # Printing the dataswet shape 
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset: \n", balance_data)
    return balance_data


# Function to split the dataset
def splitdataset(data_set):
    print("Dataset after encode: \n", data_set)
    # Separating the target variable
    X = data_set.values[:, : 4]
    Y = data_set.values[:, 4 : ]
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.001)
    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with entropy.
def training_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy")
    # Performing training 
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions 
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)


# Driver code
def main():
    # Building Phase
    data = importdata()
    data["hometown_n"] = LabelEncoder().fit_transform(data["hometown"])
    data["habit_n"] = LabelEncoder().fit_transform(data["habit"])
    data["age_n"] = LabelEncoder().fit_transform(data["age"])
    data["gender_n"] = LabelEncoder().fit_transform(data["gender"])
    data["result_n"] = LabelEncoder().fit_transform(data["result"]);
    new_input = data.drop(["hometown", "habit", "gender", "result", "id", "age"], axis="columns")
    X, Y, X_train, X_test, y_train, y_test = splitdataset(new_input)
    clf_entropy = training_using_entropy(X_train, X_test, y_train)
    print("Results Using Entropy:")
    # Prediction using entropy 
    prediction(X_test, clf_entropy)
    print("Right result:\n", np.reshape(y_test, len(y_test)))
# Calling main function 
if __name__ == "__main__":
    main() 