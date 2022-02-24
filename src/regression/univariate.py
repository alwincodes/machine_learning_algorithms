from matplotlib.axis import YAxis
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    #load the data we are using pandas
    file_location = "../../Data/Salary_Data.csv"
    df = pd.read_csv(file_location)

    """
    split data into X and y. Here we are not using testing and cross validation sets because we have limited data 
    """

    X = df.iloc[:, :1].values
    y = df.iloc[:, 1:].values

    #we will plot our data to see what we are working with

    plt.title("Salary / Experience")
    plt.scatter(X, y)
    plt.show()

if __name__ == "__main__":
    main()

