"""
@ Computational Data Mining @

@ Final Project @

@ author: sina heydari @

@ student ID: 14024133 @
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining a class for Principal Component Analysis
class PCA:
    
    # Constructor method to initialize the class instance
    def __init__ (self, Dataorg):
        
        # Initializing the original data
        self.Data = Dataorg
        
        # Preprocessing the data: dropping first column and resetting index
        self.Data = self.Data.iloc[: , 1:]
        self.Data = self.Data.reset_index(drop=True)
        
        # Converting data to numpy array for further processing
        self.Dataarray = np.array(self.Data)
        
        
    # Method to scale the data (standardization)
    def Scale(self):
               
        # Copying the data to a new variable
        ScaledMatrix = self.Dataarray
        
        # Calculating number of rows and columns in the data
        numberofrows = (len(ScaledMatrix[:,0]))
        numberofcolumns = (len(ScaledMatrix[0,:])) 
        
        # Calculating mean of each column
        self.mean = np.mean(ScaledMatrix, axis = 0)
        
        # Subtracting mean from each element in the respective column
        for column_index in range(0, numberofcolumns-1):
            
            for row_index in range(0, numberofrows-1):
                ScaledMatrix[row_index, column_index] = ScaledMatrix[row_index, column_index] - self.mean[column_index]
                
        # Storing scaled data matrix
        self.ScaledDataMatrix = ScaledMatrix
        
    # Method to compute the covariance matrix
    def CovarianceMatrix(self):
        
        # Computing covariance matrix of the scaled data
        self.Covarincematrix = np.cov(self.ScaledDataMatrix , rowvar=False)
        
    # Method to compute eigenvalues and eigenvectors
    def EigenValuesVectors(self):
        
        # Computing eigenvalues and eigenvectors of the covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.Covarincematrix)
        
    # Method to sort the indices of eigenvalues
    def SortIndices(self):
        
        # Sorting the indices of eigenvalues in ascending order
        self.EigenIndices = np.argsort(self.eigenvalues)
        
    # Method to sort eigenvectors and eigenvalues based on sorted indices
    def SortEigenVectors(self):
        
        # Sorting eigenvectors and eigenvalues based on sorted indices
        self.SortedEigenVectors = self.eigenvectors[self.EigenIndices]
        self.SortedEigenValues = np.flip(self.SortEigenVectors)
        
        self.SortedEigenValues = self.eigenvalues[self.EigenIndices]
        self.SortedEigenValues = np.flip(self.SortedEigenValues)
        
    # Method to create a Scree Plot
    def ScreePlot(self):
        # Computing eigenvalue variance ratio
        sumofeigenvalues = sum(self.SortedEigenValues)
        X_axis_eigenvalues = np.zeros(len(self.SortedEigenValues))
        for i in range(0 , (len(self.SortedEigenValues))):
            X_axis_eigenvalues[i] = self.SortedEigenValues[i]/sumofeigenvalues
        
        # Plotting Scree Plot
        plt.bar([i for i in range(1, 201)], X_axis_eigenvalues, color="black", alpha=1 )
        plt.title("Scree Plot")
        plt.xlabel("Principal Components")
        plt.ylabel("Eigenvalues Variance Ratio")
        
        # Marking breakpoint for the fifth principal component
        plt.axhline(y=X_axis_eigenvalues[4], color='red', linestyle="dotted", linewidth=1, label="PC5 Breakepoint")
        
        plt.legend()
        plt.savefig('high_resolution_scree_plot.png', dpi=1500)
        
    # Method to create a Score Plot
    def ScorePlot(self):
        
        # Selecting number of principal components for plotting
        numofPCAs = 2
        selected_eigenvectors = self.SortedEigenVectors[:, :numofPCAs]
        
        # Projecting data onto selected principal components
        scoreplt = np.dot(self.ScaledDataMatrix, selected_eigenvectors)
        PC1 = scoreplt[:, 0]
        PC2 = scoreplt[:, 1]
        
        # Coloring data points based on a threshold
        color = np.array(['blue' for i in range(400)])
        for i in range(199, 400):
            color[i] = 'red'
                
        # Plotting Score Plot
        plt.scatter(PC1, PC2, c=color)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Score Plot')
        
        # Adding legend for data points
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='PC1')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='PC2')
        plt.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        plt.savefig('high_resolution_score_plot.png', dpi=1500)
        
    # Method to create a Loading Plot
    def LoadingPlot(self):
        
        # Setting parameters for plot
        max_arrow_length = 0.5
        max_loading_value = np.max(np.abs(self.SortedEigenVectors * self.SortedEigenValues[:, np.newaxis]))
        scaling_factor = max_arrow_length / max_loading_value
         
        features_num = min(15, len(self.SortedEigenVectors))
        
        # Plotting Loading Plot
        for i in range(features_num):
            plt.arrow(0, 0, self.SortedEigenVectors[i, 0] * self.SortedEigenValues[i] * scaling_factor,
                      self.SortedEigenVectors[i, 1] * self.SortedEigenValues[i] * scaling_factor,
                      color='maroon', width=0.0001, head_width=0.0005)
            plt.text(self.SortedEigenVectors[i, 0] * self.SortedEigenValues[i] * scaling_factor,
                     self.SortedEigenVectors[i, 1] * self.SortedEigenValues[i] * scaling_factor,
                     f'Feature {i + 1}', fontsize=8, c='black')
     
        plt.title('Loading Plot')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('high_resolution_Loading_plot', dpi=1000)
        plt.show()

######################################################################################################

# Creating an instance of PCA class and performing PCA

PCA_instance = PCA(pd.read_csv("dataMatrix.csv"))
PCA_instance.Scale()
PCA_instance.CovarianceMatrix()
PCA_instance.EigenValuesVectors()
PCA_instance.SortIndices()
PCA_instance.SortEigenVectors()

# Uncomment below lines one by one to generate respective plots :
    
#PCA_instance.ScreePlot()
#PCA_instance.ScorePlot()
#PCA_instance.LoadingPlot()
