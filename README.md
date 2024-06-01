# Principal Component Analysis

What Is Principal Component Analysis?
Principal component analysis, or PCA, is a dimensionality reduction method that is often used
to reduce the dimensionality of large data sets, by transforming a large set of variables into a
smaller one that still contains most of the information in the large set.
Reducing the number of variables of a data set naturally comes at the expense of accuracy, but
the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize, and thus make analyzing data points much easier and faster for machine learning algorithms without extraneous variables to process.
So, to sum up, the idea of PCA is simple: 
Reduce the number of variables of a data set, while preserving as much information as possible.

# Scree Plot

To help determining the number of principal components to retain for further analysis, each
point on the scree plot represents an eigenvalue, which is a measure of the variance explained
by a particular principal component. The principal components are ordered by the amount of
variance they explain, with the first principal component explaining the most variance and
subsequent components explaining progressively less variance.
In the scree plot above, the x-axis represents the number of principal components (PC), and the
y-axis represents the eigenvalue (variance ratio). The eigenvalues tend to decrease sharply at
first, then level off after a certain number of components. The "elbow" of the curve is often
used as a heuristic to decide how many components to keep. In this case, the elbow appears to
be around the 5th or 6th principal component. This suggests that the first 5 or 6 principal
components capture the most important information in the data set.
Ultimately, the decision of how many principal components to retain is a balance between
capturing the most important information in the data set and avoiding overfitting. Overfitting
occurs when the model is too complex and captures noise in the data rather than the
underlying structure.

# Score Plot

A score plot is a scatter plot that shows the relationship between two principal components
(PCs). Principal components are uncorrelated variables created by PCA to capture the most
variance in a high-dimensional data set.
In this specific score plot, the x-axis represents the first principal component (PC1) scores and the y-axis represents the second principal component (PC2) scores. Each data point in the plot represents an observation in the original data set.

1.There are clusters of points in the plot. This suggests that there may be underlying
groups or structures in the data set.
2.There are some points that appear to be outliers. These points are further away from
the other points in the plot and may represent unusual observations in the data set.
3.The overall shape of the point distribution is elliptical, which suggests that the data may
be normally distributed.

A score plot only shows the information captured by the first two principal components. If there are more than two principal components, the score plot will not show the complete picture. Hence we could do more two by two comparison using other PCAs
we captured.

# Loading Plot

A loading plot is a way to visualize the relationship between the original variables and the
principal components. Each point in the plot represents a variable, and the position of the point shows how much that variable contributes to each of the principal components.
The axes of the loading plot represent the principal components (PC), and the distance of a
variable from the origin represents the strength of its contribution. Variables with loadings
closer to 1 or -1 on a particular PC contribute more strongly to that component, while variables
with loadings closer to 0 contribute less. The sign of the loading indicates the direction of the relationship. A positive loading means that the variable increases along with the principal
component, while a negative loading means that the variable decreases as the principal
component increases.
In this loading plot, it appears there are two principal components (PC1 and PC2)
shown.

1.Feature 14 and Feature 11 have high positive loadings on PC1 and negative loadings on
PC2. This suggests that these two features are positively correlated with each other and
tend to decrease as PC2 increases.
2.Feature 1 and Feature 15 have high negative loadings on PC1 and positive loadings on
PC2. This suggests that these two features are positively correlated with each other and
tend to increase as PC2 increases.
3.Feature 3 and Feature 10 have positive loadings on both PC1 and PC2. This suggests that
these two features tend to increase with both principal components.
4.Feature 4 and Feature 6 have negative loadings on both PC1 and PC2. This suggests that
these two features tend to decrease with both principal components.

It is important to keep in mind that a loading plot only shows the loadings for the first two
principal components. If there are more than two principal components, the loading plot will
not show the complete picture of how the variables contribute to all of the components.
