<h1>Physicochemical Analysis & Predictive Modeling of Viticulture Quality</h1>

<h2>The Context</h2>
In the wine industry, quality assessment is traditionally a subjective process involving human "sommeliers." 
However, sensory perception is inconsistent and expensive. There is a significant industrial need to correlate 
<b>objective physicochemical properties</b> (like pH, residual sugar, and alcohol content) with <b>perceived quality scores</b>
to automate grading and ensure production consistency.

<h2>The Goal</h2>
The goal of this project is to develop a robust mathematical framework to model the relationship between 11 chemical 
input features and a discrete quality rating (3–8) using a dataset of approximately 1,600 red wine samples.

<h3>The Core Challenge is Two-Fold:</h3>

  * Predictive Accuracy: Determining whether a wine sample is "high quality" based on chemical signatures.
  * Feature Significance: Mathematically identifying which chemical markers (e.g., volatile acidity vs. alcohol) 
   have the highest impact on the final quality score.

<h2>Technical Constraint</h2>
To ensure a deep understanding of the underlying algorithmic logic, this project implements ML algorithms like Linear 
Regression, Decision Tree, Random Foreest from scratch.

<h2>The Exploratory Data Analysis (EDA) Phase</h2>

  * First, I plotted the box plot to analyse and detect the outliers in the dataset.
  * While the initial box plots suggested a linear relationship for <b>Citric Acid</b> and <b>Alcohol</b>, the visualizations
  for the remaining physicochemical features (notably Residual Sugar, Chlorides, and Sulfur Dioxide) appeared statistically misleading
  (as the plot presented a very small Interquartile Range (IQR) and massive amount of outliers). Which led me to mathematically calculate 
  the skewness score of the features and to analyse of probability density of each features using <b>Violin Plots</b> and 
  <b>Kernel Density Estimate (KDE) plots</b>
  * I also used Correlation Heatmap to understand the multicolinearity of the features.</li>
  * To calculate skewness, we look at the <b>third standardized moment</b>.<br>
    The formula for a population skewness is:<br> $$\gamma = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2\right)^{3/2}}$$
    <br>Numerator: The average of the cubed deviations from the mean (emphasizes the tail)
    <br>Denominator: The standard deviation cubed (normalizes the scale)
  * And the following interpretation is used for determining skewness:<br>
      <table border="4" cellspacing="1" align="center" width="50%" height="50%">
        <tr>
         <td align="center"><b>Skewness Score</b>
         <td align="center">Interpretation
        </tr>
        <tr>
         <td align="center">>1.0
         <td align="center">Highly-Skewed
        </tr>
        <tr>
         <td align="center">o.5 to 1.0
         <td align="center">Moderately-Skewed
        </tr>
        <tr>
         <td align="center">-0.5 to 0.5
         <td align="center">Approximately Symmetric
        </tr>
      </table>
  * As expeceted, majority of features have skewness! With "chloride: 5.6461", "residual sugar: 4.3955" and "sulphates: 2.5525" dominating the list.
  * This level of skewness is really bad for the gradient descent.
   
  * To fix this, I used log transformation and standard/robust scaling as follows:

  $$
  f(x) = 
  \begin{cases} 
  \text{Standardize}(\ln(x+e)) & \text{if } \gamma > 1.0 \\
  \text{RobustScale}(x) & \text{if } 0.5 < \gamma \leq 1.0 \\
  \text{Standardize}(x) & \text{if } \gamma \leq 0.5 
  \end{cases}
  $$

  where $\gamma$ is the Fisher-Pearson Skewness coefficient.

<h2>Buliding Regression Models</h2>

 <h3>Linear Regression</h3>
  I implemented a custom linear regression model, which acts as the performance baseline. This involved two distinct mathematical approaches to solving for the optimal weight vector W.
  <h3>1. Gradient Descent (Iterative Optimisation)</h3>
  Developed a robust Gradient Descent solver to minimize the Mean Squared Error (MSE) cost function J(W).
  
  * **Mean Square Error Cost Function:** $$J(W) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2$$
<br>where $h_w = w^{T}x$

  * The **Gradient** of MSE is given by: $$\nabla J(W) = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

  * We can translate this summation above into a much faster **Vectorized Matrix Multiplication** which avoids using a for loop for that $\sum$ symbol and calculate the entire gradient for all features at once: $$\nabla J(W) = \frac{1}{m} X^T (XW - y)$$
<br>where $XW - y$ is our "Error Vector".

  * **The Update Rule:** $$W = W - \frac{\alpha}{m} X^T (XW - y)$$
  * **Optimization:** The feature scaling ensured the cost function contours were circular, leading to faster convergence.

  <h3>2. Closed Form Solution (Analytical Solution)</h3>
  Implemented the closed-form solution to find the global minimum of the Cost Function without iteration. 
  
  $$W = (X^T X)^{-1} X^T y$$

  * **Key Insight:** While computationally expensive for large datasets ($O(n^3)$ due to matrix inversion), it provided the exact theoretical "best fit" for the wine chemistry features.
  * **Implementation:** Used NumPy's linear algebra module to handle the transpose and inversion of the feature matrix.

  <h3>Why Both?</h3>
  Implementing both allowed for a direct comparison between Iterative Learning and Analytical Solving.
  
  * **The Baseline Result:** Both methods converged to a Test MSE of ~0.42 (which was also compared to skikit-learn module's linear regression model), revealing that the linear relationship between chemistry and quality was insufficient to capture the complexities of the dataset.
  
  * **The Motivation:** This "Linear Ceiling" provided the mathematical justification to move toward non-linear, high-capacity models like the Decision Tree and Random Forest.

 <h3>Decision Tree</h3>
  A Decision Tree Regressor replaces the linear equation with a series of hierarchical "if-then" splits. This allows the model to partition the feature space into localized regions.
  
  * **Non-Linearity:** It naturally captures non-linear relationships by segmenting data based on variance reduction (MSE).
  
  * **Automatic Interaction Detecetion:** For example, by splitting on "Alcohol" and then immediately splitting that branch on "Sulphates," the tree implicitly learns how these variables interact without human intervention.

  * **Mathematical Basis (MSE Reduction):** At each node, the tree chooses a split $s$ for feature $j$ that minimizes the sum of squared errors:

  $$\sum_{i: x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2$$
    
  <br>Where, <br>

  $\hat{y}_{R_1}$: The average wine quality of all samples in the first bucket.

  $\hat{y}_{R_2}$: The average wine quality of all samples in the second bucket.

  <h3>The Greedy Nature of Decision Tree</h3>
  
  * Decision Tree is a Greedy Algorithm.
  * It make the best possible split right now at the current node.
  * It doesn't look ahead to see if a slightly "worse" split now might lead to a much better split two steps down the line.
  * This is why a single tree can sometimes get stuck in a "local" pattern that doesn't generalize well (which probably what happened earlier to give worse MSE than fo linear regression), which leads right back to the **Random Forest** solution—where we use many greedy trees to balance each other out.
 <h3>Random Forest</h3>
  While a single Decision Tree is powerful, it is notoriously "greedy" and prone to overfitting (high variance). A Random Forest solves this by using an ensemble technique called Bagging (Bootstrap Aggregating).
  
  * **Average of All Trees:** It trains hundreds of diverse trees on different subsets of the data and features, then averages their predictions. This cancels out the "noise" from individual trees.
  
  * **Feature Importance:** Random Forests provide a clear ranking of which chemical features (e.g., Alcohol vs. Chlorides) actually drive the variance in quality. Every time the tree successfully splits a node, it should calculate the difference:$$Importance = RSS_{parent} - (RSS_{left} + RSS_{right})$$If we track that drop for every feature across all 100 or more trees, we get a definitive ranking of which chemicals dictate wine quality.
  
  * **Out-of-Bag (OOB) Error:** It provides a built-in validation mechanism, allowing us to estimate test performance during the training phase. Because we draw bootstrap samples with replacement, mathematically, about 36.8% of your training data is left out of every individual tree. Instead of needing a separate validation set, we use this left out data points to test the tree on data points it didn't see, for an instant, built-in metric for determining how well the forest is generalizing.
  
  * **Grid Search Optimization:** Systematic hyperparameter tuning (Depth, Estimators, Features) to find the performance ceiling.

 <h2>Performance Comparison</h2>

 <table border="4" cellspacing="1" align="center" width="50%" height="50%">
        <tr>
         <td align="center"><b>Model  Type</b>
         <td align="center">Train MSE
         <td align="center">Test MSE
        </tr>
        <tr>
         <td align="center">Linear Regression
         <td align="center">0.4091
         <td align="center">0.4238
        </tr>
        <tr>
         <td align="center">Decision Tree
         <td align="center">0.3916
         <td align="center">0.4889
        </tr>
        <tr>
         <td align="center">Random Forest
         <td align="center">0.1610
         <td align="center">0.3824
        </tr>
        <tr>
         <td align="center">Random Forest (Best Estimator)
         <td align="center">0.0438
         <td align="center">0.3681
        </tr>
      </table>

 <h2>The "Bayes Error" Conclusion</h2>
 After extensive fine-tuning (reaching a Test MSE of ~0.37), the model hit a plateau. This identifies the Irreducible Error of the dataset, stemming from:

 * **Subjective Labels:** Wine quality scores (3-8) are human-assigned and sensory evalution is inherently noisy.
 *  **Latent Variables:** Missing data (e.g., grape origin, aging process) limits predictive power.

 <h2>Feature Insights</h2>
 The model consistently identified three primary chemical drivers for quality:

  * **Alcohol:** The strongest positive predictor.
  * **Sulphates:** Secondary positive predictor for structural integrity.
  * **Volatile Acidity:** The primary negative predictor (Vinegar indicator).

 <h2>Technical Stack</h2>

  * **Language:** Python
  * **Libraries:** NumPy (Matrix Ops), Pandas (Data Handling), Matplotlib/Seaborn (Visualization), Scikit-Learn (Validation)
  * **Environment:** Pixi / VS Code
