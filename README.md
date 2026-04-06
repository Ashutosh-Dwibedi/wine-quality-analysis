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
