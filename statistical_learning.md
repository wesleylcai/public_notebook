# Table of Contents
[AdaBoost](#adaboost)

[Gradient Boost](#gradient-boost)
<!---toc--->


# AdaBoost

I'm learning this method to gain a fundamental understanding of XGBoost, which is a popular classifier algo and used in Litchfield et al, 2021.

#### StatQuest
<img src="images/statistical_learning/statquest.adaboost.t1.jpg" width="300">

Table 1

Concept
1. Combines a lot of "weak learners" to make classifications
2. Weak leaners are usually stumps (1 node, 2 leaf decision trees)
3. Stumps are weighted
4. The order of each stump is important: errors of first stump influence that of next, etc

Algorithm
1. Each sample (row of table with column factors and classification as last column) gets a sample weight, starting as 1/N.
2. First take the single factor that does the best job classifying samples (example chest pain in Table 1, to do this calculate Gini Impurity index for each stump)
    1. Gini impurity for each leaf = 1 - P(Yes)^2 - P(No)^2
    2. Index = (N_leaf1/N_total) * G(leaf1) + (N_leaf2/N_total) * G(leaf2)
3. Calculate Gini Impurity index for each stump and use the stump with lowest index as the first stump
    1. Example: Chest pain yes -> how many correct yes and incorrect yes? Chest pain no -> how many correct no and incorrect no?
    2. <img src="images/statistical_learning/statquest.adaboost.f3.jpg" width="300">
4. Calculate total error for the first stump (sum of **SAMPLE WEIGHTS** of incorrectly classified samples divided by total samples)
5. `Amount of say = 1/2 log((1 - Total Error)/Total Error)`
    1. Figure 1, red line
    2. Total Error goes from 0 to 1
    3. The graph looks like a log plot rotated 90 deg
        1. Low error = positive weight
        2. 50% error is like coin flip = 0 weight
        3. High error = negative weight
        4. Equation is not good for 0 and 1, so error term is added in practice
6. Update all sample weights before moving on to next stump, higher sample weights for incorrectly classified samples
    1. **Incorrectly classified samples:** `New sample weight = sample weight * exp(amount of say)`
        1. Why exp? If the amount of say is good, we scale with a large number
        2. Figure 1, blue line
    2. **Correctly classified samples:** `New sample weight = sample weight * exp(-amount of say)`
        1. Figure 1, purple line
    3. <img src="images/statistical_learning/statquest.adaboost.f1.png" width="200"> Figure 1
7. Normalize new sample weights to get 1 (divide each new sample weight by sum of total new weights)
8. Generate the second stump, using new weights
    1. Can use weighted Gini index to determine variable for next stump
    2. Or generate another table that takes into account weights
9. In this case, use second method
    1. Randomly pick a number from 0 to 1
    2. Use the new weights of samples as like a cumulative function, depending on where the random number falls, pick the sample with the closest cumulative weight
        1. <img src="images/statistical_learning/statquest.adaboost.f2.jpg" width="500"> Figure 2
    3. Iterate this process until new table samples match the old
    4. Assign baseline weights 1/N
    5. Use this new table to pick the variable for the second stump
10. Final classifier: Using the forest of stumps, separate into stumps with "Has heart Disease" and those with "Does not have heart disease" based on the new sample
11. Add up amount of say in each group of stumps. Higher sum of amount of say = final class

#### Elements of Statistical Learning

Algorithm 10.1 AdaBoost.M1.
1. Initialize the observation weights ![f1]
2. For ![f2]
    1. Fit a classifer ![f3] to the training data using weights ![f4].
    2. Compute ![f5]
    3. Compute ![f6]
    4. Set ![f7]
3. Output ![f8]

The I() function must output either -1 or 1, I believe.

[f1]: https://chart.apis.google.com/chart?cht=tx&chl=w_i=1/N,\\;\\;i=1,\\;2,\\;...,\\;N
[f2]: http://chart.apis.google.com/chart?cht=tx&chl=m=1\\;\\;to\\;\\;M
[f3]: http://chart.apis.google.com/chart?cht=tx&chl=G_m(x)
[f4]: http://chart.apis.google.com/chart?cht=tx&chl=w_i
[f5]: http://chart.apis.google.com/chart?cht=tx&chl=err_m=\frac{\sum_{i=1}^{N}w_iI(y_i\\;\neq\\;G_m(x_i))}{\sum_{i=1}^{N}w_i}
[f6]: http://chart.apis.google.com/chart?cht=tx&chl=\alpha_m=log((1-err_m)/err_m)
[f7]: http://chart.apis.google.com/chart?cht=tx&chl=w_i\\;\leftarrow\\;w_i\\;\cdot\\;exp[\alpha_m\cdot\\;I(y_\\;\neq\\;G_m(x_i))]\\;,\\;\\;i=1,\\;2,\\;...,\\;N
[f8]: http://chart.apis.google.com/chart?cht=tx&chl=G(x)=sign[\sum_{m=1}^{M}\alpha_mG_m(x)]

#### When to use?
This looks like it's best for binary classification.

# Gradient Boost

Objective: Understanding this to get into the guts of XGBoost

## Statquest
<img src="images/statistical_learning/statquest.gradientboost.t1.jpg" width="300">

Table 1

Concept

1. Gradient boost is similar to Adaboost except it uses leaf instead of stump
2. Then builds tree around leaf, constrained by pre-determined number of leaves. Unlike Adaboost, it scales all trees the same.

Algorithm

1. First leaf is the average of all weights (71.2)
2. Calculate pseudo-residuals (pseudo because Gradient boost not linear regression) into another column
3. Build a tree with column variables to predict residuals
    a. f more than 1 variable per leaf, calculate average of variables




## Elements of Statistical Learning


## When to use?

