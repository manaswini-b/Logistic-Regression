# Logistic-Regression
## Implementing a perceptron for logistric regression. 
For the training data, generated 3000 training instances in two sets of random data points (1500 in each) from multi-variate normal distribution with µ1 = [1,0], µ2 = [0,1.5], Σ1 =[[1,0.75],[0.75,1]], Σ2 =[[1,0.75],[0.75,1]] and labelled them as 0 and 1.<br/> <br/>
Test data is generated in the same manner but sampled 500 instances for each class, i.e., 1000 in total.<br/><br/>
Used sigmoid function for activation function and cross entropy for objective function, and performed batch training.<br/><br/>
Maximum number of iterations are set to 3000.<br/><br/>
Plotted an ROC curve and computed Area Under the Curve (AUC) to evaluate the implementation
