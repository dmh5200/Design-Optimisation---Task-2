README
Overview
This repository contains two main scripts: AdaptiveSamplingFinal2 and TestScript. These scripts are used for adaptive sampling and testing purposes, respectively. AdaptiveSamplingFinal2 is designed for use during the black box test day, while TestScript is used for evaluating the performance of the adaptive sampling strategy.

Scripts
1. AdaptiveSamplingFinal2
Purpose: AdaptiveSamplingFinal2 is used for the black box test day to perform adaptive sampling on a given black box function. The script prompts the user to manually input the black box outputs for the sampled points and updates the Radial Basis Function (RBF) model accordingly.

Usage:

Inputs:

ub: Upper bounds of the input variables.
lb: Lower bounds of the input variables.
res: Resolution of the input variables.
Execution:

The script generates initial samples using Latin Hypercube Sampling (LHS).
The user is prompted to manually input the black box outputs for the initial samples.
The RBF model is updated with the new samples.
The script iteratively selects new sample points using the Expected Improvement (EI) acquisition function and prompts the user to input the black box outputs for these points.
The RBF model is updated with each new sample.
The final sample is used to check the actual value of the predicted global minimum.
Example:

2. TestScript
Purpose: TestScript is used for testing the performance of the adaptive sampling strategy. It runs the adaptive sampling process multiple times and evaluates the results.

Usage:

Inputs:

The user selects the black box function from a menu.
The user inputs the upper and lower bounds for the input variables.
The user inputs the performance variables, including the number of initial samples, the number of final samples, and the value of r.
Execution:

The script runs the adaptive sampling process for a specified number of runs.
The results, including the global minimum values, locations, and goodness scores, are recorded.
The script plots the distribution of goodness scores and overlays the mean and standard deviation.
Example:

Detailed Steps for AdaptiveSamplingFinal2
Generate Initial Samples:

Load the LHS samples from a file.
Prompt the user to input the black box outputs for the initial samples.
Normalize and denormalize the samples as needed.
Iterative Sampling:

For each iteration, define the acquisition function (EI+).
Optimize the acquisition function to find the next sample point.
Prompt the user to input the black box output for the next sample point.
Update the RBF model with the new sample.
Final Optimization:

Use the final sample to check the actual value of the predicted global minimum.
Optimize the surrogate model to find the best sample point.
Detailed Steps for TestScript
User Input:

Display a menu for the user to select the black box function.
Prompt the user to input the upper and lower bounds for the input variables.
Prompt the user to input the performance variables.
Run Adaptive Sampling:

Run the adaptive sampling process for the specified number of runs.
Record the results, including the global minimum values, locations, and goodness scores.
Plot Results:

Plot the distribution of goodness scores.
Overlay the mean and standard deviation on the plot.