# Spatio-temporal Time-Series Forecasting using an Iterative Kernel-Based Regression

Spatio-temporal time-series analysis is a growing area of research that includes different types of tasks, such as forecasting, prediction, clustering, and
visualization.

In many domains, like epidemiology or economics, time series data is collected in order to describe the observed phenomenon in particular
locations over a predefined time slot and predict future behavior.

Regression methods provide a simple mechanism for evaluating empirical functions over scattered data points. In particular, kernel-based regressions are suitable for cases in which the relationship between the data points and the function is not linear.

In this work, we propose a kernel-based iterative regression model, which fuses data from several spatial locations for improving the forecasting
accuracy of a given time series. In more detail, the proposed method approximates and extends a function based on two or more spatial input modalities
coded by a series of multiscale kernels, which are averaged as a convex combination.

The proposed spatio-temporal regression resembles ideas that are present in deep learning architectures, such as passing information between scales. Nevertheless, the construction is easy to implement and it is also suitabl for modeling data sets of limited size. 

Experimental results demonstrate the proposed model for solar energy prediction, forecasting epidemiology infections and future number of fire events.

The method is compared with well-known regression techniques and highlights the benefits of the proposed kernel-based regression in terms of accuracy and flexibility.
