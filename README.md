# Machine-Learning
 [Regression] You will implement linear regression as part of this question for the dataset provided. For each sub-question, you are expected to report the following - (i) plot of the best fit curve, (ii) equation of the best fit curve along with coefficients, (iii) value of final least squared error over the test data and (iv) scatter plot of model output vs expected output and for both train and test data. You can also generate a .csv file with your predictions on the test data which we should be able to reproduce when we run your command-line instruction.
Note that you can only regress over the points in the train dataset and you are not supposed to fit a curve on the test dataset. Whatever solution you get for the train data, you have to use that to make predictions on the test data and report results.

part(a). Use standard linear regression to get the best fit curve. Vary the maximum degree term of the polynomial to arrive upon an optimal solution.
part(b).In the above problem, increase the maximum degree of the polynomial such that the curve overfits the data.
part(c).Use ridge regression to reduce the overfit in the previous question, vary the value of lambda (λ) to arrive at the optimal value. Report the optimal λ along with other deliverables previously mentioned.


[Classification] You will implement classification algorithms that you have seen in class as part of this question. You will be provided train and test data as before, of which you are only supposed to use the train data to come up with a classifier which you will use to just make predictions on the test data. For each sub-question below, plot the test data along with your classification boundary and report confusion matrices on both train and test data. Again, your code should generate a .csv file with your predictions on the test data as before.
(a) (2 marks) Implement the Preceptron learning algorithm with starting weights as w = [0,1]T for x = [x,y]T and with a margin of 1.
(b) (1 mark) Calculate (code it up!) a Discriminant Function for the two classes assuming Normal distribution when the covraiance matrices for both the classes are equal and C1 = C2 = σ2I for some σ.
(c) (1 mark) Calculate a Discriminant Function for the two classes assuming Normal distri- bution when both C1 and C2 are full matrices and C1 = C2.
(d) 1 mark) Calculate a Discriminant Function for the two classes assuming Normal distri- bution when both C1 and C2 are full matrices and C1 ̸= C2.

