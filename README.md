# A Model to Identify Risky Loans

This is a report that describes the steps taken to develop a prediction model capable of identifying risky bank loans.  This model uses a cleaner version of the german credit dataset donated to the UCI Machine Learning Repository by Hans Hofmann from the University of Hamburg. The original dataset can be found on the [UCI Machine Learning Repository]  (https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data). The cleaner version of the german credit dataset used in this report  is available thanks to Brett Lantz and his book "Machine Learning with R - Expert techniques for predictive modeling", 3rd edition, Packt Publishing - see the References section below.

The dataset contains information about 1000 loans, along with demographics about their corresponding loan applicants, the size, purpose and period of the loan and if those loans eventually went into default or no.  

The method that we'll use to judge how well our algorithm is able to predict default loans will be the overall accuracy and sensitivity (i.e. the percentage of defaulted loans that have been correctly predicted as default). These measures will be applied on a test set - a dataset the we have not used when building our algorithm but for which we already know the actual default status of the loans.

The approach used to build our prediction algorithm takes into account a few steps:
- first we split our credit data into a train dataset (90% of the data) and a final holdout test set (remaining 10% of the data).  
- we further split our train dataset into training and testing datasets to be able to train and test our algorithms. 
- next we use Rpart regression trees algorithm and  2 implementations of the decision tree algorithms: C5.0 and Random Forests. Unlike other classification machine learning algorithms which are like black boxes (i.e. hard to understand the way classification decisions have been made) the advantage of using decision trees is that the results are easy to read in plain language. Initially we will use the default parameters of each algorithm and later we will use cross-validation to tune and pick the parameters that give us the highest accuracy and sensitivity.  
- in the end we compare the algorithms and we choose the best one to test on the final holdout test set.

The objective will be to build a predictive model that would give us over 75% accuracy and the highest possible sensitivity meaning that we would be able to correctly predict the highest number of loans that have defaulted.
