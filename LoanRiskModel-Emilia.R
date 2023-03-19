
### A MODEL TO IDENTIFY RISKY LOANS


# Let's first call the packages that we will use in this analysis

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(C50)) install.packages("C50", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(gmodels)) install.packages("gmodels", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(C50)
library(randomForest)
library(data.table)
library(dplyr)
library (knitr)
library (GGally)
library(gmodels)

#### COLLECTING THE DATA

# Clean german dataset on Kaggle with 9 attributes:
# https://www.kaggle.com/datasets/uciml/german-credit/download?datasetVersionNumber=1

# downloading the clean credit file from github
dl <- "credit.csv"
if(!file.exists(dl))
  download.file("https://raw.githubusercontent.com/emilia-iacob/LoanRisk/main/credit.csv", dl)

# creating the credit object
credit <- read.csv("credit.csv", stringsAsFactors = TRUE)

# saving the credit object for later use
save(credit, file = "credit.rda")

# loading the credit object at the start of a new R session
load("credit.rda")

### PREPARING THE TRAIN DATASET

# We will now split the credit data into a train dataset (aka "credit_main") on which we will train and test multiple algorithms 
# and a final holdout test set (aka "credit_final_holdout_test") on which we'll evaluate our best performing algorithm
# The train dataset will be 90% of the credit dataset and the final holdout test will be the remaining 10%
set.seed (123, sample.kind = "Rounding")
test_index <- createDataPartition(y=credit$default, times = 1, p = 0.1, list = FALSE)
credit_main <- credit [-test_index,]
credit_final_holdout_test <- credit [test_index,]

# Now that we have created the train dataset (aka credit_main) we'll split it further into a train and test dataset with 90% and 10% of the credit_main dataset respectively.
set.seed(123, sample.kind = "Rounding")
main_test_index <- createDataPartition( y= credit_main$default, times = 1, p = 0.1, list = FALSE)
credit_main_train <- credit_main [-main_test_index,]
credit_main_test <- credit_main [main_test_index,]


# To save the resulting datasets for future use we'll do the following:
save(credit_main, file = "credit_main.rda")
save(credit_final_holdout_test, file = "credit_final_holdout_test.rda")
save(credit_main_train, file = "credit_main_train.rda")
save(credit_main_test, file = "credit_main_test.rda")


# To quickly load the saved credit_main and credit_final_holdout_test files and go to the next steps of this analysis 
# we can use the load function:
load("credit_main.rda")
load("credit_final_holdout_test.rda")
load("credit_main_train.rda")
load("credit_main_test.rda")



#### EXPLORING THE CREDIT DATASET

dim(credit)
str(credit)
# describing numerical features only
describe (credit, omit = TRUE, quant = c(.25, .75), IQR = TRUE, check = FALSE)

# let's now see how many of these loans have defaulted
table(credit$default)

# let's see the distribution of the checking balance
summary(credit$checking_balance)
# and now the same distribution in percentages
prop.table(table(credit$checking_balance))*100

# let's see the distribution of checking_balance by default status:
credit %>% ggplot(aes(x=checking_balance, fill = default)) +
  geom_bar(position = "dodge")  + 
  ggtitle ("Checking balance distribution by loan default status")

# let's see the distribution of savings_balance by default status:
credit %>% ggplot(aes(x=savings_balance, fill = default)) +
  geom_bar(position = "dodge") +
  ggtitle ("Savings balance distribution by loan default status")


# distribution by purpose of the loan
credit %>% ggplot(aes(x=purpose, fill = default)) +
  geom_bar()+
  facet_grid(rows = vars(default))
  
# distribution by credit history
credit %>% ggplot(aes(x=credit_history, fill = default)) +
  geom_bar(position = "dodge")

# let's now explore the distribution of credit history by default status with the CrossTable function from the gmodels package
# the CrossTable function also offers us the calculation for Pearson's chi-squared test for independence between 2 variables
# This test tells us how likely it is that the difference in cell counts in the table is due to chance alone. The lower the chi-squared values
# the higher the evidence that there is an association between the 2 variables

prop.table(table(credit$credit_history, credit$default))
CrossTable(x=credit$default, y= credit$credit_history, chisq = TRUE)

# let's apply the CrossTable function to savings and checking balances 
# and see if there is an association between the default status and the savings and checking balances respectively 
CrossTable(x = credit$default , y = credit$checking_balance, chisq = TRUE)


# let's create a correlogram with ggpairs
credit_1 <- credit %>% select(default, months_loan_duration, amount, percent_of_income, age, existing_loans_count)
ggpairs(credit_1, ggplot2::aes(colour= default))

# let's also calculate the correlations between some of the numerical attributes
cor(credit$amount, credit$existing_loans_count)
cor(credit$years_at_residence, credit$age)
cor(credit$amount, credit$months_loan_duration)
cor(credit$age, credit$months_loan_duration)

# let's see the distribution of the amount of loan based on default status
credit %>% ggplot(aes(x= default, y= amount, fill = default)) + geom_boxplot()
credit %>% filter(default == "yes") %>% select (amount) %>% summary(credit$amount)
credit %>% filter(default == "no") %>% select (amount) %>% summary(credit$amount)

# let's see the distribution of the number of months of loan based on default status
summary(credit$months_loan_duration)

credit %>% ggplot(aes(x= default, y=months_loan_duration, fill = default)) +
  geom_boxplot()

credit %>% filter(default == "yes") %>% select (months_loan_duration) %>%  summary (credit$months_loan_duration)
credit %>% filter(default == "yes") %>% select (age) %>%  summary (credit$age)


# let's also see the distribution of age by default status
credit %>% ggplot(aes(y=age, fill = default)) +
  geom_boxplot()
credit %>% ggplot(aes(x=age, fill = default)) +
  geom_bar(position = "dodge")



### TRAINING AND EVALUATING DIFFERENT MODELS


### First Model: C5.0 Decision Trees with default parameters

# Let's train a decision tree model with the default parameters
set.seed(123, sample.kind = "Rounding")
train_dt <- train(default ~ ., data = credit_main_train, method = "C5.0")

# and let's see a summary of the results:
train_dt

# and also plot them:
ggplot(train_dt)


# To see the resulting tree let's access the finalModel:
train_dt$finalModel
train_dt$bestTune

# We see that 35 predictors have been used in the finalModel. Let's list these predictors
train_dt$finalModel$predictors

# And let's see the importance of each of these predictors:
varImp(train_dt$finalModel)

# Let's now predict and evaluate our model on the test dataset
predict_dt <- predict(train_dt, credit_main_test, type = "raw")

accuracy_dt <- confusionMatrix(predict_dt, 
                               credit_main_test$default, 
                               positive = "yes")$overall["Accuracy"]

sensitivity_dt <- sensitivity(predict_dt, 
                              credit_main_test$default, 
                              positive = "yes")

# Let's create a table with the accuracy of each of the models that we build:
risk_models <- data.frame(Model = "C5.0 Decision Trees (default)", 
                          Accuracy = accuracy_dt,
                          Sensitivity = sensitivity_dt)


# Our accuracy is 71% and the sensitivity rate is only 48% which means that only 48% of the default loans have been predicted correctly. 
# Our model right now cannot be deployed in real life as this would mean that we can only correctly predict 48% of the default loans which would be very costly for the bank
# We need to do better than this.




#### Second model: C5.0 decision trees model with tuned parameters

# let's first see the parameters that can be tuned for the C5.0 model:
modelLookup("C5.0")

# We already saw that the best model with the default parameters of the C5.0 model had 20 trials, no winnowing and the model type was tree

# Let's create a tuning grid to optimize the *model*, *trials* and *winnow* parameters available for the C5.0 decision tree algorithm
tune_grid <- expand.grid(model = "tree",
                         trials = seq(1,30, 2),
                         winnow = FALSE)
tune_grid

# Also, the trainControl() function in the caret package controls the parameters of the train() function.
# Let's tune our model by using the trainControl() function to create a control object (named control) that uses 10 fold cross-validation 
control <- trainControl(method = "cv", number = 10)

# Let's now pass these tuned parameters to the train function again:

set.seed(123, sample.kind = "Rounding")
train_dt_tuned <- train(default ~., credit_main_train, 
                        method = "C5.0",
                        trControl = control,
                        tuneGrid = tune_grid)

train_dt_tuned

# let's compare the best tune for each of the models:
train_dt_tuned$bestTune
train_dt$bestTune

# We can see that the bestTune for this model has now 17 trials compared to our default model which had 20 trials.
# Also, because we are now using 10 fold cross-validation instead of the 25 bootstrapped samples, our sample size has been reduced now to 728 compared to 810 in the default model.
# the size of the tree also dropped from 57 to 55 decisions deep

train_dt$finalModel
train_dt_tuned$finalModel

# let's see how our new model performs
predict_dt_tuned <- predict(train_dt_tuned, credit_main_test, type = "raw")

# let's now calculate accuracy and sensitivity for the new model:
accuracy_dt_tuned <- confusionMatrix(predict_dt_tuned, credit_main_test$default, positive = "yes")$overall["Accuracy"]
sensitivity_dt_tuned <- sensitivity(predict_dt_tuned, credit_main_test$default, positive = "yes")

# and let's add them to our risk_models table:
risk_models <- rbind(risk_models, list("C5.0 Decision Trees (tuned)", accuracy_dt_tuned, sensitivity_dt_tuned))
risk_models %>% kable()



### Third model: Rpart Regression trees (tuned)

# In the rpart method of classification trees the only tuning parameter available is cp (complexity paramenter)
set.seed(123, sample.kind = "Rounding")
train_rpart <- train(default ~., credit_main_train, method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.5, len = 25)))

# If we plot the trained model we notice that the cp that gives the highest accuracy is 0.02
plot(train_rpart)
train_rpart$bestTune

# Let's now see the decisions in the resulting tree 
train_rpart$finalModel

#...and plot them:
plot(train_rpart$finalModel)
text(train_rpart$finalModel, cex=0.75)

# To extract the predictor names from the rpart model that we trained we can use this code below (reference Data Science book) to see the leafs of the tree
ind <- !(train_rpart$finalModel$frame$var == "<leaf>")
tree_terms <- 
  train_rpart$finalModel$frame$var[ind]  %>% 
  unique()  %>% 
  as.character()
tree_terms

# let's see how well the final model performs
predict_rpart <- predict(train_rpart, credit_main_test, type = "raw")

accuracy_rpart <- confusionMatrix(predict_rpart, 
                                  credit_main_test$default, 
                                  positive = "yes")$overall["Accuracy"]
sensitivity_rpart <- sensitivity(predict_rpart, 
                                 credit_main_test$default, 
                                 positive = "yes")

# Let's add these to our risk_models table
risk_models <- rbind(risk_models, list("Rpart Decision Trees (tuned)", 
                                       accuracy_rpart, 
                                       sensitivity_rpart))
risk_models %>% kable()



### Fourth model: Random Forests - default parameters


#Let's use the caret package implementation of random forests algorithm with default parameters

set.seed(123, sample.kind = "Rounding")
train_rf <- train(default ~., credit_main_train, method = "rf")
train_rf
ggplot(train_rf)

# Let's now see the final model
train_rf$finalModel

#To find out the importance of each feature we can use the varImp function from the caret package
varImp(train_rf)

# and we can see which features are used the most by plotting the variable importance:
ggplot(varImp(train_rf))


# let's see our best tune:
train_rf$bestTune

# let's see how this model performs on the test data:
predict_rf <- predict(train_rf, credit_main_test, type = "raw")

# let's calculate the accuracy and sensitivity of our new model:
accuracy_rf <- confusionMatrix(predict_rf, 
                               credit_main_test$default, 
                               positive = "yes")$overall["Accuracy"]
sensitivity_rf <- sensitivity(predict_rf, 
                              credit_main_test$default, 
                              positive = "yes")

# let's add these to our risk_models table for an easy comparison:
risk_models <- rbind(risk_models, list("Random Forests (default)", 
                                       accuracy_rf, 
                                       sensitivity_rf))
risk_models %>% kable()

# We can now see that our new model performs significantly better when it comes to sensitivity since we are now able to correctly predict 87% of the default loans
# Let's see if we can do even better


### Fifth model: Random Forests algorithm with tuned parameters


# if we type 
modelLookup("rf") 
# we see that the only parameter we can tune in the train function for random forests in the caret package is mtry


## Option 1:

# let's tune the mtry parameter
set.seed(123, sample.kind = "Rounding")
mtry_grid <- expand.grid(mtry = c(15,18,22,25))

train_rf_tuned <- train(default ~ ., credit_main_train, 
                        method = "rf",
                        tuneGrid = mtry_grid)

# this time, the mtry chosen is 15
train_rf_tuned$bestTune

train_rf_tuned

# let's see the performance with the tuned model
predict_rf_tuned <- predict(train_rf_tuned, credit_main_test, type = "raw")

# let's calculate the accuracy and sensitivity of our new model:
accuracy_rf_tuned <- confusionMatrix(predict_rf_tuned, 
                                     credit_main_test$default, 
                                     positive = "yes")$overall["Accuracy"]
sensitivity_rf_tuned <- sensitivity(predict_rf_tuned, 
                                    credit_main_test$default, 
                                    positive = "yes")

# let's add these to our risk_models table for an easy comparison:
risk_models <- rbind(risk_models, list("Random Forests - tuned", accuracy_rf_tuned, sensitivity_rf_tuned))
risk_models %>% kable()


## Option 2

# We saw that in the previous random forest model the mtry used was 15 for the final model with the best accuracy
# Since in the caret package the only parameter that we can tune is mtry, let's use the randomForest function from the randomForest package instead where we can also tune 
# the minimum number of data points in the nodes of the tree. The higher this number the smoother our estimate can be.
# let's use the caret package to optimize this minimum node size

# let's create a function to calculate the accuracy 
node_size <- seq(1,50,10)
set.seed(123, sample.kind = "Rounding")
accuracy_nodes <- sapply(node_size, function(n){
  train(default ~., credit_main_train,
        method = "rf",
        tuneGrid = data.frame(mtry= 15),
        nodesize = node_size)$results$Accuracy
})

qplot(node_size, accuracy_nodes)
# we can see the node size with the highest accuracy
node_size[which.max(accuracy_nodes)]

# let's now apply the optimized node size to the train model:
set.seed(123, sample.kind = "Rounding")
train_rf_optz <- randomForest(default ~., credit_main_train,
                              nodesize = node_size[which.max(accuracy_nodes)])

train_rf_optz

plot(train_rf_optz)

# let's see how well it performs on the test data set
predict_rf_optz <- predict(train_rf_optz, credit_main_test)
accuracy_rf_optz <- confusionMatrix(predict_rf_optz, 
                                    credit_main_test$default, 
                                    positive = "yes")$overall["Accuracy"]
sensitivity_rf_optz <- sensitivity(predict_rf_optz, 
                                   credit_main_test$default, 
                                   positive = "yes")

# Let's add these results to our *risk_models* table: 
risk_models <- rbind(risk_models, list("Random Forests (optimized tuning)", 
                                       accuracy_rf_optz,
                                       sensitivity_rf_optz))

risk_models %>% kable()

# It turns out that the model that provides a high enough accuracy and the highest sensitivity
# is still the random forests algorithm with default parameters so let's apply it to our *credit_final_holdout_test* dataset that we haven't seen yet.
predict_final <- predict(train_rf, credit_final_holdout_test)
accuracy_final <- confusionMatrix(predict_final, 
                                  credit_final_holdout_test$default, 
                                  positive = "yes")$overall["Accuracy"]
accuracy_final

sensitivity_final <- sensitivity(predict_final, 
                                 credit_final_holdout_test$default, 
                                 positive = "yes")
sensitivity_final

# Conclusion and next steps  

# We saw that using the Random Forests algorithm with the default parameters has provided a higher sensitivity rate than any other model, 
# allowing us to accurately predicting 50% of the defaulted loans and offer an overall accuracy of 78%.   

# Unfortunately, the challenge of correctly predicting the default status of a loan based on only 900 examples has proven greater than anticipated. 
# This may be either because our training dataset was not large enough to properly train our algorithms or perhaps this is a truly difficult challenge in real life.  

# The advantage of using the Random Forests algorithm over other black box algorithms like kNN for example 
# consists in its transparency since the results of the model can be formulated in plain language.  

# Next step would be to see if using more sofisticated algorithms would produce higher accuracy and sensitivity 
# even if this means losing the transparency provided by the Random Forests algorithm. 