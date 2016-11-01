
library(ROCR)
library(randomForest)
library(caret)
library(e1071)
library(pROC)
library(rattle)
library(ModelMetrics)

source("Data Analysis.R")

############################ Data Preparation starts here ############################

## Introduce a new variable "scaledy" to represent the dependent variable
onp$scaledy = scale(onp$shares, center = TRUE, scale = TRUE)

set.seed(1)
    
library(boot)   

## Do a binary classification bases on deciles with 50-50 distribution
onp$classes2 = 0
onp$classes2 = ifelse(onp$shares > 1400, 1,0)

## Classify data into 4 classes with distribution based on deciles with 30-30-20-20 distribution
onp$classes = 1
for (i in 1:nrow(onp)) {
  if(onp[i,"shares"] > 1000 & onp[i,"shares"] <= 1800){
    onp[i,"classes"] = 2
  } else if(onp[i,"shares"] > 1800 & onp[i,"shares"] <= 3400){
    onp[i,"classes"] = 3
  } else if(onp[i,"shares"] > 3400){
    onp[i,"classes"] = 4
  }
}

#Factorize dependent variables
onp$classes = as.factor(onp$classes)
onp$classes2 = as.factor(onp$classes2)

## Split data into training and test sets
library(caTools)
splitONP = sample.split(onp, SplitRatio = 0.75)
TrainONP = subset(onp, splitONP == TRUE)
TestONP = subset(onp, splitONP == FALSE)

###################### Modelling and performance evaluation starts here ######################
color.lr = '#efab69'
color.cart = '#ab69ef'
color.rf = '#adef69'

model1.lr = lm(scaledy ~ kw_avg_avg+
                 self_reference_avg_sharess+
                 LDA_03+
                 data_channel_is_world+
                 num_imgs+
                 global_subjectivity+
                 num_hrefs+
                 kw_min_avg+
                 is_weekend+
                 kw_min_max+
                 kw_max_min+
                 num_videos+
                 data_channel_is_socmed+
                 title_subjectivity+
                 data_channel_is_entertainment+
                 avg_positive_polarity+
                 average_token_length+
                 data_channel_is_lifestyle+
                 LDA_01+
                 LDA_02+
                 LDA_04+
                 data_channel_is_bus, data = TrainONP)

## Determine residuals are independent by testing for serial correlation
acf(model1.lr$residuals, type = "correlation", plot = TRUE)
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")


## Construct a binomial logistic regression model
PredONP.logit = glm(classes2 ~ kw_avg_avg+
                      self_reference_avg_sharess+
                      LDA_03+
                      data_channel_is_world+
                      num_imgs+
                      global_subjectivity+
                      num_hrefs+
                      kw_min_avg+
                      is_weekend+
                      kw_min_max+
                      kw_max_min+
                      num_videos+
                      data_channel_is_socmed+
                      title_subjectivity+
                      data_channel_is_entertainment+
                      avg_positive_polarity+
                      average_token_length+
                      data_channel_is_lifestyle+
                      LDA_01+
                      LDA_02+
                      data_channel_is_bus,  data = TrainONP, family = binomial)

## Predict probabilities that a news item falls in one class of shares or the other
predictONP.logit = predict(PredONP.logit, type = "response", newdata = TestONP)
ROCRpred = prediction(predictONP.logit, TestONP$classes2)
predrocr.logit = roc(TestONP$classes2, predictONP.logit)

## Determine AUC and plot the preformance for the logit model
confusionM.logit = table(TestONP$classes2, predictONP.logit > 0.5)
print(paste("Accuracy for Logit=", round(sum(diag(confusionM.logit))/sum(confusionM.logit),4)*100, "%"))
print(paste("AUC for Logit=", round(performance(ROCRpred, "auc")@y.values[[1]], digits = 4)*100, "%"))
RP.perf.logit <- performance(ROCRpred, "tpr", "fpr")
plot (RP.perf.logit, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")
plot(predrocr.logit, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE, 
     auc.polygon.col=color.lr, print.thres="best", print.thres.best.method="closest.topleft", print.thres.best.weights=c(5, 0.2), colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))


## Train a CART model using cross-validation to get the complexity parameter
fitControl = trainControl(method = "repeatedcv", repeats = 4)
cartGrid = expand.grid(.cp = (1:50)*0.01)
train(classes2 ~ kw_avg_avg+
        self_reference_avg_sharess+
        LDA_03+
        data_channel_is_world+
        num_imgs+
        global_subjectivity+
        num_hrefs+
        kw_min_avg+
        is_weekend+
        kw_min_max+
        kw_max_min+
        num_videos+
        data_channel_is_socmed+
        title_subjectivity+
        data_channel_is_entertainment+
        avg_positive_polarity+
        average_token_length+
        data_channel_is_lifestyle+
        LDA_01+
        LDA_02+
        data_channel_is_bus, data = TrainONP, method = "rpart", trControl = fitControl, tuneGrid = cartGrid)

## Build a CART binary classification model using cp obtained from above
predONP.CART = rpart(classes2 ~ kw_avg_avg+
                       self_reference_avg_sharess+
                       LDA_03+
                       data_channel_is_world+
                       num_imgs+
                       global_subjectivity+
                       num_hrefs+
                       kw_min_avg+
                       is_weekend+
                       kw_min_max+
                       kw_max_min+
                       num_videos+
                       data_channel_is_socmed+
                       title_subjectivity+
                       data_channel_is_entertainment+
                       avg_positive_polarity+
                       average_token_length+
                       data_channel_is_lifestyle+
                       LDA_01+
                       LDA_02+
                       data_channel_is_bus, method = "class", data = TrainONP, control = rpart.control(cp=0.01))

## Predict probabilities and get the AUC for CART model
predictONP.prob = predict(predONP.CART, newdata = TestONP, type = "prob")
predictONP.class = predict(predONP.CART, newdata = TestONP, type = "class")
confusionM.cart = table(TestONP$classes2, predictONP.class)
pred.class = prediction(predictONP.prob[,2], TestONP$classes2)
predrocr.CART = roc(TestONP$classes2, predictONP.prob[,2])

## Determine AUC and plot the preformance for CART model
print(paste("Accuracy for CART =", round(sum(diag(confusionM.cart))/sum(confusionM.cart),4)*100, "%"))
print(paste("AUC for CART =", round(performance(pred.class, "auc")@y.values[[1]], digits = 4)*100, "%"))
RP.perf.cart <- performance(pred.class, "tpr", "fpr")
plot (RP.perf.cart, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")
confusionMatrix(predictONP.class, TestONP$classes2)
plot(predrocr.CART, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE, 
     auc.polygon.col=color.cart, print.thres="best", print.thres.best.method="closest.topleft",print.thres.best.weights=c(5, 0.2), colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))

## Build a Random Forest binary classification model
predONP.rforest = randomForest(classes2 ~ kw_avg_avg+
                                 self_reference_avg_sharess+
                                 LDA_03+
                                 data_channel_is_world+
                                 num_imgs+
                                 global_subjectivity+
                                 num_hrefs+
                                 kw_min_avg+
                                 is_weekend+
                                 kw_min_max+
                                 kw_max_min+
                                 num_videos+
                                 data_channel_is_socmed+
                                 title_subjectivity+
                                 data_channel_is_entertainment+
                                 avg_positive_polarity+
                                 average_token_length+
                                 data_channel_is_lifestyle+
                                 LDA_01+
                                 LDA_02+
                                 data_channel_is_bus, data = TrainONP, nodesize = 25, ntree = 500)

## Check the model performance for the ntree parameter
plot(predONP.rforest)
legend("topright", colnames(predONP.rforest$err.rate),col=1:4,cex=1.2,fill=1:4)
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")

## Predict probabilities and get the AUC
PredictONP.forest = predict(predONP.rforest, newdata = TestONP, type = "prob")
predictONP.forsest.2class = predict(predONP.rforest, newdata = TestONP, type = "class")
predrocr = prediction(PredictONP.forest[,2], TestONP$classes2)
confusionM.RF = table(predictONP.forsest.2class, TestONP$classes2)
predrocr.RF = roc(TestONP$classes2, PredictONP.forest[,2])


## Determine AUC and plot the preformance for Binary RF
print(paste("Accuracy for Binary RF =", round(sum(diag(confusionM.RF))/sum(confusionM.RF),4)*100, "%"))
print(paste("AUC for Binary RF =", round(performance(predrocr, "auc")@y.values[[1]], digits = 4)*100, "%"))
RP.perf.rf <- performance(predrocr, "tpr", "fpr")
plot (RP.perf.rf, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")
## Confusion Matrix for Binary RF
confusionMatrix(predictONP.forsest.2class, TestONP$classes2)
plot(predrocr.RF, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE, 
auc.polygon.col=color.rf, print.thres="best", print.thres.best.method="youden",print.thres.best.weights=c(5, 0.2), colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))


## Build a Random Forest classification model with 4 classes
predONP.rforest.4class = randomForest(classes ~ kw_avg_avg+
                                        self_reference_avg_sharess+
                                        LDA_03+
                                        data_channel_is_world+
                                        num_imgs+
                                        global_subjectivity+
                                        num_hrefs+
                                        kw_min_avg+
                                        is_weekend+
                                        kw_min_max+
                                        kw_max_min+
                                        num_videos+
                                        data_channel_is_socmed+
                                        title_subjectivity+
                                        data_channel_is_entertainment+
                                        avg_positive_polarity+
                                        average_token_length+
                                        data_channel_is_lifestyle+
                                        LDA_01+
                                        LDA_02+
                                        data_channel_is_bus, data = TrainONP, nodesize = 30, ntree = 260, mtry = 4)

## Check the model performance for the ntree parameter
plot(predONP.rforest.4class)
legend("topleft", colnames(predONP.rforest.4class$err.rate),col=1:4,cex=0.8,fill=1:4)
grid(nx = 10, ny = 10L, col = "lightgray", lty = "dotted")

## Make predictions for 4-Class Random Forest
PredictONP.forest.4class = predict(predONP.rforest.4class, newdata = TestONP, type = "class")
PredictONP.forest.4class.prob = as.data.frame(predict(predONP.rforest.4class, newdata = TestONP, type = "prob"))
Predicted.4class = PredictONP.forest.4class.prob/rowSums(PredictONP.forest.4class.prob)

## Check the confusion Matrix, calculate the accuracy using Exact Match Ratio, and Multiclass AUC
confusion.matrix = table(TestONP$classes, PredictONP.forest.4class)
confusion.matrix
print(paste("Accuracy for 4-class RF =", round(sum(diag(confusion.matrix))/sum(confusion.matrix),4)*100, "%"))
mauc(TestONP$classes, Predicted.4class)


