
library(ROCR)
library(randomForest)
library(caret)
library(e1071)

############################ Data Preparation starts here ############################

onp = read.csv("OnlineNewsPopularity.csv")


## Cap the variables to 95% and 5% quantiles to remove outliers
for(i in 2:ncol(onp)){
  q = quantile(onp[,i], probs = c(0.05,  0.95))
  onp[,i] = ifelse(onp[,i] > q[[2]], q[[2]], onp[,i])
  onp[,i] = ifelse(onp[,i] < q[[1]], q[[1]], onp[,i])
}

## Scale the variables of interest. 
## Introduce a new variable "scaledy" to represent the dependent variable
onp$scaledy = scale(onp$shares, center = TRUE, scale = TRUE)
onp$self_reference_min_shares = scale(onp$self_reference_min_shares)
onp$self_reference_avg_sharess = scale(onp$self_reference_avg_sharess)
onp$self_reference_max_shares = scale(onp$self_reference_max_shares)
onp$kw_avg_min = scale(onp$kw_avg_min)
onp$kw_avg_max = scale(onp$kw_avg_max)
onp$kw_avg_avg = scale(onp$kw_avg_avg)
   
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


## Factorize the dependent variables for classification
onp$classes = as.factor(onp$classes)
onp$classes2 = as.factor(onp$classes2)

## Factorize the independent variables wherever applicable
onp$data_channel_is_bus = as.factor(onp$data_channel_is_bus)
onp$data_channel_is_entertainment = as.factor(onp$data_channel_is_entertainment)
onp$is_weekend = as.factor(onp$is_weekend)

## Split data into training and test sets
library(caTools)
splitONP = sample.split(onp, SplitRatio = 0.7)
TrainONP = subset(onp, split = TRUE)
TestONP = subset(onp, split = FALSE)

###################### Modelling and performance evaluation starts here ######################

model1.lr = lm(scaledy ~ average_token_length + 
     avg_positive_polarity + 
     factor(data_channel_is_bus) + 
     factor(data_channel_is_entertainment) + 
     global_subjectivity + 
     factor(is_weekend) + 
     kw_avg_avg + 
     kw_avg_max + 
     kw_avg_min + 
     LDA_02 + 
     LDA_03 + 
     LDA_04 + 
     num_hrefs + 
     num_imgs +  
     num_videos + 
     self_reference_avg_sharess + 
     title_sentiment_polarity + 
     title_subjectivity, data = TrainONP)

## Determine residuals are independent by testing for serial correlation
acf(model1.lr$residuals, type = "correlation", plot = TRUE)

## Construct a binomial logistic regression model
PredONP.logit = glm(classes2 ~ average_token_length + 
                  avg_positive_polarity + 
                  factor(data_channel_is_bus) + 
                  factor(data_channel_is_entertainment) + 
                  global_subjectivity + 
                  factor(is_weekend) + 
                  kw_avg_avg + 
                  kw_avg_max + 
                  kw_avg_min + 
                  LDA_02 + 
                  LDA_03 + 
                  LDA_04 + 
                  num_hrefs + 
                  num_imgs +  
                  num_videos + 
                  self_reference_avg_sharess + 
                  title_sentiment_polarity + 
                  title_subjectivity,  data = TrainONP, family = binomial)

## Predict probabilities that a news item falls in one class of shares or the other
predictONP.logit = predict(PredONP.logit, type = "response", newdata = TestONP)
ROCRpred = prediction(predictONP.logit, TestONP$classes2)

## Determine AUC and plot the preformance for the logit model
as.numeric(performance(ROCRpred, "auc") @y.values)
RP.perf.logit <- performance(ROCRpred, "tpr", "fpr")
plot (RP.perf.logit, colorize=TRUE)


## Train a CART model using cross-validation to get the complexity parameter
fitControl = trainControl(method = "repeatedcv", repeats = 4)
cartGrid = expand.grid(.cp = (1:50)*0.01)
train(classes2 ~ average_token_length + 
          avg_positive_polarity + 
          factor(data_channel_is_bus) + 
          factor(data_channel_is_entertainment) + 
          global_subjectivity + 
          factor(is_weekend) + 
          kw_avg_avg + 
          kw_avg_max + 
          kw_avg_min + 
          LDA_02 + 
          LDA_03 + 
          LDA_04 + 
          num_hrefs + 
          num_imgs +  
          num_videos + 
          self_reference_avg_sharess + 
          title_sentiment_polarity + 
          title_subjectivity, data = TrainONP, method = "rpart", trControl = fitControl, tuneGrid = cartGrid)

## Build a CART binary classification model using cp obtained from above
predONP.CART = rpart(classes2 ~ average_token_length + 
               avg_positive_polarity + 
               data_channel_is_bus + 
               data_channel_is_entertainment + 
               global_subjectivity + 
               is_weekend + 
               kw_avg_avg + 
               kw_avg_max + 
               kw_avg_min + 
               LDA_02 + 
               LDA_03 + 
               LDA_04 + 
               num_hrefs + 
               num_imgs +  
               num_videos + 
               self_reference_avg_sharess + 
               title_sentiment_polarity + 
               title_subjectivity, method = "class", data = TrainONP, control = rpart.control(cp=0.01))

## Predict probabilities and get the AUC for CART model
predictONP.prob = predict(predONP.CART, newdata = TestONP, type = "prob")
predictONP.class = predict(predONP.CART, newdata = TestONP, type = "class")
table(TestONP$classes2, predictONP.class)
pred.class = prediction(predictONP.prob[,2], TestONP$classes2)
print(paste("AUC =", round(performance(pred.class, "auc")@y.values[[1]], digits = 4)*100, "%"))

## Determine AUC and plot the preformance for CART model
RP.perf.cart <- performance(pred.class, "tpr", "fpr")
plot (RP.perf.cart, colorize=TRUE)

## Build a Random Forest binary classification model
predONP.rforest = randomForest(classes2 ~ average_token_length + 
      avg_positive_polarity + 
      data_channel_is_bus + 
      data_channel_is_entertainment + 
      global_subjectivity + 
      is_weekend + 
      kw_avg_avg + 
      kw_avg_max + 
      kw_avg_min + 
      LDA_02 +  
      LDA_03 + 
      LDA_04 + 
      num_hrefs + 
      num_imgs +  
      num_videos + 
      self_reference_avg_sharess + 
      title_sentiment_polarity + 
      title_subjectivity, data = TrainONP, nodesize = 25, ntree = 200)


## Predict probabilities and get the AUC
PredictONP.forest = predict(predONP.rforest, newdata = TestONP, type = "prob")
predrocr = prediction(PredictONP.forest[,2], TestONP$classes2)
print(paste("AUC =", round(performance(predrocr, "auc")@y.values[[1]], digits = 4)*100, "%"))

## Determine AUC and plot the preformance
RP.perf.rf <- performance(predrocr, "tpr", "fpr")
plot (RP.perf.rf, colorize=TRUE)

## Build a Random Forest classification model with 4 classes
predONP.rforest.4class = randomForest(classes ~ average_token_length + 
    avg_positive_polarity + 
    data_channel_is_bus + 
    data_channel_is_entertainment + 
    global_subjectivity + 
    is_weekend + 
    kw_avg_avg + 
    kw_avg_max + 
    kw_avg_min + 
    LDA_02 +  
    LDA_03 + 
    LDA_04 + 
    num_hrefs + 
    num_imgs +  
    num_videos + 
    self_reference_avg_sharess + 
    title_sentiment_polarity + 
    title_subjectivity, data = TrainONP, nodesize = 25, ntree = 200)

## Make predictions with test data and output the accuracy using Exact Match Ratio
PredictONP.forest.4class = predict(predONP.rforest.4class, newdata = TestONP, type = "class")
confusion.matrix = table(TestONP$classes, PredictONP.forest.4class)
print(paste("Accuracy =", round(sum(diag(confusion.matrix))/sum(confusion.matrix),4)*100, "%"))

    