
require(Hmisc)

###################### Data Analysis to select significant variables ######################

## Read the data file and remove the first column which is non-numeric
## setwd(<Local Woring Directory>)

onp = read.csv("OnlineNewsPopularity.csv")

## Factorize the dependent variables for data analysis

for(i in c(14:19, 32:39)){
  onp[,i] = as.factor(onp[,i])
}

## Cap the variables to 95% and 5% quantiles to remove outliers
for(i in (2:ncol(onp))[-c(13:18, 31:38)]){
  q = quantile(onp[,i], probs = c(0.05,  0.95))
  onp[,i] = ifelse(onp[,i] > q[[2]], q[[2]], onp[,i])
  onp[,i] = ifelse(onp[,i] < q[[1]], q[[1]], onp[,i])
}

## Scale the independent variables
onp$self_reference_min_shares = scale(onp$self_reference_min_shares, center = TRUE, scale = TRUE)
onp$self_reference_avg_sharess = scale(onp$self_reference_avg_sharess, center = TRUE, scale = TRUE)
onp$self_reference_max_shares = scale(onp$self_reference_max_shares, center = TRUE, scale = TRUE)
onp$kw_avg_min = scale(onp$kw_avg_min, center = TRUE, scale = TRUE)
onp$kw_avg_max = scale(onp$kw_avg_max, center = TRUE, scale = TRUE)
onp$kw_avg_avg = scale(onp$kw_avg_avg, center = TRUE, scale = TRUE)

## Remove the first column which is a text field
onp1 = onp[,-1]

## Determine the correlations between variables and significance levels for each
corr.onp = rcorr(as.matrix(onp1), type = c("pearson"))

## Flatten the correlation matrix to get each combination of variables as a single row.
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

corr.flattened = flattenCorrMatrix(corr.onp$r, corr.onp$P)

## Select only the variable pairs that have more than 10% correlation
## among them with significance levels of more than 95%. Remove correlations
## for the dependent variable.

corr.flattened.5pc = subset(corr.flattened, abs(cor) > 0.1 & p < 0.05 & column != "shares")

## Take correlations for the dependent variable in a separate Data Frame
corr.shares = subset(corr.flattened, column == "shares")

## Introduce three new variables to facilitate data analysis
## rcor = correlation of a row variable in the flattened matrix with number of shares
## ccor = correlation of a column variable in the flattened matrix with number of shares
## selection = selected variable among the two based on higher correlation with number of shares
## selectio_Corr = Correlation of the selected variable with the number of shares

corr.flattened.5pc$rcor = as.vector(rep(0, each = nrow(corr.flattened.5pc)))
corr.flattened.5pc$ccor = as.vector(rep(0, each = nrow(corr.flattened.5pc)))
corr.flattened.5pc$selection = as.vector(rep("", each = nrow(corr.flattened.5pc)))
corr.flattened.5pc$selection_Corr = as.vector(rep(0, each = nrow(corr.flattened.5pc)))

## Populate the variables with appropriate values
for(i in 1:nrow(corr.flattened.5pc)){
  if(abs(subset(corr.shares, row == as.character(corr.flattened.5pc$row[i]))$cor) > abs(subset(corr.shares, row == as.character(corr.flattened.5pc$column[i]))$cor)){
    corr.flattened.5pc$selection[i] = as.character(corr.flattened.5pc$row[i])
    corr.flattened.5pc$selection_Corr[i] = abs(subset(corr.shares, row == as.character(corr.flattened.5pc$row[i]))$cor)*100
  }
  else{
    corr.flattened.5pc$selection[i] = as.character(corr.flattened.5pc$column[i])
    corr.flattened.5pc$selection_Corr[i] = abs(subset(corr.shares, row == as.character(corr.flattened.5pc$column[i]))$cor)*100
  }
  corr.flattened.5pc$rcor[i] = subset(corr.shares, row == as.character(corr.flattened.5pc$row[i]))$cor*100
  corr.flattened.5pc$ccor[i] = subset(corr.shares, row == as.character(corr.flattened.5pc$column[i]))$cor*100
  
}

## Convert the correlations to percentages
corr.shares$cor = abs(corr.shares$cor*100)

## Write the data frames to a file
write.csv(corr.shares, "corrshares.csv")
write.csv(corr.flattened.5pc, "corrflat5pc.csv")
