library(rpart)
library(rpartScore)
library(plyr)
library(caret);
source("allib.r")

k<-2;
set.seed(r);
prune <- "mc";
split <- "abs";

data <- load_web_dataset();

data$mos <- as.numeric(data$mos);

folds <- createFolds(factor(data$mos), k = k, list = FALSE);

f <- 1;

idx <- which(folds == f);
train <- data[-idx,];
test <- data[idx,];

form = web_form();

cols <- names(train[,1:15]);
fold.model <- rpartScore(form,data=train,split=split,prune=prune, control=rpart.control(minsplit=10,cp=0.01));
fold.predict <- predict(fold.model, test[,c(cols)],model=TRUE);

conf.mat <- table(test[,"mos"], fold.predict);
