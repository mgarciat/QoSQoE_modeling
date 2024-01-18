library(rpart)
library(rpartScore)
library(plyr)
library(caret);
source("allib.r")

#code to perform hyperparameter tuning (section 4.1) with RPART on web data.

cv_ht <- function(dataset, k=5, split="quad", prune="mc", minsplit=10, cp=0.01) {
    folds <- createFolds(factor(dataset$mos), k = k, list = FALSE);
    accs <- rep(NA, k);
    message(paste("\t\tParameter tuning split: ",split, ", prune: ", prune));
    for (f in 1:k) {
        idx <- which(folds == f);
        train <- dataset[-idx,];
        test <- dataset[idx,];

        form = "mos ~ cu + ru + mp + bl + cc + lt + ram + rm + jit + cf + dly + bw + cnl + emo + scr";
        cols <- names(train[,1:15]);

        fold.model <- rpartScore(form,data=train,split=split,prune=prune, control=rpart.control(minsplit=10,cp=0.01));
        fold.predict <- predict(fold.model, test[,c(cols)],model=TRUE);
        conf.mat <- table(test[,"mos"], fold.predict);
        accs[f] <- sum(diag(conf.mat))/sum(conf.mat);
    }
    return(accs);
}


nruns <- 10;

accs_abs_mc  <- rep(NA, nruns);
accs_quad_mc <- rep(NA, nruns);
accs_abs_mr  <- rep(NA, nruns);
accs_quad_mr <- rep(NA, nruns);

rerrs <- rep(NA, nruns);
raccs <- rep(NA, nruns);

for (r in 1:nruns) {
    message(paste("run #",r,"/",nruns));
    set.seed(r);
    nfolds <- 10;
    data <- load_web_dataset();
    data$mos <- as.numeric(data$mos);
    
    #data <- read.csv("datasets/mosaic-web.csv",header=TRUE);
    
    #data$mos <- as.numeric(factor(data$mos, levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE));
    #data$scr <- as.numeric(factor(data$scr, levels=c("320x240", "480x320", "800x480", "854x480", "960x540", "1280x720", "1280x768", "1920x1080"), ordered=TRUE));
    #data$emo <- as.numeric(factor(data$emo, levels=c("Alegre", "Indiferente", "Triste", "Enojado"), ordered=TRUE));
    
    #aux <- scale(data[,1:15]);
    #df <- as.data.frame(aux);
    
    #df$mos <- data$mos;
    #data <- df;
    #message("\tcreate folds");
    folds <- createFolds(factor(data$mos), k = nfolds, list = FALSE);
    #message("\temd of create folds");
    
    accs_abs_mc_k<- rep(NA, nfolds);
    accs_quad_mc_k<- rep(NA, nfolds);
    accs_abs_mr_k<- rep(NA, nfolds);
    accs_quad_mr_k<- rep(NA, nfolds);
    
    for (f in 1:nfolds) {
        message(paste("\tfold #",f,"/",nfolds));
        idx <- which(folds == f);
        
        train <- data[-idx,];
        
        accs_abs_mc_k[f]  <- mean(cv_ht(train, k=5, split="abs", prune="mc", minsplit=15, cp=0.1));
        accs_quad_mc_k[f] <- mean(cv_ht(train, k=5, split="quad", prune="mc", minsplit=15, cp=0.1));
        accs_abs_mr_k[f]  <- mean(cv_ht(train, k=5, split="abs", prune="mr", minsplit=15, cp=0.1));
        accs_quad_mr_k[f] <- mean(cv_ht(train, k=5, split="quad", prune="mr", minsplit=15, cp=0.1));
    }
    accs_abs_mc[r]  <- mean(accs_abs_mc_k);
    accs_quad_mc[r] <- mean(accs_quad_mc_k);
    accs_abs_mr[r]  <- mean(accs_abs_mr_k);
    accs_quad_mr[r] <- mean(accs_quad_mr_k);
}

print(sprintf("RPART^abs_mc [minsplit=15,cp=0.1]: %.3f (%.3f) percent", 100*mean(accs_abs_mc), 100*sd(accs_abs_mc)));
print(sprintf("RPART^quad_mc [minsplit=15,cp=0.1]: %.3f (%.3f) percent", 100*mean(accs_quad_mc), 100*sd(accs_quad_mc)));
print(sprintf("RPART^abs_mr [minsplit=15,cp=0.1]: %.3f (%.3f) percent", 100*mean(accs_abs_mr), 100*sd(accs_abs_mr)));
print(sprintf("RPART^quad_mr [minsplit=15,cp=0.1]: %.3f (%.3f) percent", 100*mean(accs_quad_mr), 100*sd(accs_quad_mr)));

