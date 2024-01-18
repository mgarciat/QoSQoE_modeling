library(plyr)
library(caret);
library(ordinalForest)

#code to perform hyperparameter tuning (section 4.1) with OrdinalRF on web data.

cv_ht <- function(dataset, k=5, nsets = 2500, ntreeperdiv = 250, ntreefinal = 2500, nbest = 250) {
    folds <- createFolds(factor(dataset$mos), k = k, list = FALSE);
    accs <- rep(NA, k);
    message(paste("\t\tParameter tuning nsets: ",nsets, ", ntreeperdiv: ", ntreeperdiv, ", ntreefinal: ", ntreefinal, ", nbest: ", nbest));
    for (f in 1:k) {
        idx <- which(folds == f);
        train <- dataset[-idx,];
        test <- dataset[idx,];

        train$mos <- factor(train$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);
        test$mos <- factor(test$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);

        form = "mos ~ cu + ru + mp + bl + cc + lt + ram + rm + jit + cf + dly + bw + cnl + emo + scr";
        cols <- names(train[,1:15]);
        
        fold.model <- ordfor("mos", train, naive=FALSE, nsets=nsets, ntreeperdiv=ntreeperdiv, ntreefinal=ntreefinal, nbest=nbest);
        fold.predict <- predict(fold.model, newdata=test[,c(cols)]);

        test_labels <- as.integer(test[,"mos"]);
        pred_labels <- as.integer(fold.predict$ypred);
        conf.mat <- table(test_labels,pred_labels);
        
        accs[f] <- sum(diag(conf.mat))/sum(conf.mat);
    }
    return(accs);
}


nruns <- 1;

accs_a <- rep(NA, nruns);
accs_b <- rep(NA, nruns);
accs_c <- rep(NA, nruns);
accs_d <- rep(NA, nruns);


nsets_a <- 100;
nsets_b <- 500;
nsets_c <- 1000;
nsets_d <- 5000;

ntreeperdiv_a <- 10;
ntreeperdiv_b <- 10;
ntreeperdiv_c <- 10;
ntreeperdiv_d <- 10;

ntreefinal_a <- 100;
ntreefinal_b <- 100;
ntreefinal_c <- 100;
ntreefinal_d <- 100;

nbest_a <- 10;
nbest_b <- 10;
nbest_c <- 10;
nbest_d <- 10;

for (r in 1:nruns) {
    message(paste("run #",r,"/",nruns));
    set.seed(r);
    nfolds <- 2;
    data <- read.csv("datasets/mosaic-web.csv",header=TRUE);
    data$emo <- as.numeric(unclass(data$emo));
    data$scr <- as.numeric(unclass(data$scr));
    aux <- scale(data[,1:15]);
    df <- as.data.frame(aux);
    
    df$mos <- data$mos;
    data <- df;

    
    data$mos <- factor(data$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"), ordered=TRUE);
    
    folds <- createFolds(factor(data$mos), k = nfolds, list = FALSE);
    
    accs_a_k <- rep(NA, nfolds);
    accs_b_k <- rep(NA, nfolds);
    accs_c_k <- rep(NA, nfolds);
    accs_d_k <- rep(NA, nfolds);
    
    for (f in 1:nfolds) {
        message(paste("\tfold #",f,"/",nfolds));
        idx <- which(folds == f);
        
        train <- data[-idx,];
        #nsets       #ntreeperdiv      #ntreefinal       #nbest      #perffunction
        accs_a_k[f]  <- mean(cv_ht(train, k=5, nsets = nsets_a, ntreeperdiv = ntreeperdiv_a, ntreefinal = ntreefinal_a, nbest = nbest_a));
        accs_b_k[f]  <- mean(cv_ht(train, k=5, nsets = nsets_b, ntreeperdiv = ntreeperdiv_b, ntreefinal = ntreefinal_b, nbest = nbest_b));
        accs_c_k[f]  <- mean(cv_ht(train, k=5, nsets = nsets_c, ntreeperdiv = ntreeperdiv_c, ntreefinal = ntreefinal_c, nbest = nbest_c));
        accs_d_k[f]  <- mean(cv_ht(train, k=5, nsets = nsets_d, ntreeperdiv = ntreeperdiv_d, ntreefinal = ntreefinal_d, nbest = nbest_d));
    }
    accs_a[r] <- mean(accs_a_k);
    accs_b[r] <- mean(accs_b_k);
    accs_c[r] <- mean(accs_c_k);
    accs_d[r] <- mean(accs_d_k);
}

print(sprintf("ORF (nsets: %.0f, ntreeperdiv: %.0f, ntreefinal: %.0f, nbest: %.0f): %.3f (%.3f) pct", nsets_a, ntreeperdiv_a, ntreefinal_a, nbest_a, 100*mean(accs_a), 100*sd(accs_a)));
print(sprintf("ORF (nsets: %.0f, ntreeperdiv: %.0f, ntreefinal: %.0f, nbest: %.0f): %.3f (%.3f) pct", nsets_b, ntreeperdiv_b, ntreefinal_b, nbest_b, 100*mean(accs_b), 100*sd(accs_b)));
print(sprintf("ORF (nsets: %.0f, ntreeperdiv: %.0f, ntreefinal: %.0f, nbest: %.0f): %.3f (%.3f) pct", nsets_c, ntreeperdiv_c, ntreefinal_c, nbest_c, 100*mean(accs_c), 100*sd(accs_c)));
print(sprintf("ORF (nsets: %.0f, ntreeperdiv: %.0f, ntreefinal: %.0f, nbest: %.0f): %.3f (%.3f) pct", nsets_d, ntreeperdiv_d, ntreefinal_d, nbest_d, 100*mean(accs_d), 100*sd(accs_d)));

