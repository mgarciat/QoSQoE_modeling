library(plyr)
library(caret);
library(ordinalForest)

#code to perform hyperparameter tuning (section 4.1) with OrdinalRF on video data.

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

        form = "mos ~ cu + bs + ru + lt + cnl + mp + bl + bt + ram + rm + jit + dly + bw";
        cols <- names(train[,1:13]);

        fold.model <- ordfor("mos", train, naive=FALSE, nsets=nsets, ntreeperdiv=ntreeperdiv, ntreefinal=ntreefinal, nbest=nbest);
        fold.predict <- predict(fold.model, newdata=test[,c(cols)]);

        test_labels <- as.integer(test[,"mos"]);
        pred_labels <- as.integer(fold.predict$ypred);
        conf.mat <- table(test_labels,pred_labels);
        
        accs[f] <- sum(diag(conf.mat))/sum(conf.mat);
    }
    return(accs);
}


nruns <- 10;

accs_a <- rep(NA, nruns);
accs_b <- rep(NA, nruns);
accs_c <- rep(NA, nruns);
accs_d <- rep(NA, nruns);


nsets_a <- 100;
nsets_b <- 500;
nsets_c <- 1000;
nsets_d <- 5000;

ntreeperdiv_a <- 100;
ntreeperdiv_b <- 100;
ntreeperdiv_c <- 100;
ntreeperdiv_d <- 100;

ntreefinal_a <- 500;
ntreefinal_b <- 500;
ntreefinal_c <- 500;
ntreefinal_d <- 500;

nbest_a <- 50;
nbest_b <- 50;
nbest_c <- 50;
nbest_d <- 50;

for (r in 1:nruns) {
    message(paste("run #",r,"/",nruns));
    set.seed(r);
    nfolds <- 10;
    data <- read.csv("datasets/mosaic-video.csv",header=TRUE);
    data$mos <- factor(data$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"), ordered=TRUE);
    #data$mos <- as.numeric(data$mos);
    
    aux <- scale(data[,1:13]);
    df <- as.data.frame(aux);
    
    df$mos <- data$mos;
    data <- df;

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

