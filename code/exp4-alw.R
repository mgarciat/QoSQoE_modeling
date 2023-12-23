library(activelearning)
library(caret);
library(rpart)
library(rpartScore)
library(ordinalForest)
library(entropy)
library(plyr)
source("allib.r")

#code to perform ordinal classification with active learning (section 4.2/4.2.1) on video data.

TEST_VIDEO <- 1
TEST_WEB <- 2
###############
#
#  PARAMETER 1
#
###############
TEST_DOMAIN <- TEST_WEB;

form_domain <- NULL;
dataset <- NULL;

if(TEST_DOMAIN == TEST_VIDEO) {
    dataset <- load_video_dataset();
    form_domain <- video_form();
} else if (TEST_DOMAIN == TEST_WEB) {
    dataset <- load_web_dataset();
    form_domain <- web_form();
} else {
    stop(paste("Only TEST_VIDEO and TEST_WEB are allowd!"));
}

nruns <- 1;
nfolds <- 10;
nfolds_al <- 5;

accs <- rep(NA, nruns);
size <- rep(NA, nruns);


RPART_CL <- 1
POLR_CL <- 2
ORDINALRF_CL <- 3

cl_label <- c("rpart", "polr", "ordinalRF");
################
#
#  PARAMETER 2
#
###############
idx_cl <- RPART_CL;

set.seed(3872);

for (r in 1:nruns) {
    message(paste("run #",r,"/",nruns));
    accs_k <- rep(NA, nfolds);
    size_k <- rep(NA, nfolds);
    folds <- createFolds(factor(dataset$mos), k = nfolds, list = FALSE);
    for (f in 1:nfolds) {
#    for (f in 2:2) {
        set.seed(f+r);        
        message(paste("     fold #",f,"/",nfolds));
        idx <- which(folds == f);
        train <- dataset[-idx,];
        test <- dataset[idx,];
        
        folds_al <- createFolds(factor(train$mos), k = nfolds_al, list = FALSE);

        output_al <- NULL;
        train_tr_al_max <- NULL;
        acc_max <- 0.;
        for (k in 1:nfolds_al) {
#        for (k in 4:5) {
            message(paste("          validation fold #",k,"/",nfolds_al));
            idx_tr <- which(folds_al == k);
            train_tr <- dataset[-idx_tr,];
            validation <- dataset[idx_tr,];
            #print("                 >>>>>validation:::0");
            output_al <- active_learning_workflow(train_tr, validation, cl=cl_label[idx_cl], pct=0.6, form=form_domain);
            #print("                 >>>>>validation:::1");
            output <- classify_data(output_al$train_tr_al, validation, cl=cl_label[idx_cl],polr.start=output_al$polr.start, form=form_domain);

            ##############################
            #str_train_tr <- gsub(" ", "", paste("test_train_tr_k",toString(k),".txt"))
            #write.csv(train_tr, str_train_tr, row.names=FALSE);
            #str_train_al_tr <- gsub(" ", "", paste("test_train_tr_al_k",toString(k),".txt"))
            #write.csv(output_al$train_tr_al, str_train_al_tr, row.names=FALSE);
            #str_validation <- gsub(" ", "", paste("test_validation_k",toString(k),".txt"))
            #write.csv(validation, str_validation, row.names=FALSE);
            ##############################
            
            if (output$acc > acc_max) {
                acc_max <- output$acc;
                train_tr_al_max <- output_al$train_tr_al;
            }
        }
        output <- classify_data(train_tr_al_max, test, cl=cl_label[idx_cl],polr.start=output_al$polr.start, form=form_domain);
        nr_k <- nrow(train_tr_al_max);
        nr_k_max <- nrow(train);
        accs_k[f] <- output$acc;
        size_k[f] <- nr_k / nr_k_max;
    }
    print(sprintf("Accuracy: %.3f (%.3f) percent", 100*mean(accs_k), 100*sd(accs_k)));
    print(sprintf("Size: %.3f (%.3f) percent", 100*mean(size_k), 100*sd(size_k)));
    accs[r] <- mean(accs_k);
    size[r] <- mean(size_k);
}

#print(sprintf("RPART^abs_mr [minsplit=10,cp=0.1]: %.3f (%.3f) percent", 100*mean(accs), 100*sd(accs)));
print(sprintf("Accuracy: %.3f (%.3f) percent", 100*mean(accs), 100*sd(accs)));
print(sprintf("Size: %.3f (%.3f) percent", 100*mean(size), 100*sd(size)));
