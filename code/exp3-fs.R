library(ordinalForest)
library(plyr)
library(caret);
library(MASS);
source("allib.r")

data_type <- c("video", "web");
VIDEO_DATA <- 1;
WEB_DATA <- 2;

fsa_type <- c("bf", "fast", "r2ci", "pgss", "mrmr");
BF_FS <- 1;
FAST_FS <- 2;
R2CI_FS <- 3;
PGSS_FS <- 4;
MRMR_FS <- 5;

RPART_CL <- 1
POLR_CL <- 2
ORDINALRF_CL <- 3

cl_label <- c("rpart", "polr", "ordinalRF");
####################
#
# PARAMETERS
#
###################
IDX_CL <- POLR_CL;
IDX_DATA <- VIDEO_DATA;
IDX_FSA <- MRMR_FS;


#fs_info <- file("datasets/ss_video.csv", open="r");
fs_info <- file(paste("datasets/",fsa_type[IDX_FSA], "_", data_type[IDX_DATA], ".csv", sep=""), open="r");
lines <- readLines(fs_info);
close(fs_info);

mtr_accs <- matrix(nrow=10,ncol=10);
arr_feats <- numeric(20);
mtr_nfeats <- matrix(nrow=10,ncol=10);

k <- 0;
row <- 0;
col <- 1;
for (i in 1:length(lines)){
    message(paste("run #",i,"/",length(lines)));
    #remove brackets
    aux <- strsplit(lines[i], "\\[|\\]");

    tr_fname <- gsub(",","",aux[[1]][1]);
    te_fname <- gsub("train","test",tr_fname);
    
    sfeats <- strsplit(aux[[1]][2],",");
    feats <- c();
    for (j in 1:length(sfeats[[1]])) {
        feats <- c(feats,sfeats[[1]][j]);
    }
    feats <- as.numeric(feats);
    feats <- feats + 1;
    for (j in 1:length(feats)) {
        idx <- feats[j];
        arr_feats[idx] <- arr_feats[idx] + 1;
    }
    #training
    training  <- read.csv(paste("datasets/",tr_fname, sep=""), header=TRUE);
    df <- as.data.frame(training);
    df <- df[,feats,drop=FALSE];
    cols <- names(df);
    df$mos <- training$mos;
    training <- df;
    training$mos <- factor(training$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);
    #test
    test  <- read.csv(paste("datasets/", te_fname, sep=""), header=TRUE);
    df <- as.data.frame(test);
    df <- df[,feats,drop=FALSE];
    df$mos <- test$mos;
    test <- df;
    test$mos <- factor(test$mos,levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);

    form_domain = "mos ~ .";
    #message("CLASSIFY:");
    output <- classify_data(training, test, cl=cl_label[IDX_CL], form=form_domain);
    #message("CLASSIFY DONE");
    col <- i %% 10;
    if (col == 1) {
        row <- row + 1;
    } else if (col == 0) {
        col <- 10;
    }
    mtr_accs[row,col] <- output$acc;
    mtr_nfeats[row,col] <- length(feats);
}

#arr_feats
