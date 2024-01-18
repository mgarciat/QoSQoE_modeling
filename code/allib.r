library(caret)
library(rpart)
library(rpartScore)
library(ordinalForest)
library(entropy)

#####COMCOM ACTIVE LEARNING LIBRARY
#' Load the video dataset
load_video_dataset <- function(path="datasets/mosaic-video.csv") {
    data <- read.csv("datasets/mosaic-video.csv",header=TRUE);
    data$mos <- factor(data$mos, levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);
    aux <- scale(data[,1:13]);
    df <- as.data.frame(aux);
    
    df$mos <- data$mos;
    dataset <- df;
    return (dataset);
}
#' Load the web dataset
load_web_dataset <- function(path="datasets/mosaic-web.csv") {
    data <- read.csv("datasets/mosaic-web.csv",header=TRUE);
    data$mos <- factor(data$mos, levels=c("Bad","Poor","Good","VeryGood","Excellent"),ordered=TRUE);
    data$scr <- as.numeric(factor(data$scr, levels=c("320x240", "480x320", "800x480", "854x480", "960x540", "1280x720", "1280x768", "1920x1080"), ordered=TRUE));
    data$emo <- as.numeric(factor(data$emo, levels=c("Alegre", "Indiferente", "Triste", "Enojado"), ordered=TRUE));
    
    aux <- scale(data[,1:15]);
    df <- as.data.frame(aux);
    
    df$mos <- data$mos;
    dataset <- df;
    return (dataset);
}
#' Generate video form for classification
#'
#' @return 
video_form <- function() {
    form = "mos ~ cu + bs + ru + lt + cnl + mp + bl + bt + ram + rm + jit + dly + bw";
    return (form);
}
#' Generate web form for classification
#'
#' @return
web_form <- function() {
    form = "mos ~ cu + ru + mp + bl + cc + lt + ram + rm + jit + cf + dly + bw + cnl + emo + scr";
    return (form);
}
#' Assigns labels
#'
#' This function
#' 
#' @param
#' @param
#' @param
#' @return
assign_labels <- function(y_u, y, lst) {
    npoints <- length(lst);
    ny <- length(y);
    yy_u <- y_u;
    
    for (i in 1:npoints) {
        x <- lst[i];
        k <- 0;
        for (j in 1:ny) {
            if (is.na(y_u[j])) {
                k <- k + 1;
            }
            if (x == k) {
                yy_u[j] <- y[j];
                break;
            }
        }
    }
    return(yy_u);
}

#' Train data
#' This function trains the classifier and returns the model
#'
#' @param trainX the training data
#' @param trainY the class label
#' @param cl the label of the classifier
#' @param polr.start a starting point to POLR classifier to avoid no convergence
#' @param form the list of features formatted to perform classification
#' @return the classification model
model_data <- function(trainX, trainY, cl='rpart', polr.start=NULL, form=form) {
    classifier <- NA;
    acc <- NA;
    model <- NULL;

    train <- as.data.frame(trainX);
    train$mos <- trainY;
    
    if (cl == 'rpart') {
        train$mos <- as.numeric(train$mos);
        model <- rpart(form, data=train, method="class", control=rpart.control(minsplit=10,cp=0.01));
    } else if (cl == 'ordinalRF') {
        model <- ordfor("mos", train, naive=FALSE, nsets=5000, ntreeperdiv=10, ntreefinal=5000, nbest=10, perffunction="probability");
    } else if (cl == 'polr') {
        if (is.null(polr.start)) {
            model <- polr(form, data = train, Hess=FALSE, method="logistic");
        } else {
            model <- polr(form, data = train, Hess=FALSE, method="logistic", start=polr.start);
        }
    } else {
        stop(paste("Classsifier ", cl," not avilable. List of classifiers: rpart, ordinalRF and polr"));
    }
    return (model);
}
#' Classify data
#' This function trains the classifier on training data and applies the model to test data
#'
#' @param train the training data
#' @param test the test data
#' @param cl the label of the classifier
#' @param polr.start a starting point to POLR classifier to avoid no convergence
#' @param form the list of features formatted to perform classification
#' @return the accuracy and the zeta coeeficient as starting point for POLR classifier
classify_data <- function(train, test, cl='rpart', polr.start=NULL, form=form) {
    classifier <- NA;
    acc <- NA;
    output <- NULL;
    #cols <- names(train[,1:ncol(train)-1]);
    cols_aux <- names(train);
    cols <- cols_aux[1:ncol(train)-1];
    
    #message(paste("  1:", cols));
    if (cl == 'rpart') {
        train$mos <- as.numeric(train$mos);
        test$mos <- as.numeric(test$mos);
        #message("  2");
        
        model <- rpartScore(form, data=train,split="abs",prune="mr", control=rpart.control(minsplit=10,cp=0.01));
        #message("  3");

        predict <- predict(model, test[,c(cols),drop=FALSE],model=TRUE);
        #conf.mat <- table(test[,"mos"], predict);
        #acc <- sum(diag(conf.mat))/sum(conf.mat);
        #message("  4");

        nvals <- length(predict);
        total <- 0;
        for (i in 1:nvals) {
            if (predict[i] == test[i,"mos"]) {
                total <- total + 1;
            }
        }
        acc <- total / nvals;
        
        output <- list(acc=acc,polr.start=NULL);
    } else if (cl == 'ordinalRF') {        
        model <- ordfor("mos", train, naive=FALSE, nsets=5000, ntreeperdiv=10, ntreefinal=5000, nbest=10);
        predict <- predict(model, newdata=test[,c(cols),drop=FALSE]);
        
        test_labels <- as.integer(test[,"mos"]);
        pred_labels <- as.integer(predict$ypred);
        #conf.mat <- table(test_labels,pred_labels);
        #acc <- sum(diag(conf.mat))/sum(conf.mat);
        nvals <- length(pred_labels);
        total <- 0;
        for (i in 1:nvals) {
            if (pred_labels[i] == test_labels[i]) {
                total <- total + 1;
            }
        }
        acc <- total / nvals;
        
        output <- list(acc=acc,polr.start=NULL);
    } else if (cl == 'polr') {
        if (is.null(polr.start)) {
            model <- polr(form, data = train, Hess=FALSE, method="logistic");
        } else {
            model <- polr(form, data = train, Hess=FALSE, method="logistic", start=polr.start);
        }
        predict <- predict(model, newdata=test[,c(cols),drop=FALSE]);
        
        test_labels <- as.integer(test[,"mos"]);
        pred_labels <- as.integer(predict);
        #conf.mat <- table(test_labels,pred_labels);
        #acc <- sum(diag(conf.mat))/sum(conf.mat);

        nvals <- length(pred_labels);
        total <- 0;
        for (i in 1:nvals) {
            if (pred_labels[i] == test_labels[i]) {
                total <- total + 1;
            }
        }
        acc <- total / nvals;


        start <- c(model$coefficients,model$zeta);
        
        output <- list(acc=acc,polr.start=start);
    } else {
        stop(paste("Classsifier ", cl," not avilable. List of classifiers: rpart, ordinalRF and polr"));
    }
    return (output);
}
#' Active Learning with Uncertainty Sampling
#'
#' The 'uncertainty sampling' approach to active learning determines the
#' unlabeled observation which the user-specified supervised classifier is
#' "least certain." The "least certain" observation should then be queried by
#' the oracle in the "active learning" framework.
#'
#' The least certainty term is quite general, but we have implemented three of
#' the most widely used methods:
#'
#' \describe{
#' \item{entropy}{query the unlabeled observation maximizing posterior
#' probabilities of each class under the trained classifier}
#' \item{least_confidence}{query the unlabeled observation with the least
#' posterior probability under the trained classifier}
#' \item{margin}{query the unlabeled observation that minimizes the difference in
#' the largest two posterior probabilities under the trained classifier}
#' }
#'
#' The \code{uncertainty} argument must be one of the three: \code{entropy} is
#' the default. Note that the three methods are equivalent (they yield the same
#' observation to be queried) with binary classification.
#'
#' We require a user-specified supervised classifier from the
#' \code{\link{caret}} R package. Furthermore, we assume that the classifier
#' returns posterior probabilities of class membership; otherwise, an error is
#' thrown. To obtain a list of valid classifiers, see the \code{\link{caret}}
#' vignettes, which are available on CRAN. Also, see the
#' \code{\link[caret]{modelLookup}}.
#'
#' Additional arguments to the specified \code{\link{caret}} classifier can be
#' passed via \code{...}.
#'
#' Unlabeled observations in \code{y} are assumed to have \code{NA} for a label.
#'
#' It is often convenient to query unlabeled observations in batch. By default,
#' we query the unlabeled observations with the largest uncertainty measure
#' value. With the \code{num_query} the user can specify the number of
#' observations to return in batch. If there are ties in the uncertainty
#' measure values, they are broken by the order in which the unlabeled
#' observations are given.
#'
#' @param x a matrix containing the labeled and unlabeled data
#' @param y a vector of the labels for each observation in x. Use NA for
#'     unlabeled.
#' @param uncertainty a string that contains the uncertainty measure. See above
#'     for details.
#' @param classifier a string that contains the supervised classifier as given
#'     in the \code{\link{caret}} package.
#' @param num_query the number of observations to be queried.
#' @param ... additional arguments that are sent to the \code{\link{caret}}
#'     classifier.
#' @return a list that contains the least_certain observation and miscellaneous
#' results. See above for details.
#' @export
#' @examples
#' x <- iris[, -5]
#' y <- iris[, 5]
#'
#' # For demonstration, suppose that few observations are labeled in 'y'.
#' y <- replace(y, -c(1:10, 51:60, 101:110), NA)
#'
#' uncertainty_sampling(x=x, y=y, classifier="lda")
#' uncertainty_sampling(x=x, y=y, uncertainty="entropy",
#'                     classifier="qda", num_query=5)
unc_sampling <- function(x, y, uncertainty="entropy", classifier, num_query=1, polr.start, form) {
  # Validates the classifier string.
    validate_classifier(classifier, posterior_prob=TRUE)
    x <- as.matrix(x)
    split_out <- split_labeled(x, y);

    train_out <- model_data(trainX=split_out$x_labeled, trainY=split_out$y_labeled, cl=classifier, polr.start=polr.start,form=form)
  
  # Extracts the class posterior probabilities for the unlabeled observations.
    posterior <- predict(train_out, newdata=as.data.frame(split_out$x_unlabeled), type="prob")

    if (classifier == "ordinalRF") {
        posterior = as.matrix(posterior$classprobs);
    } else if (classifier == "polr" || classifier == "rpart") {
        posterior <- as.matrix(posterior)
    } else {
        print("classifier available (classifier): ordinalRF or polr");
    }

  # Computes the specified uncertainty for each of the unlabeled observations
  # based on the posterior probabilities of class membership.
  obs_uncertainty <- switch(uncertainty,
                            entropy=entropy_uncertainty(posterior),
                            least_confidence=least_confidence(posterior),
                            margin=margin_uncertainty(posterior)
                            )

  # Determines the order of the unlabeled observations by uncertainty measure.
    query <- order(obs_uncertainty, decreasing=T)[seq_len(num_query)]
    
  list(query=query, posterior=posterior, uncertainty=obs_uncertainty)
}
#' Run active learning workflow
#'
#' 
active_learning_workflow <- function(train, validation, pct = 0.25, nq = 5, cl = "rpart", conv=3, form) {
    #start active learning#
    #######################
    ntrain <- nrow(train);
    ntrain_labels <- round(ntrain * pct);
    idx_l <- sort(sample(1:ntrain, ntrain_labels));
    ori_output <- classify_data(train, validation, cl, form=form);
    
    x <- train[, 1:ncol(train)-1];
    y <- train[, ncol(train)];
    y_u <- replace(y, -idx_l, NA);
    
    #model accuracy
    idx_na <- is.na(y_u);
    train_v <- train[!idx_na,];
    output <- classify_data(train_v, validation, cl, polr.start=ori_output$polr.start, form=form);
    best_acc <- output$acc; 
    stop <- FALSE;
    r <- 1;
    cit <- 0;
    y_u_best <- y_u;

    while (!stop) {        
#        print(paste("       It: ", r));
      #check uncertainty and selet 5 points
        us <- unc_sampling(x=x, y=y_u, uncertainty="entropy", classifier=cl, num_query=nq,  polr.start = ori_output$polr.start, form=form);
        yy_u <- assign_labels(y_u, y, us$query);
        #model accuracy
        idx_na <- is.na(yy_u);
        train_v <- train[!idx_na,];
        
        current_output <- classify_data(train_v, validation, cl, ori_output$polr.start, form=form);
        current_acc <- current_output$acc; 
        print(paste("            It#",r, ", #rows: ", nrow(train_v),", acc=", format((current_acc*100), digits=4) ));
        #print(paste("            train_v: ", toString(summary(train_v$mos))))
        #print(paste("            val: ", toString(summary(validation$mos))));
        #print(paste("    current acc=",toString(current_acc)));
        if (current_acc > best_acc) {
            print("              [*]");
            best_acc <- current_acc;
            y_u <- yy_u;
            y_u_best <- y_u;
            cit <- 0;
            if (sum(is.na(y_u)) == 0) {
                print("              [stop allin]");
                stop <- TRUE;
            }
        } else if (cit == conv){
#        } else if (cit >= conv && current_acc > 50){
            print("              [stop]");
            stop <- TRUE;
        } else {
            y_u <- yy_u;
            cit <- cit + 1;
        }
        r <- r + 1;
    }
    idx_na <- is.na(y_u_best);
    train_f <- train[!idx_na,];
    output <- list(train_tr_al=train_f, polr.start=ori_output$polr.start);
    
    return(output);
}



#####ORACLE
#' Automatic query of an oracle.
#'
#' This function reports the true classification for a set of observations.
#'
#' @export
#' @param i a vector of the queried observation indices
#' @param y_truth the true classification labels for the data
#' @return a vector containing the classifications of observations x[i]
query_oracle <- function(i, y_truth) {
  as.vector(y_truth[i])
}

#####DISAGREEMENT
#' Computes entropy of committee's classifications
#'
#' Computes the disagreement measure for each of the unlabeled observations
#' based on the either the predicted class labels or the posterior
#' probabilities of class membership.
#'
#' @importFrom itertools2 izip
#' @importFrom entropy entropy
vote_entropy <- function(x, type='class', entropy_method='ML') {
  it <- do.call(itertools2::izip, x)
  disagreement <- sapply(it, function(obs) {
    entropy(table(unlist(obs)), method=entropy_method)
  })
  disagreement
}

#' @importFrom entropy entropy.plugin
post_entropy <- function(x, type='posterior') {
  avg_post <- Reduce('+', x) / length(x)
  apply(avg_post, 1, function(obs_post) {
    entropy.plugin(obs_post)
  })
}

kullback <- function(x, type='posterior') {
  avg_post <- Reduce('+', x) / length(x)
  kullback_members <- lapply(x, function(obs) {
    rowSums(obs * log(obs / avg_post))
  })

  Reduce('+', kullback_members) / length(kullback_members)
}

#' @importFrom entropy entropy.plugin
entropy_uncertainty <- function(posterior) {
  apply(posterior, 1, entropy.plugin)
}

least_confidence <- function(posterior) {
  apply(posterior, 1, max)
}

margin_uncertainty <- function(posterior) {
  apply(posterior, 1, function(post_i) {
    post_i[order(post_i, decreasing=T)[1:2]] %*% c(1, -1)
  })
}

#####HELPERS
# Returns a vector of indices of unlabeled observations.
which_unlabeled <- function(y) {
  which(is.na(y))
}

# Returns a vector of indices of labeled observations.
which_labeled <- function(y, return_logical = FALSE) {
  which(!is.na(y))
}

# Splits a matrix and its class labels into labeled and unlabeled pairs.
split_labeled <- function(x, y) {
  x <- as.matrix(x)
  y <- factor(y)

  unlabeled_i <- which_unlabeled(y)
  list(x_labeled=x[-unlabeled_i, ],
       y_labeled=y[-unlabeled_i],
       x_unlabeled=x[unlabeled_i, ],
       y_unlabeled=y[unlabeled_i])
}

#' Validates the classifier specified from the 'caret' package
#'
#' We ensure that the specified classifier is a valid classifier in the
#' \code{caret} package.
#'
#' @export
#' @param classifier string that contains the supervised classifier as given in
#' the \code{caret} package.
#' @param posterior_prob Are posterior probabilities required? If so, set to
#' \code{TRUE}. By default, set to \code{FALSE}.
#' @return \code{TRUE} invisibly if no errors occur.
#' @examples
#' validate_classifier('lda')
#' validate_classifier('What else floats? ... Very small rocks. ... Gravy.')
validate_classifier <- function(classifier, posterior_prob = FALSE) {
  # Tests that the specified classifier is given in 'caret', is actually a
  # classifier, and provides posterior probabilities of class membership.
  if (missing(classifier) || is.null(classifier) || is.na(classifier)) {
    stop("A classifier must be specified")
  }
  caret_lookup <- try(modelLookup(classifier), silent = TRUE)
  if (inherits(caret_lookup, "try-error")) {
    stop("Cannot find, '", classifier, "' in the 'caret' package")
  } else if (!any(caret_lookup$forClass)) {
    stop("The method, '", classifier, "' must be a classifier")
  }

  if (posterior_prob && !any(caret_lookup$probModel)) {
    stop("The method, '", classifier, "' must return posterior probabilities")
  }

  invisible(TRUE)
}














