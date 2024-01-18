rm(list = ls());


library(caret);
library(iml)
library(ggthemes)
library(ggplot2)
source("allib.r")


set.seed(1357);
data <- load_video_dataset();
names(data) <- c("cu","bs","ru","lt","cn","mp","bl","bt","ram","rm","jit","dly","bw","mos");

#train/test split
smp_size <- floor(0.75 * nrow(data));
set.seed(123);
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train_data <- data[train_ind, ]
test_data <- data[-train_ind, ]

cols_aux <- names(train_data);
cols <- cols_aux[1:ncol(train_data)-1];

X_train <- train_data[, c(cols)]
Y_train <- train_data$mos;
X_test <- test_data[, c(cols)]
Y_test <- test_data$mos;


control <- (method="none");
hparam <- data.frame(.nsets=5000, .ntreeperdiv=10, .ntreefinal =5000)

model <- train(X_train, Y_train, method = 'ordinalRF', tuneGrid=hparam);

pred <- Predictor$new(model, data = X_test, y = Y_test)


if (FALSE) {

#variable importance
vi <- varImp(model)


png(
  "test.png",
  width     = 4.5,
  height    = 3.25,
  units     = "in",
  res       = 1200,
  pointsize = 4
)
plot(vi)
dev.off()


#interaction
interact <- Interaction$new(pred)

png(                
  "test.png",                                                                        
  width     = 6.5,
  height    = 5.0,
  units     = "in",
  res       = 1600,
  pointsize = 4                                                                                         
)
> interact$results %>%
  ggplot(aes(x = reorder(.feature, .interaction), y = .interaction, fill = .class)) +
    facet_wrap(~ .class, ncol = 2) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_tableau() +
    coord_flip() +
    guides(fill = FALSE) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
panel.background = element_blank(), axis.line = element_line(colour = "black")) + ylab("Interaction") + xlab("Factors")
> dev.off( )



    



    
interact.bw <- Interaction$new(pred,feature="bw")
tree <- TreeSurrogate$new(pred, maxdepth = 5)

interact$results %>%
  ggplot(aes(x = reorder(.feature, .interaction), y = .interaction, fill = .class)) +
    facet_wrap(~ .class, ncol = 2) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_tableau() +
    coord_flip() +
    guides(fill = FALSE) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
panel.background = element_blank(), axis.line = element_line(colour = "black")) + ylab("Interaction") + xlab("Factors")


pd.bw <- FeatureEffect$new(pred, feature="bw")
pd.bw$center(min(X_test$bw))


pd.bw$plot() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
panel.background = element_blank(), axis.line = element_line(colour = "black")) + ylab("s") + xlab("bw")
}
