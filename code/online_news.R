library(tidyverse)
library(h2o)
library(data.table)

my_seed <- 20210305

#h2o.shutdown()
#h2o.init()
h2o.init(min_mem_size='100G', max_mem_size='200G')
h2o.removeAll()
h2o.show_progress()


orig_data_train <- read_csv("data/train.csv")
orig_data_test <- read_csv("data/test.csv")

###############################################################################################################################################################
# EDA


skimr::skim(orig_data_train)

#convert to factors
factors <- c('data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
             'data_channel_is_tech', 'data_channel_is_world', 
             'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
             'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend')#, 'article_id')#, 'is_popular')

orig_data_train[,factors] <- lapply(orig_data_train[,factors], factor)
orig_data_test[,factors] <- lapply(orig_data_test[,factors], factor)


LogTransformSkewedVars <- function(df, treshold, exclude){
  library(data.table)
  library(moments) # for skewness
  
  # Calculate skewness
  numerics <- names(df)[sapply(df, is.numeric)]
  skewness_of_vars <- t(as.data.frame(lapply(df[,numerics], skewness)))
  
  # Get skewed vars
  skewness_of_vars <- as.data.table(skewness_of_vars, keep.rownames=TRUE)
  skewed_vars <- skewness_of_vars[V1>0.4]$rn
  
  # remove var to exclude
  skewed_vars <- skewed_vars[skewed_vars!=exclude]
  
  df[,skewed_vars] <- lapply(df[,skewed_vars], function(x) log(x+1))
  return(df)
}

orig_data_train <- LogTransformSkewedVars(orig_data_train, 0.4, 'is_popular')
orig_data_test <- LogTransformSkewedVars(orig_data_test, 0.4, 'is_popular')


###############################################################################################################################################################
# H2O data

h2o_data_train <- as.h2o(orig_data_train)
h2o_data_test <- as.h2o(orig_data_test)

data_train <- h2o_data_train
#splitted_data <- h2o.splitFrame(h2o_data_train, ratios = 0.75, seed = my_seed)
#data_train <- splitted_data[[1]]
#data_test <- splitted_data[[2]]

y <- "is_popular"
X <- setdiff(names(h2o_data_train), y)


h2oSubmittion <- function(model, name, test_data){
  to_submit <- data.table(
    article_id = as.numeric(as.character(orig_data_test$article_id)),
    score = as.data.frame(h2o.predict(object = model, newdata = test_data))
  )
  to_submit <- to_submit[,1:2]
  colnames(to_submit) <- c("article_id", "score")
  
  write.csv(to_submit, paste0("submissions/",name,".csv"), row.names=FALSE)
}


###############################################################################################################################################################
### 1, A linear model prediction after parameter tuning
lasso <- h2o.glm(
  X, y,
  training_frame = data_train,
  model_id = "lasso",
  family = "binomial",
  alpha = 1,
  lambda_search = TRUE,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE 
)
saveRDS(lasso, "models/lasso")

lasso <- readRDS("models/lasso")
h2oSubmittion(lasso, "lasso", h2o_data_test)

###############################################################################################################################################################
### 2, A random forest prediction after parameter tuning
#
rf_params <- list(
  ntrees = c(100, 300),
  mtries = c(4, 6, 7, 8),
  sample_rate = c(1),
  max_depth = c(10, 20)
)

rf_grid <- h2o.grid(
  "randomForest", x = X, y = y,
  training_frame = data_train,
  grid_id = "rf",
  nfolds = 5,
  seed = my_seed,
  hyper_params = rf_params
)
# parallelism = 0 #auto

h2o.getGrid(rf_grid@grid_id, "mse")
best_rf <- h2o.getModel(
  h2o.getGrid(rf_grid@grid_id, sort_by = "mse")@model_ids[[1]]
)
saveRDS(best_rf, "models/id_data_best_rf")

best_rf <- readRDS("models/id_data_best_rf")


rf_performance_summary <- h2o.getGrid(rf_grid@grid_id, "mse")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("mse", names(rf_params)), as.numeric))

ggplot(rf_performance_summary, aes(ntrees, mse, color = factor(mtries))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "mtry")
ggsave("plots/id_data_rf_summary_plot.png")

h2oSubmittion(best_rf, "id_data_best_rf", h2o_data_test)



###############################################################################################################################################################
### 3, A gradient boosting prediction after parameter tuning

gbm_params <- list(
  learn_rate = c(0.05, 0.1, 0.3),  # default: 0.1
  ntrees = c(50, 100, 300),
  max_depth = c(2, 5),
  sample_rate = c(0.2, 0.5, 1)
)
gbm_grid <- h2o.grid(
  "gbm", x = X, y = y,
  grid_id = "gbm",
  training_frame = data_train,
  nfolds = 5,
  seed = my_seed,
  hyper_params = gbm_params
)

best_gbm <- h2o.getModel(
  h2o.getGrid(gbm_grid@grid_id, sort_by = "mse")@model_ids[[1]])
gbm_performance_summary <- h2o.getGrid(gbm_grid@grid_id, "mse")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("mse", names(gbm_params)), as.numeric))
ggplot(gbm_performance_summary, aes(ntrees, mse, color = factor(learn_rate))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "learning rate")

ggsave("plots/id_data_gbm_summary_plot.png")

saveRDS(best_gbm, "models/id_data_best_gbm")

best_gbb <- readRDS("models/id_data_best_gbm")

h2oSubmittion(best_gbm, "id_data_best_gbm", h2o_data_test)

###############################################################################################################################################################
###  GBM clever tuning#
#TODO try this, takes lot time!
gbm_params <- list(
  learn_rate = c(0.05, 0.01, 0.1),  # default: 0.1
  ntrees = c(100, 500),
  max_depth = c(2, 5),
  sample_rate = c(0.2, 0.5, 1)
)
gbm_grid <- h2o.grid(
  "gbm", x = X, y = y,
  grid_id = "gbm",
  training_frame = data_train,
  nfolds = 5,
  seed = my_seed,
  hyper_params = gbm_params
)

best_gbm_tuned <- h2o.getModel(
  h2o.getGrid(gbm_grid@grid_id, sort_by = "mse")@model_ids[[1]])
gbm_performance_summary <- h2o.getGrid(gbm_grid@grid_id, "mse")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("mse", names(gbm_params)), as.numeric))
ggplot(gbm_performance_summary, aes(ntrees, mse, color = factor(learn_rate))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "learning rate")

ggsave("plots/id_data_best_gbm_tuned_summary_plot.png")

saveRDS(best_gbm_tuned, "models/id_data_best_gbm_tuned")

best_gbm_tuned <- readRDS("models/id_data_best_gbm_tuned")

h2oSubmittion(best_gbm_tuned, "id_data_best_gbm_tuned", h2o_data_test)

############################################################################################################
#XGBOOST SIMPLE

#titanic_xgb <- h2o.xgboost(x = X,
#                           y = y,
#                           training_frame = data_train,
#                           validation_frame = data_test,
#                           booster = "dart",
#                           normalize_type = "tree",
#                           seed = my_seed)
#
#h2o.mse(h2o.performance(titanic_xgb,data_test))
#
#h2oSubmittion(titanic_xgb, "simple_xgboost", h2o_data_test)


############################################################################################################
#XGBOOST

xgboost_params <- list(
  learn_rate = c(0.1, 0.3),  # same as "eta", default: 0.3
  ntrees = c(50, 100, 300),
  max_depth = c(2, 5),
  gamma = c(0, 1, 2),  # regularization parameter
  sample_rate = c(0.5, 1)
)

xgboost_grid <- h2o.grid(
  "xgboost", x = X, y = y,
  grid_id = "xgboost",
  training_frame = data_train,
  nfolds = 5,
  seed = my_seed,
  hyper_params = xgboost_params
)

best_xgboost <- h2o.getModel(
  h2o.getGrid(xgboost_grid@grid_id, sort_by = "mse")@model_ids[[1]]
)

xgboost_performance_summary <- h2o.getGrid(xgboost_grid@grid_id, "mse")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("mse", names(xgboost_params)), as.numeric))
ggplot(xgboost_performance_summary, aes(ntrees, mse, color = factor(learn_rate))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "learning rate")

ggsave("plots/id_data_xgboost_summary_plot.png")

saveRDS(best_xgboost, "models/id_data_best_xgboost")

best_xgboost <- readRDS("models/id_data_best_xgboost")

h2oSubmittion(best_xgboost, "id_data_best_xgboost", h2o_data_test)

############################################################################################################
### 4, A neural network prediction after parameter tuning.
nn1 <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  model_id = "nn1",
  hidden = c(32),
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)
saveRDS(nn1, "models/nn1")

nn1 <- readRDS("models/nn1")
h2oSubmittion(nn1, "nn1", h2o_data_test)

nn2 <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  model_id = "nn2",
  hidden = c(32, 8),
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)
saveRDS(nn2, "models/nn2")

nn2 <- readRDS("models/nn2")
h2oSubmittion(nn2, "nn2", h2o_data_test)
