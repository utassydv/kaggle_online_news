library(tidyverse)
library(h2o)

my_seed <- 20210305

h2o.init()
h2o.removeAll()
orig_data_train <- read_csv("data/train.csv")
orig_data_test <- read_csv("data/test.csv")

###############################################################################################################################################################
# EDA
## TODO:

skimr::skim(orig_data_train)

#convert to factors
factors <- c('data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
             'data_channel_is_tech', 'data_channel_is_world', 
             'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
             'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'article_id')#, 'is_popular')

orig_data_train[,factors] <- lapply(orig_data_train[,factors], factor)
orig_data_test[,factors] <- lapply(orig_data_test[,factors], factor)

###############################################################################################################################################################
# H2O data

h2o_data_train <- as.h2o(orig_data_train)
h2o_data_test <- as.h2o(orig_data_test)


splitted_data <- h2o.splitFrame(h2o_data_train, ratios = 0.75, seed = my_seed)
data_train <- splitted_data[[1]]
data_test <- splitted_data[[2]]

y <- "is_popular"
X <- setdiff(names(h2o_data_train), y)


h2oSubmittion <- function(model, name, test_data){
  to_submit <- data.table(
    article_id = as.numeric(orig_data_test$article_id),
    score = as.data.frame(h2o.predict(object = model, newdata = test_data))
  )
  colnames(to_submit) <- c("article_id", "score")
  
  write.csv(to_submit, paste0("submissions/",name,".csv"), row.names=FALSE)
}


###############################################################################################################################################################
### 1, A linear model prediction after parameter tuning

###############################################################################################################################################################
### 2, A random forest prediction after parameter tuning

rf_params <- list(
  ntrees = c(10, 50, 100, 300, 500),
  mtries = c(2, 4, 6, 8, 9),
  sample_rate = c(0.2, 0.632, 1),
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

h2o.getGrid(rf_grid@grid_id, "mse")
best_rf <- h2o.getModel(
  h2o.getGrid(rf_grid@grid_id, sort_by = "mse", decreasing = TRUE)@model_ids[[1]]
)
saveRDS(best_rf, "models/best_rf")

best_rf <- readRDS("models/best_rf")


rf_performance_summary <- h2o.getGrid(rf_grid@grid_id, "mse")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("mse", names(rf_params)), as.numeric))

ggplot(rf_performance_summary, aes(ntrees, mse, color = factor(mtries))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "mtry")
ggsave("plots/rf_summary_plot.png")

h2oSubmittion(best_rf, "best_rf", h2o_data_test)


###############################################################################################################################################################
### 3, A gradient boosting prediction after parameter tuning

gbm_params <- list(
  learn_rate = c(0.01, 0.05, 0.1, 0.3),  # default: 0.1
  ntrees = c(10, 50, 100, 300, 500),
  max_depth = c(2, 5),
  sample_rate = c(0.2, 0.5, 0.8, 1)
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
  h2o.getGrid(gbm_grid@grid_id, sort_by = "auc", decreasing = TRUE)@model_ids[[1]])
gbm_performance_summary <- h2o.getGrid(gbm_grid@grid_id, "auc")@summary_table %>%
  as_tibble() %>%
  mutate(across(c("auc", names(gbm_params)), as.numeric))
ggplot(gbm_performance_summary, aes(ntrees, auc, color = factor(learn_rate))) +
  geom_line() +
  facet_grid(max_depth ~ sample_rate, labeller = label_both) +
  theme(legend.position = "bottom") +
  labs(color = "learning rate")

ggsave("plots/gbm_summary_plot.png")

saveRDS(best_gbm, "models/best_gbm")

h2oSubmittion(best_gbm, "best_gbm", h2o_data_test)

#h2o.auc(best_gbm, xval = TRUE)

### 4, A neural network prediction after parameter tuning.

###############################################################################################################################################################