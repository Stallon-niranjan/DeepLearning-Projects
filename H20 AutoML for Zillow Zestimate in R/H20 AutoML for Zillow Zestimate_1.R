library(data.table)
library(h2o)

# Load train and properties data

properties <- fread("properties_2016.csv", header=TRUE, stringsAsFactors=FALSE, colClasses = list(character = 50))
train      <- fread("train_2016.csv")
training   <- merge(properties, train, by="parcelid",all.y=TRUE)

# Initialise h20
h2o.init(nthreads = -1, max_mem_size = "8g")

# Mark predictor and response variables
x <- names(training)[which(names(training)!="logerror")]
y <- "logerror"

# Import data into H2O
train <- as.h2o(training)
test <- as.h2o(properties)

# Fit H2O AutoML Mode;
aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_runtime_secs = 1800, stopping_metric='MAE')

# Store the H2O AutoML Leaderboard
lb <- aml@leaderboard
lb

# Use Best Model in the leaderboard
aml@leader

# Generate Predictions using the leader Model
pred <- h2o.predict(aml, test)

predictions <- round(as.vector(pred), 4)

# Prepare predictions for submission file
result <- data.frame(cbind(properties$parcelid, predictions, predictions,
                          predictions, predictions, predictions,
                          predictions))

colnames(result)<-c("parcelid","201610","201611","201612","201710","201711","201712")
options(scipen = 999)

# Wite results to submission file
write.csv(result, file = "submission_xgb_ensemble.csv", row.names = FALSE )
