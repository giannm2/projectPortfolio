
library(tidyverse)
library(mlbench)
library(ranger)
library(caret)
library(Boruta)

#load in soybean data from: https://archive.ics.uci.edu/ml/datasets/Soybean+%28Large%29
data(Soybean)

#let's look at the data
glimpse(Soybean)

#the target column - soybean species has 19 levels
#this is multiclass classification, which random forest is well-suited for
summary(Soybean$Class)
length(levels(Soybean$Class))

set.seed(123) # ensure reproducibility of split

#change factors to numerics
soybeanNumeric <- Soybean %>% 
  mutate_at(vars(-("Class")), as.numeric) 

#make gathered df to view distributions of features
soybeanGather <- pivot_longer(soybeanNumeric, -Class, names_to = "valueName")
ggplot(data = soybeanGather, aes(x=as.factor(value), fill=as.factor(value))) +
  geom_bar(stat='count', width=1) +   
  facet_wrap(~valueName)

#random forest can;t have NAs, so remove them
soybeanNumeric <- na.omit(soybeanNumeric)

#split data into train and test
trainDataIndex <- createDataPartition(soybeanNumeric$Class, p = 0.7, list = F)
trainData <- soybeanNumeric[trainDataIndex,]
testData <- soybeanNumeric[-trainDataIndex,]

#scale data based on training means/SD
scaler <- preProcess(trainData[,-1], method = c("center", "scale"))
soyTrainX <- predict(scaler, trainData[,-1])
soyTrainY <- trainData[,1]
soyTrain <- cbind.data.frame(soyTrainX, soyTrainY) %>% dplyr::rename(Class = soyTrainY)

soyTestX <- predict(scaler, testData[,-1])
soyTestY <- testData[,1]
soyTest <- cbind.data.frame(soyTestX, soyTestY) %>% dplyr::rename(Class = soyTestY)


#train random forest (uses ranger for faster runtime random forest algorithm)
soyRF <- ranger(Class ~., 
                data = soyTrain,
                importance = "impurity" #this is based on GINI index
)

#calculate feature importances
rfImportance <- cbind.data.frame(colnames(soyTest[,-36]), importance(soyRF)) %>% 
  dplyr::rename(variable = `colnames(soyTest[, -36])`, 
                importance = `importance(soyRF)`)

#view importance plot
ggplot(rfImportance, aes(x=reorder(variable,importance), y=importance,fill=importance))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  scale_fill_gradient(low="red", high="blue")
#based on this, we could try to remove some of the features with low importance in a 
#RFE-style method, but it is sloppy so I will showcase a Random Forest-based feature selection method later


#lets use the random forest to predict the class
soyRFPredict <- csrf(Class ~., 
                     training_data = soyTrain,
                     test_data = soyTest
)

#view accuracies and other metrics of performance
confusionMatrix(soyRFPredict, testData$Class)
#92.81% accuracy

#let's see if we can increase accuracy with Boruta feature selection
#this works based on making shadow features, or permutations of features (shuffled features even)
#then, it calculates random forest importance (mean decreased acc) for a random forest on this shadow data
#this iterates, checking to see if the real feature is better than the shadow feature
#it removes features based on whether the feature has a higher z-score than any of the shadows
#now, let's put it in R

#run Boruta
soyBoruta <- Boruta(Class ~.,  data = soyTrain, doTrace = 1) #dotrace make output printing change
print(soyBoruta)

#fix tentative features
finalSoyBoruta <- TentativeRoughFix(soyBoruta)
print(finalSoyBoruta)

#get significant features
features <- getSelectedAttributes(finalSoyBoruta)
significantVariables <- as.data.frame(colnames(soyTrain) %in% features)
rownames(significantVariables)<- colnames(soyTrain)
significantVariables <- as.matrix(significantVariables)

#extract data to only have significant values
trainBoruta <- soyTrain %>% 
  select_if(significantVariables)

trainBoruta <- cbind.data.frame(trainBoruta, soyTrainY) %>% 
  dplyr::rename(Class = soyTrainY)

#lets use the random forest with selected features to predict the class
soyRFBorutaPredict <- csrf(Class ~., 
                           training_data = trainBoruta,
                           test_data = soyTest
)

#view accuracies and other metrics of performance
confusionMatrix(soyRFBorutaPredict, testData$Class)
#97.6% accuracy


# #let's also try it with caret, just to show for future

#fix factor to have only values in the df
trainBoruta$Class <- droplevels(trainBoruta$Class)

#actual training - note runtime
caretRF <- train(x = trainBoruta[,-33], 
                 y = trainBoruta[,33],
                 method = "rf")

#make predictions
caretPreds <- predict(caretRF, soyTest)

#view accuracies and other metrics of performance
confusionMatrix(factor(caretPreds, levels = levels(testData$Class)), testData$Class)
#94.01% accuracy








######################################################################################################################

library(tidymodels)
#library(ranger)
library(randomForest)
library(AmesHousing)

#Set system time zone to UTC (no adjustment for DLST)
Sys.setenv(TZ="UTC")

#Remove all previous variables (start fresh)
rm(list=ls(all=TRUE))

#Nuclear Option Baby!
options(stringsAsFactors = FALSE)

########################################
Amy<-AmesHousing::make_ames()

#You only need to do this once! (I checked.....)
set.seed(123)



#Recipe for Housing Prices

TurkeyRecipe<- recipe(Sale_Price ~. ,  data=Amy) %>% ## formula (outcome ~ predictors)
  step_zv(all_predictors()) %>% #delete any zero-variance predictors that have a single unique value.
  step_normalize(all_numeric_predictors())%>% #centers and scales the numeric predictors.
  step_corr(all_numeric_predictors(), threshold = 0.85) 

TurkeyRecipe

TurkeyRecipe<- recipe(Sale_Price ~. ,  data=Amy) %>% ## formula (outcome ~ predictors)
  step_zv(all_predictors()) %>% #delete any zero-variance predictors that have a single unique value.
  step_normalize(all_numeric_predictors())%>% #centers and scales the numeric predictors.
  step_corr(all_numeric_predictors(), threshold = 0.85) %>% # drop highly correlated predictors
  prep()
  
TurkeyRecipe




#Random Forest Model
TurkeyModel <- 
  # specify that the model is a random forest
  rand_forest() %>%
  # specify that the `mtry` parameter needs to be tuned
  #set_args(mtry = tune()) %>% #I'm skipping the tuning for now!
  # select the engine/package that underlies the model
  set_engine("ranger", importance = "permutation") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("regression") 


####
#We’re now ready to put the model and recipes together into a workflow. You initiate a workflow using workflow()
# set the workflow
TurkeyWorkflow <- workflow() %>%
  # add the recipe
  add_recipe(TurkeyRecipe) %>%
  # add the model
  add_model(TurkeyModel)


TurkeySplit <- initial_split(Amy,prop = 3/4)
#Since all of this information is contained within the workflow object, 
#we will apply the last_fit() function to our workflow and our train/test split object. 
#This will automatically train the model specified by the workflow using the training data, 
#and produce evaluations based on the test set.

TurkeyFit <- TurkeyWorkflow %>%
  # fit on the training set and evaluate on test set
  last_fit(TurkeySplit)



#Performance of the final model applied to the test set.
TurkeyFit$.metrics
TurkeyFit$.predictions 

#AH_Train <- fit(AH_workflow, training(AH_split))
#AH_Train




#But once you’ve determined your final model, you often want to train it on your full dataset 
#and then use it to predict the response for new data.

#If you want to use your model to predict the response for new observations, 
#you need to use the fit() function on your workflow and the dataset that you want to fit the final model on 
#(e.g. the complete training + testing dataset).

TurkeyDinner <- fit(TurkeyWorkflow, Amy)
TurkeyDinner


#If want to predict on new data (new data needs to be in same format as model data)
OneOff<-Amy[1,]
OneOff$Sale_Price
OneOff$Sale_Price<-NULL

predict(TurkeyDinner, new_data = OneOff)


#If you want to extract the variable importance scores from your model, 
#The function that extracts the model is pull_workflow_fit() and then you need to grab the 
#fit object that the output contains.

Parsnips <- extract_fit_parsnip(TurkeyDinner)$fit
Parsnips

#Then you can extract the variable importance from the ranger object itself 
#(variable.importance is a specific object contained within ranger output -

TurkeyDessert<-Parsnips$variable.importance
#AHImpt<-AHImpt[order(-AHImpt)]

ImpDF <- data.frame(Imp = TurkeyDessert)
ImpDF$Var<-rownames(ImpDF)
rownames(ImpDF)<-NULL
ImpDF<-ImpDF[order(-ImpDF$Imp),]
ImpDF<-ImpDF[1:30,]



p1<-ggplot(ImpDF) +
  geom_col(aes(x=reorder(Var,Imp,sum), y=Imp),
           col = "black", show.legend = F) +
  ylab("Variable Importance")+
  xlab("Variable Name")+
  ggtitle("Random Forrest")+
  coord_flip() +
  scale_fill_grey() +
  theme_bw()


p1







####################

#Hyper paramater tuning
ames_split <- initial_split(Amy, prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)


# default RF model
m1 <- randomForest(
  formula = Sale_Price ~ .,
  data    = ames_train
)

m1


plot(m1)
# number of trees with lowest MSE
which.min(m1$mse)


# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])

#mtry: Number of variables randomly sampled as candidates at each split. 
#ntree: Number of trees to grow

# names of features
features <- setdiff(names(ames_train), "Sale_Price")

m2 <- tuneRF(
  x          = ames_train[features],
  y          = ames_train$Sale_Price,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)

m2

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = 24,
    min.node.size   = 5,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

m1
optimal_ranger


df <- data.frame(imp = m1$importance)
df$var<-rownames(df)
rownames(df)<-NULL
colnames(df)[1]<-"imp"
df<-df[order(-df$imp),]
df2<-df[1:30,]



p1<-ggplot(df2) +
  geom_col(aes(x=reorder(var,imp,sum), y=imp),
           col = "black", show.legend = F) +
  ylab("Variable Importance")+
  xlab("Variable Name")+
  ggtitle("Random Forrest Bloom Status")+
  coord_flip() +
  scale_fill_grey() +
  theme_bw()


p1





optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 important variables")


############################################


#library(tidyverse)
#library(mlbench)
#library(ranger)
#library(caret)
#library(Boruta)
library(openxlsx)




#Task: Create a decision tree with RF for GDP growth data


#set the seed
set.seed(100)

#import the data
gd <- read.xlsx("/Users/moriarty/Documents/RPI/ClassWork/MLEB/Class20/growth.xlsx")
glimpse(gd)
gd <- gd %>% mutate(growth = as.factor(growth > 0))

#drop the oil column with no data
gd <- gd %>% select(!starts_with("oil") & !starts_with("country"))
View(gd)

#split data
growthTrainIndex <- createDataPartition(gd$growth, p = 0.7, list = F)
g.trainData <- gd[growthTrainIndex, ]
g.testData <- gd[-growthTrainIndex, ]



#RF
growthRF <- csrf(growth ~ ., 
                 training_data = g.trainData,
                 test_data = g.testData)

#view accuracies and other metrics of performance
confusionMatrix(growthRF, g.testData$growth)
#89.47% accuracy







##############################################



#Read in Turkey Time data

Turk<-read.csv("/Users/moriarty/Documents/RPI/ClassWork/MLEB/TurkeyTime.csv")

nrow(Turk)
Turk<-Turk %>%
  filter(!Type == 'Armadillo ')
nrow(Turk)

str(Turk)

#So many characters
Turk <- Turk %>% 
  mutate_if(is.character,as.factor)

str(Turk)



#Split into Test and training (70/ 30)

Tsplit<-initial_split(Turk, prop=0.7)

#Look at split
Tsplit

#Look at training dataset
Tsplit %>%
  training() %>%
  glimpse()
####################################
#All Kitchen metaphors 

#Prep cook
#step_corr() - Removes variables that have large absolute correlations with other variables
#step_center() - Normalizes numeric data to have a mean of zero
#step_scale() - Normalizes numeric data to have a standard deviation of one


TurkeyRecipe<- training(Tsplit) %>%
  recipe(Type ~.) %>% ## formula (outcome ~ predictors)
  #step_corr(all_predictors()) %>%
  #step_center(all_predictors(), -all_outcomes()) %>%
  #step_scale(all_predictors(), -all_outcomes()) %>%
  step_center(Height, Weight) %>%
  step_scale(Height, Weight) %>% 
  #If wanted to upsample  themis::step_upsample(x,over_ratio = 1)
  prep()
  

TurkeyRecipe


#Pre processing

TurkeyTesting <- TurkeyRecipe %>%
  bake(testing(Tsplit)) 

glimpse(TurkeyTesting)



TurkeyTraining <- juice(TurkeyRecipe)
glimpse(TurkeyTraining)




#Tidy models can work with all sorts of packages

FreeRange <- rand_forest(trees = 100, mode = "classification") %>%
  set_engine("ranger") %>%
  fit(Type ~ ., data = TurkeyTraining)

#Could have used
#set_engine("randomForest") %>%

#Armadillo only had one observation!




predict(FreeRange, TurkeyTesting)


#Add the predictions to the baked testing data with bind_cols() 
FreeRange %>%
  predict(TurkeyTesting) %>%
  bind_cols(TurkeyTesting) %>%
  glimpse()

#Use the metrics() function to measure the performance of the model. 
#It will automatically choose metrics appropriate for a given type of model
FreeRange %>%
  predict(TurkeyTesting) %>%
  bind_cols(TurkeyTesting) %>%
  metrics(truth = Type, estimate=.pred_class)



#Per classifer metrics
FreeRange %>%
  predict(TurkeyTesting, type = "prob") %>%
  glimpse()
  

TurkeyProbs <- FreeRange %>%
  predict(TurkeyTesting, type = "prob") %>%
  bind_cols(TurkeyTesting)

glimpse(TurkeyProbs)


TurkeyProbs%>%
  gain_curve(Type, .pred_Bunny:.pred_Walrus) %>%
 autoplot()



FreeRange <- rand_forest(trees = 100, mode = "classification") %>%
  set_engine("ranger") %>%
  fit(Type ~ ., data = TurkeyTraining)


Bloom_performance <- FreeRange %>% collect_metrics()
Bloom_performance





# extract the test set predictions themselves using the collect_predictions() function
# generate predictions from the test set (lots of rows here- not much useful help)
Bloom_predictions <- Bloom_fit %>% collect_predictions()
Bloom_predictions



# generate a confusion matrix
Bloom_predictions %>% 
  conf_mat(truth = Type, estimate = .pred_class)


#We could also plot distributions of the predicted probability distributions for each class.

Bloom_predictions %>%
  ggplot() +
  geom_density(aes(x = .pred_0, fill = Type), 
               alpha = 0.5)

Bloom_predictions %>%
  ggplot() +
  geom_density(aes(x = .pred_1, fill = Type), 
               alpha = 0.5)













MyPred<-FreeRange %>%
  predict(TurkeyTesting) 
MyPred<-data.frame(MyPred)
MyPred<-MyPred$.pred_class

ConfuseMe<-data.frame(TurkeyTesting)
ConfuseMe$Pred_class <-MyPred
ConfuseMe<-ConfuseMe[,c("Type", "Pred_class")]
colnames(ConfuseMe)<-c("Type", "Pred_class")

ConfuseMe %>% 
  conf_mat(truth = Type, estimate = Pred_class)



Bloom_predictions %>%
  ggplot() +
  geom_density(aes(x = .pred_0, fill = Type), 
               alpha = 0.5)

Bloom_predictions %>%
  ggplot() +
  geom_density(aes(x = .pred_1, fill = Type), 
               alpha = 0.5)


#################################






