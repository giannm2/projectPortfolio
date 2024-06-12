library(tidyverse)
library(janitor)
library(rLakeAnalyzer)
library(cowplot)
library(corrplot)
library(psych)
library(ranger)

#load in data, clean names
infoDf <- read_csv("DO_Data/Info_Data.csv") %>% clean_names() %>% select(-row_id)
rawDf <- read_csv("DO_Data/Raw_Data.csv") %>% clean_names() %>% select(-row_id)
useDf <- read_csv("DO_Data/Use_Data.csv") %>% clean_names() %>% select(-row_id, -comment)
elevationDf <- read_csv("DO_Data/Elevation_Data.csv") %>% clean_names() %>% select(-row_id, -data_source)

#set seed
set.seed(123)

#filter to US lakes
infoDf <- infoDf %>% 
  filter(country == "USA")

#join use and elevation to this (only US lakes)
join1 <- left_join(infoDf, elevationDf, by=c("lake_id", "name"))
join2 <- left_join(join1, useDf, by=c("lake_id", "name"))

#join this onto time series data now for main dataframe
df <- inner_join(rawDf, join2, by=c("lake_id", "name"))

#check that it is only US
levels(as.factor(df$country))

#filter out some columns
importantCols <- c("date","depth", "temp", "lat", "long", "max_depth", "surf_area", "elevation", 
                   "perag", "perdev", "perwater", "perfor", "perwet", "pergrass", "pershrub", "do_con", "lake_id")
modelDf <- df %>% 
  select(all_of(importantCols))

#make slice areas for all depths in data
modelDf <- modelDf %>% 
  mutate(
    sliceArea = as.numeric((approx.bathy(Zmax = max_depth, lkeArea = surf_area, method = "cone", zinterval = 1, depths = depth))$Area.at.z),
    month = lubridate::month(ymd(date))) %>% 
  select(-max_depth, -surf_area, -date)

#look at data, lets visualize distributions of columns
my_plots <- lapply(names(modelDf), function(var_x){
  p <- 
    ggplot(modelDf) +
    aes_string(var_x)
  
  if(is.numeric(modelDf[[var_x]])) {
    p <- p + geom_density()
    
  } else {
    p <- p + geom_bar()
  } 
  
})
plot_grid(plotlist = my_plots)

#lets look at correlations between columns
corrplot(cor(na.omit(modelDf)), method = "number")


#lets do factor analysis as dimensionality reduction technique to look for latent variables
#let's drop values with NAs
finalModelDf <- na.omit(modelDf)

#make correlation matrix
modelCor <- cor(finalModelDf)

#kaiser-mayer-olkin test
KMO(modelCor)
#MSA = 0.37, which is less than 0.5, seems not great for EFA, but this might be due to high number of features so let's still try

#parallel analysis (more precise scree)
fa.parallel(modelCor, n.obs = 771705)
#recommends 2 factors, 1 PC

#factor analysis
modelFA <- fa(finalModelDf, nfactors = 7, rotate = "varimax")

#summarise the factor analysis, then view the diagram
summary(modelFA)
#poor tli value
fa.diagram(modelFA)


#lets make a random forest model as our regression predictor

#split data on lakeIDS, not just a random split
#get lakeIDs
lakeIDs <- unique(finalModelDf$lake_id)

#get ~70% of lakes in data
trainIDs <- sample(x = lakeIDs, size = 224)

#get train data
trainData <- finalModelDf %>% 
  dplyr::filter(lake_id %in% trainIDs) %>% 
  select(-lake_id)

#get test data
testData <- finalModelDf %>% 
  filter(!(lake_id %in% trainIDs))%>% 
  select(-lake_id)

#train RF model for regression
model <- ranger(do_con ~., 
                data = trainData,
                importance = "impurity" #this is based on GINI index
                )

#calculate feature importances
rfImportance <- cbind.data.frame(colnames(trainData[,-13]), importance(model)) %>% 
  dplyr::rename(variable = `colnames(trainData[, -13])`, 
                importance = `importance(model)`)

#view importance plot
ggplot(rfImportance, aes(x=reorder(variable,importance), y=importance,fill=importance))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  scale_fill_gradient(low="red", high="blue")

#make predictions
modelPredictions <- predict(model, testData)

#r-squared helper function
rsq <- function (x, y) cor(x, y) ^ 2

#view accuracies and other metrics of performance
rsq(modelPredictions$predictions, testData$do_con)
#0.3884964 r-squared
RMSE(modelPredictions$predictions, testData$do_con)
#3.501408 RMSE
MAE(modelPredictions$predictions, testData$do_con)
#2.273699 MAE
