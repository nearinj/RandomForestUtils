# RandomForestUtils

<!-- badges: start -->
<!-- badges: end -->

The goal of RandomForestUtils is to create a reproducible Random Forest pipeline that allows indivivudals to get accruacate performance metrics for classifcation and regression. The pipeline uses data splitting and cross validation to ensure that the models being tested are not overfit.

**The regression pipeline has not been validated and for now I would suggest using caret to train and test any random forest regression models you would like to create**

## Installation

You can install this R package using devtools with:

``` r
devtools::install_github("nearinj/RandomForestUtils")
```

## Pipeline Overview
The general workflow falls into three different steps outlined below

**Step 1: Splitting data into training and test datasets**
This is a critical step to evaluating model performance and to avoid reporting the performance of models that are grossly over fit. The idea here is that you take your dataset and split them up into training data and testing data. This split then allows you to determine how well your model would perform on data that it has never seen before which is important to evaluate whether or not your model is over fit. Generally this split is done so that the majority of data is kept for training and a small subset is used for testing purposes. In this tutorial we will use 20% of the data for testing and 80% for training. 

**Step 2: Model Training**
Once you have your training dataset the next step is begin training your model. There are two parameters that can be changed in this step for Random Forest modelling; the number of trees and the number of randomly chosen variables to be tested at each split within the trees (mtry). 

In general as the number of trees increase the performance of the model will increase until a plateau is reached. This plateau can vary between data sets and sometimes its worth trying multiple numbers to find the optimal number of trees. You can always set a very large number for this value and not worry about this issue, however, this will significantly increase the run time of your models.

Determining the best mtry (the only hyper-parameter in Random Forest) is a bit more difficult because it varies from dataset to dataset. A good starting point is usually the square root of the number of features, however, this is not always the optimal solution. If you set up mtry equal to the number of features within the dataset then you end up with trees in your model that are all very similar to one another (which sometimes can be helpful but is generally not the optimal solution either). In order to find the best mtry parameter for your dataset you will need to do multiple cross-validations. Luckily the scripts presented in this tutorial make this task fairly straight-forward. 

Cross-validation is one of many different methods that can be used to combat over fitting when tuning hyper parameters such as mtry. In this case we will focus of k-fold repeated cross validation. K-fold cross validation works by taking the training dataset supplied to the model algorithm and further splitting them up into k even numbered data sets. A model is then trained on each dataset and validated on the other data sets. This helps further protect against picking hyper-parameters that just happen to work really well on the split you made between training and test. Generally this process is then repeated n times to determine which mtry value works best overall and then retraining a model on the entire set of training data the model was supplied with.


**Step 3 Model Validation**
This final step is where the test dataset comes into play. During this step you will take the final model from training and use it to make classifications (for categorical variables) or regressions (for continuous variables) on the hold out test set. This will allow you to determine how well you can expect your model to work on data that it has never come across before. This is important when reporting things such as how well your model can predict disease outcome.


## Example

This is a basic example of how to use the pipeline to run a RandomForest classifcation problem. This dataset that comes along with the package was published in [2015 by Singh et al.,](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-015-0109-2) who were interested in looking a differences between healthy individuals and those with enteric infections. Some pre-processing of the data is required before it can be used! 

**Note that if you build the vignettes for this package there is a provided example of data analysis using this pipeline**

``` r
library(RandomForestUtils)

data("Edd_Singh_data")
indexs_to_keep <- which(Edd_Singh_data[[2]]$total_reads >= 1000)

clean_Genus_values <- Edd_Singh_data[[1]][,indexs_to_keep]
clean_metadata <- Edd_Singh_data[[2]][indexs_to_keep,]
rownames(clean_metadata) <- gsub("-","\\.",rownames(clean_metadata))
all.equal(colnames(clean_Genus_values), rownames(clean_metadata))

dim(clean_Genus_values)


non_rare_genera <- remove_rare(clean_Genus_values, 0.2)
dim(non_rare_genera)

input_features <- data.frame(apply(non_rare_genera +1, 2, function(x) {log(x) - mean(log(x))}))
input_features <- data.frame(t(input_features))

classes <- clean_metadata$Status
head(classes)

classes <- factor(classes, levels(classes)[c(2,1)])
head(classes)

#recode to case and control

library(forcats)


classes <- fct_recode(classes, Case="Patient",
                      Control="Healthy")
head(classes)


SAVE_PATH <- "~/etc/my_Tutorials/RF_tutorial/testing/"
set.seed(1995)
seeds <- sample.int(10000000, 10)
RandomForestUtils::set_cores(10)
rf_results <- RandomForestUtils::Run_RF_Pipeline(feature_table = input_features,
                                                 classes=classes,
                                                 metric = "ROC",
                                                 sampling=NULL,
                                                 repeats=10,
                                                 path=SAVE_PATH,
                                                 nmtry=4,
                                                 ntree=501,
                                                 nfolds=3,
                                                 ncrossrepeats = 5,
                                                 pro=0.8,
                                                 list_of_seeds = seeds)
```

