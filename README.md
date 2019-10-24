# RandomForestUtils

<!-- badges: start -->
<!-- badges: end -->

The goal of RandomForestUtils is to create a reproducible Random Forest pipeline that allows indivivudals to get accruacate performance metrics for classifcation and regression. The pipeline uses data splitting and cross validation to ensure that the models being tested are not overfit.

## Installation

You can install this R package using devtools with:

``` r
devtools::install_github("nearinj/RandomForestUtils")
```

## Example

This is a basic example which shows you how to solve a common problem:

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

#non_rare_genera_RA <- sweep(non_rare_genera, 2, colSums(non_rare_genera), '/')
#head(colSums(non_rare_genera_RA))

#input_features <- data.frame(t(non_rare_genera_RA))

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

