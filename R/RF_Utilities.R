#####################################################################
# Author: Jacob Nearing;
# Date April 10th 2019
# Title: Script to run the main RF pipeline
#####################################################################


#' Set number of cores to run random forest pipeline on
#' 
#' @param x Number of cores to run pipeline. Defaults to 1.
#' @examples 
#' set_cores(10)
set_cores <- function(x){
  CORES_TO_USE <- x
  registerDoMC(cores=CORES_TO_USE)
}


#' Random Forest Classifcation pipeline
#' 
#' @param feature_table The feature table that contains the information to be input into the random forest classifier. Note that this 
#' table should not include information about the classes that are being predicted.
#' @param classes A vector that represents the classes that each sample (row) in the feature table represents. This can be coded as
#' Case (level 1 factor) and control (level 2 factor). Make sure the factor levels are correct with using AUPRC or results will not always be correct.
#' @param metric A string that indicates whether the pipeline should use AUROC or AUPRC. For AUROC set metric="ROC". For AUPRC set metric="PR". 
#' Defaults to "ROC".
#' @param ntree An integer that represents the number of trees that you want to use during randoom forest construction. Defaults to 999.
#' @param nmtry An integer representing the number of different mtry values that you want to test during cross validation. The values of mtry to test
#' is calculated as follows:  mtry <- round(seq(1, number_of_features/3, length=nmtry)). Defaults to 7.
#' @param sampling A string indicating that type of sampling that should be done incase of inbalanced class designs. Options include: "up", "down" "SMOTE" and NULL.
#' @param nfolds An integer that represents the number of folds to used during cross validation. Defaults to 5.
#' @param ncrossrepeats An integer that represents the number of times to run cross validation on k folds. Defaults to 10. 
#' @param pro The proporition of samples that should be used for training versus testing during cross validation. Defaults to 0.8
#' @param SEED The random seed used to split the samples during cross validation. Defaults to 1995.
#' 
#' 
#' @description This function should not be run alone. You should use Run_RF_pipeline function to run the main pipeline for this package. 
#' 
rf_classification_pipeline <- function(feature_table, classes, metric="ROC", 
                                      ntree=999, nmtry=7, sampling=NULL,
                                      nfolds=5, ncrossrepeats=10, pro=0.8, SEED=1995){
  #create vectors to save cv and test metric values for each data split 

  
  
  ### create data split
  message("splitting data")
  
  ### Head function will need to pass in SEED values to run 100 data partitions or it won't be reproducible
  set.seed(SEED)
  train_index <- createDataPartition(classes, p=pro, list=FALSE)
  message(train_index)
  ## make training feature and class tables
  train_classes <- classes[train_index]
  train_features <- feature_table[train_index,]
  ## make test feature and class tables
  test_classes <- classes[-train_index]
  test_features <- feature_table[-train_index,]
  message("finished splitting data")
  ### do we want to do cross fold validation or montecarlo??
  ### for now we will default to cv
  
  folds <- nfolds
  repeats <- ncrossrepeats
  mtry_num <- nmtry
  ## change back to 100 in future
  set.seed(SEED)
  cvIndex <- caret::createMultiFolds(factor(train_classes), folds, times=repeats)
  set.seed(SEED)
  seeds_len <- (repeats*folds) + 1
  seeds <- vector(mode="list", length=(seeds_len))
  for(i in 1:(seeds_len-1)){
    seeds[[i]] <- sample.int(n=1000, mtry_num)
  }
  seeds[seeds_len] <- sample.int(1000, 1)
  if(metric=="PR"){
    message("Metric to test on is AUPRC")
    cv <- trainControl(method="repeatedcv",
                       number=folds,
                       index=cvIndex,
                       returnResamp = "final",
                       summaryFunction=prSummary,
                       classProbs=TRUE,
                       savePredictions = TRUE,
                       seeds=seeds)
    metric="AUC"
    }else if(metric=="ROC"){
      cv <- trainControl(method="repeatedcv",
                         number=folds,
                         index=cvIndex,
                         returnResamp = "final",
                         classProbs=TRUE,
                         summaryFunction=twoClassSummary,
                         savePredictions = TRUE,
                         seeds=seeds)
      metric <- "ROC"
    }
  ### set sampling method
  cv$sampling <- sampling
  ### set up grind to search for best mtry
  
  ### set up mtry values to test
  n_features <- ncol(feature_table)
  message(n_features)
  #set up 6 different mtry values to test
  mtry <- round(seq(1, n_features/3, length=nmtry))
  mtry <- mtry[mtry <= n_features]
  message(mtry)
  grid <- expand.grid(mtry = mtry)
  ## train the model
  message("Training model")
  set.seed(SEED)
  trained_model <- train(train_features, train_classes,
                         method="rf",
                         trControl=cv,
                         metric=metric,
                         tuneGrid=grid,
                         ntree=ntree,
                         importance=TRUE)
  
  message("Finished training model")
  # Mean AUC value over repeates of the best hyperparameter during training
  if(metric=="ROC"){
    cv_auc <- getTrainPerf(trained_model)$TrainROC
  }else if(metric=="AUC"){
    cv_auc <- getTrainPerf(trained_model)$TrainAUC
  }
  
  ### get important features for later validation...
  important_features <- trained_model$finalModel$importance
  
  ## predict on the test set and get predicted probabilities
  rpartProbs <- predict(trained_model, test_features, type="prob")
  if(metric=="ROC"){
    ### critcal that factor levels are correct for this to calculate.... 
    test_roc <- pROC::roc(ifelse(test_classes=="Case", 1, 0), rpartProbs[[2]])
    test_auc <- test_roc$auc
  }else if(metric=="AUC"){
    ######### this doesn't compute right needs to be fixed!!!!
    #get probs for postive class
    matriz <- cbind(test_classes, predict(trained_model, test_features, type="prob"), predict(trained_model, test_features))
    names(matriz) <- c("obs", levels(test_classes), "pred")
    pr_test_stats <- prSummary(matriz, levels(test_classes))
    test_auc <- pr_test_stats[1]
  }else{
    message("metric not used by this pipeline")
    return(NULL)
  }

  ### return in a list the cv_auc, the test_auc, the trained_model results and the important features...
  results <- list(cv_auc, test_auc, trained_model$results, important_features, trained_model)
  return(results)
}


#' rf_regression_pipeline
#' 
#' @param feature_table The input feature table that contains the features (columns) and samples (rows) that should be used for model training and cross validation.
#' @param actual A vector containing the actual values for the parameter that is being regressed. 
#' @param SEED An integer representing the random seed to split samples during cross validation.
#' @description Function to run a random forest regression cross validation pipeline. Should not be run by itself. Currently a work in progress. 
#' 
rf_regression_pipeline <- function(feature_table, actual, SEED=1995, sampling=NULL ){
  
  message("Running random forest regression pipeline")
  message("Make sure that the actual variable is a numeric class")
  
  message("Splitting data into test and train")
  set.seed(SEED)
  #setting indexs for training and test data
  train_index <- createDataPartition(actual, p=.8, list=FALSE)
  train_features <- feature_table[train_index,]
  test_features <- feature_table[-train_index,]
  train_actual <- actual[train_index]
  test_actual <- actual[-train_index]
  
  ### okay data is split
  folds <- 5
  repeats <- 10
  mtry_num <- 7
  
  set.seed(SEED)
  cvIndex <- createMultiFolds(train_actual, folds, times=repeats)
  set.seed(SEED)
  seeds_len <- (repeats*folds) + 1
  seeds <- vector(mode="list", length=(seeds_len))
  for(i in 1:(seeds_len-1)){
    seeds[[i]] <- sample.int(n=1000, mtry_num)
  }
  seeds[seeds_len] <- sample.int(1000, 1)
  cv <- trainControl(method="repeatedcv",
                     number=folds,
                     index=cvIndex,
                     returnResamp = "final",
                     classProbs = FALSE,
                     indexFinal = NULL,
                     savePredictions = TRUE,
                     seeds = seeds)
  cv$sampling <- sampling
  ### set up grind to search for best mtry
  
  ### set up mtry values to test
  n_features <- ncol(feature_table)
  
  #set up 6 different mtry values to test
  mtry <- floor(seq(1, n_features/3, length=6))
  mtry <- mtry[mtry <= n_features]
  mtry <- c(mtry, 2)
  grid <- expand.grid(mtry = mtry)
  message(mtry)
  message("Training model")
  set.seed(SEED)
  trained_model <- train(train_features, train_actual,
                         method="rf",
                         trControl=cv,
                         metric="RMSE",
                         tuneGrid=grid,
                         ntree=2001,
                         importance=TRUE)
  #get RMSE value over repeats for the best mtry parameter
  cv_best <- getTrainPerf(trained_model)
  cv_results <- trained_model$results
  
  #get RMSE value for best model on test daata
  predictions <- predict(trained_model, test_features, type="raw")
  test_results <- postResample(test_actual, predictions)
  
  #get important features
  important_features <- trained_model$finalModel$importance
  
  results <- list(cv_best, cv_results, test_results, important_features,
                  trained_model)
  return(results)
}


#' Run_RF_Pipeline
#' 
#' @param feature_table The feature table that contains the information to be input into the random forest classifier. Note that this 
#' table should not include information about the classes that are being predicted.
#' @param classes A vector that represents the classes that each sample (row) in the feature table represents. This can be coded as
#' Case (level 1 factor) and control (level 2 factor). Make sure the factor levels are correct with using AUPRC or results will not always be correct.
#' @param metric A string that indicates whether the pipeline should use AUROC or AUPRC. For AUROC set metric="ROC". For AUPRC set metric="PR". 
#' Defaults to "ROC".
#' @param ntree An integer that represents the number of trees that you want to use during randoom forest construction. Defaults to 1001.
#' @param nmtry An integer representing the number of different mtry values that you want to test during cross validation. The values of mtry to test
#' is calculated as follows:  mtry <- round(seq(1, number_of_features/3, length=nmtry)). Defaults to 7.
#' @param sampling A string indicating that type of sampling that should be done incase of inbalanced class designs. Options include: "up", "down" "SMOTE" and NULL.
#' @param nfolds An integer that represents the number of folds to used during cross validation. Defaults to 3.
#' @param ncrossrepeats An integer that represents the number of times to run cross validation on k folds. Defaults to 10. 
#' @param pro The proporition of samples that should be used for training versus testing during cross validation. Defaults to 0.8
#' @param SEED The random seed used to split the samples during cross validation. Defaults to 1995.
#' @param path A string representing the PATH were output files should be saved.
#' @param repeats The number of times data should be split into testing and cross-validation datasets. 
#' @param list_of_seeds A vector containing a number of seeds that should be equal to the number of repeats.
#' @return This function returns a list with the following characteristics: 
#' "Object[[1]] contains all the median cross validation AUCS from each data split using the best mtry value"
#' "Object[[2]] contains all the test AUC values from each data split"
#' "Object[[3]] contains all the tested mtry values and the median ROC for each from each data split"
#' "Object[[4]] contains the list of important features from the best model selected from each data split"
#' "Object[[5]] contains each caret random forest model from each data split"
#' "This function will also write a csv with cross validation AUCS and test AUCS, to the given path as well as an RDS file that contains the resulting object from this function"

Run_RF_Pipeline <- function(feature_table, classes, metric="ROC", sampling=NULL, 
                           repeats=10, path, nmtry=6, ntree=1001, 
                           nfolds=3, ncrossrepeats=10, pro=0.8, list_of_seeds){
  start_time <- Sys.time()
  cv_aucs <- c()
  test_aucs <- c()
  results_total <- list()
  important_features <- list()
  models <- list()
  #run rf for the number of repeats given
  for(i in 1:repeats){
    message("On Round ",i)
    int_res <- rf_classification_pipeline(feature_table = feature_table, classes = classes, 
                           metric = metric, sampling = sampling, 
                           SEED = list_of_seeds[i],
                           nmtry= nmtry,
                           ntree=ntree,
                           nfolds=nfolds,
                           ncrossrepeats=ncrossrepeats,
                           pro = pro)
    cv_aucs <- c(cv_aucs, int_res[[1]])
    test_aucs <- c(test_aucs, int_res[[2]])
    list_names <- paste0("t",i)
    results_total[[list_names]] <- int_res[[3]]
    important_features[[list_names]] <- int_res[[4]]
    colnames(important_features[[list_names]]) <- gsub("^",list_names,colnames(important_features[[list_names]]))
    models[[list_names]] <- int_res[[5]]
  }
  #take data save make one master list to save it all as an rds... (given the path)
  #take the test_auc and cv_auc and write it out into a csv
  #take the results list rbind thw whole thing and write it out as a csv
  #take the important features list rbind the whole thing and write it out as a csv
  #don't do anything with models just return it in the master list and save the rds as above
  
  #master ret list
  ret_list <- list(cv_aucs, test_aucs, results_total, important_features,
                   models)
  
  #auc dataframe
  auc_data <- data.frame(cv_auc = cv_aucs, test_auc = test_aucs)
  write.csv(auc_data, file=paste0(path,"_aucs.csv"))
  
  #take results list and rbind it
  cv_results <- bind_rows(results_total)
  write.csv(cv_results, file=paste0(path,"_cv_results.csv"))
  
  ret_features <- do.call(cbind, important_features)
  write.csv(ret_features, file=paste0(path,"_imprt_feats.csv"))

  #save rds
  saveRDS(ret_list, file=paste0(path,"_masterlist.rds"))
  endtime <- Sys.time()
  total_time <- endtime - start_time
  message(total_time)
  return(ret_list)
}


#' get_random_rf_results
#' 
#' @param feature_table The feature table that contains the information to be input into the random forest classifier. Note that this 
#' table should not include information about the classes that are being predicted.
#' @param list_of_scrambles A list of vectors that is equal to the number of repeats that cross validation should be run. Each item within this list should
#' contain a random scrambling of the classes set to each sample. 
#' @param metric A string that indicates whether the pipeline should use AUROC or AUPRC. For AUROC set metric="ROC". For AUPRC set metric="PR". 
#' Defaults to "ROC".
#' @param ntree An integer that represents the number of trees that you want to use during randoom forest construction. Defaults to 1001.
#' @param nmtry An integer representing the number of different mtry values that you want to test during cross validation. The values of mtry to test
#' is calculated as follows:  mtry <- round(seq(1, number_of_features/3, length=nmtry)). Defaults to 7.
#' @param sampling A string indicating that type of sampling that should be done incase of inbalanced class designs. Options include: "up", "down" "SMOTE" and NULL.
#' @param nfolds An integer that represents the number of folds to used during cross validation. Defaults to 3.
#' @param ncrossrepeats An integer that represents the number of times to run cross validation on k folds. Defaults to 10. 
#' @param pro The proporition of samples that should be used for training versus testing during cross validation. Defaults to 0.8
#' @param SEED The random seed used to split the samples during cross validation. Defaults to 1995.
#' @param path A string representing the PATH were output files should be saved.
#' @param repeats The number of times data should be split into testing and cross-validation datasets. 
#' @param list_of_seeds A vector containing a number of seeds that should be equal to the number of repeats.
#' @return This function returns a list with the following characteristics: 
#' "Object[[1]] contains all the median cross validation AUCS from each data split using the best mtry value"
#' "Object[[2]] contains all the test AUC values from each data split"
#' "Object[[3]] contains all the tested mtry values and the median ROC for each from each data split"
#' "Object[[4]] contains the list of important features from the best model selected from each data split"
#' "Object[[5]] contains each caret random forest model from each data split"
#' "This function will also write a csv with cross validation AUCS and test AUCS, to the given path as well as an RDS file that contains the resulting object from this function"
#' @description Runs a similar pipleline as Run_RF_Pipeline however takes in random scramblings of the class assignments for each sample (row in feature table). The results from this
#' function can act as a null distrubition to compare models against. 
#' 
get_random_rf_results <- function(feature_table, list_of_scrambles, metric="ROC", sampling=NULL, 
                                  repeats=10, path, nmtry=6, ntree=1001, 
                                  nfolds=3, ncrossrepeats=10, pro=0.8, list_of_seeds){
  message("Number of repeats must be equal to number of scramble lists")
                                      
  start_time <- Sys.time()
  cv_aucs <- c()
  test_aucs <- c()
  results_total <- list()
  important_features <- list()
  models <- list()
  #run rf for the number of repeats given
  for(i in 1:repeats){
    message("On Round ",i)
    int_res <- rf_classification_pipeline(feature_table = feature_table, 
                                          classes = list_of_scrambles[[i]], 
                                          metric = metric, sampling = sampling, 
                                          SEED = list_of_seeds[i],
                                          nmtry= nmtry,
                                          ntree=ntree,
                                          nfolds=nfolds,
                                          ncrossrepeats=ncrossrepeats,
                                          pro = pro)
    cv_aucs <- c(cv_aucs, int_res[[1]])
    test_aucs <- c(test_aucs, int_res[[2]])
    list_names <- paste0("t",i)
    results_total[[list_names]] <- int_res[[3]]
    important_features[[list_names]] <- int_res[[4]]
    colnames(important_features[[list_names]]) <- gsub("^",list_names,colnames(important_features[[list_names]]))
    models[[list_names]] <- int_res[[5]]
  }
  #take data save make one master list to save it all as an rds... (given the path)
  #take the test_auc and cv_auc and write it out into a csv
  #take the results list rbind thw whole thing and write it out as a csv
  #take the important features list rbind the whole thing and write it out as a csv
  #don't do anything with models just return it in the master list and save the rds as above
  
  #master ret list
  ret_list <- list(cv_aucs, test_aucs, results_total, important_features,
                   models)
  
  #auc dataframe
  auc_data <- data.frame(cv_auc = cv_aucs, test_auc = test_aucs)
  write.csv(auc_data, file=paste0(path,"_aucs.csv"))
  
  #take results list and rbind it
  cv_results <- bind_rows(results_total)
  write.csv(cv_results, file=paste0(path,"_cv_results.csv"))
  
  ret_features <- do.call(cbind, important_features)
  write.csv(ret_features, file=paste0(path,"_imprt_feats.csv"))
  
  #save rds
  saveRDS(ret_list, file=paste0(path,"_masterlist.rds"))
  endtime <- Sys.time()
  total_time <- endtime - start_time
  message(total_time)
  return(ret_list)
}

#' Run_RF_Regression_Pipeline
#' 
#' @param feature_table A feature table containing the samples (rows) and the features (columns) to run random forest regression on. Note that this table should
#' not include the value that is trying to be predicted
#' @param actual A vector containing the actual values for the value that this trying to be predectied.
#' @param sampling The sampling technique to use during cross validation. Defaults to NULL.
#' @param repeats The number of data splits that should occur between testing data and cross validation data.
#' @param path The path that the output should be saved to.
#' @param list_of_seeds A list of seeds equal to the length of repeats that is used for each random data split.
#' @return Returns a list containing the following:
#' "Object[[1]] contains all the median cross validation RMSE from each data split using the best mtry value"
#' "Object[[2]] contains all the test RMSE values from each data split"
#' "Object[[3]] contains all the tested mtry values and the median RMSE from each from each data split"
#' "Object[[4]] contains the list of important features from the best model selected from each data split"
#' "Object[[5]] contains each caret random forest model from each data split"
#' "This function will also write a csv with cross validation RMSE and test RMSE, to the given path as well as an RDS file that contains the resulting object from this function"

Run_RF_Regression_Pipeline<- function(feature_table, actual, sampling=NULL,
                                  repeats, path, list_of_seeds){
  start_time <- Sys.time()
  cv_best <- list()
  cv_results <-list()
  test_results <- list()
  important_features <- list()
  models <- list()
  
  for(i in 1:repeats){
    message("On Round ", i)
    int_res <- rf_regression_pipeline(feature_table, actual,
                                      sampling=sampling, 
                                      SEED=list_of_seeds[i])
    list_name <- paste0("t",i)
    cv_best[[list_name]] <- int_res[[1]]
    cv_results[[list_name]] <- int_res[[2]]
    test_results[[list_name]] <- int_res[[3]]
    important_features[[list_name]] <- int_res[[4]]
    colnames(important_features[[list_name]]) <- 
      gsub("^",list_name,colnames(important_features[[list_name]]))
    models[[list_name]] <- int_res[[5]]
  }
  

  cv_best_data <- bind_rows(cv_best)
  write.csv(cv_best_data, file=paste0(path, "_cv_best.csv"))
  
  cv_results_data <- bind_rows(cv_results)
  write.csv(cv_results_data, file=paste0(path, "_cv_results.csv"))
  
  test_results <- bind_rows(test_results)
  write.csv(test_results, file=paste0(path,"_test_results.csv"))
  
  impt_feats <- do.call(cbind, important_features)
  write.csv(impt_feats, file=paste0(path,"_impt_feats.csv"))

  
  
  end_time <- Sys.time()
  totaltime <- end_time - start_time
  ret_list <- list(cv_best, test_results, cv_results,
                   important_features, models, totaltime)
  saveRDS(ret_list, file = paste0(path,"_master_list.rds"))  
  return(ret_list)
}


#' Calc_mean_accuracy_decrease
#' 
#' @param impt_feat_list A list of dataframes containing the list of important features generated from running Run_RF_Pipeline.
#' @return Returns a dataframe containing the mean, sd, max, and min decrease in accuracy for each feature over all of the random forest models that worked best during cross
#' validation.
#' 
Calc_mean_accuray_decrease <- function(impt_feat_list){
  pass=TRUE
  #make sure colnames for each impt_feat_table matches up
  for(j in 1:length(impt_feat_list)){
    if(!all.equal(rownames(impt_feat_list[[1]]), rownames(impt_feat_list[[i]]))){
      pass=FALSE
    }
  }
  
  if(pass==FALSE){
    message("rownames don't match")
    return(NULL)
  }
  
  ret_data_list <- list()
  for(i in 1:length(impt_feat_list)){
    prefix <- paste0("t",i)
    message(prefix)
    acc <- impt_feat_list[[i]][,3]
    ret_data_list[[prefix]] <- acc
  }
  
  
  ret_data_frame <- do.call(cbind, ret_data_list)
  test <- as.data.frame(cbind(ret_data_frame, t(apply(ret_data_frame, 1, Stats))))
  ret_final_frame <- test[order(test$Mean, decreasing=TRUE), ]
  return(ret_final_frame)
}

#' Stats
#' 
#' @param x A vector of values
#' @return Retruns the mean, sd, min and max of X.
Stats <- function(x){
  Mean <- mean(x, na.rm=TRUE)
  SD <- sd(x, na.rm=TRUE)
  Min <- min(x, na.rm=TRUE)
  Max <- max(x, na.rm=TRUE)
  return(c(Mean=Mean, SD=SD, Min=Min, Max=Max))
}



#' Generate_ROC_curve
#' 
#' @param RF_models A list containing caret RF_models. If using Run_RF_Pipeline this information will be contained within the fifth index of the returned list.
#' @param dataset The feature_table that was input into Run_RF_Pipeline
#' @param labels The actual labels for each sample(row) within the dataset
#' @param title The title of the plot
#' @return Returns a ggplot object that plots the testing AUC of each of the best cross-validated random forest models generated when data is split into 
#' test and cross-validation datasets. Note that the redline represents the mean value for the senstivity and specificity at each step tested during ROC calculation

generate_ROC_curve <- function(RF_models, dataset, labels, title){

  
  AUC_data <- vector()
  ROC_data <- list()
  ROC_curve_data <- list()
  
  ROC_sens <- list()
  
  ROC_specs <- list()
  
  for(i in 1:length(RF_models)){
    
    prefix <- paste0("t",i)
    #grab the samples that the model was trained on
    training_samples <- rownames(RF_models[[i]]$trainingData)
    
    
    #get the samples to predict data from
    prediction_set <- dataset[!rownames(dataset) %in% training_samples,]
    prediction_set_labels <- labels[!rownames(labels) %in% training_samples,,drop=F]
    message(prediction_set_labels$classes)
    message("getting hold-out data from each model training session for test validation")
    
    #make predictions on this dataset using the final model
    
    
    predicts <- predict(RF_models[[i]], prediction_set, type="prob")
    message("making predictions")
    message(length(prediction_set_labels))
    roc_data <- pROC::roc(prediction_set_labels$classes, predicts[,1], levels=c("Control", "Case"))
    
    sens <- roc_data$sensitivities
    specs <- roc_data$specificities
    
    indexs_to_keep <- floor(seq(1, length(sens), length=40))
    ROC_sens[[prefix]] <- sens[indexs_to_keep]
    ROC_specs[[prefix]] <- specs[indexs_to_keep]
    AUC_data <- c(AUC_data, roc_data$auc)
    if(i==1){
      plot(roc_data, xlim=c(1,0), ylim=c(0,1), col="red")
    }else{
      plot(roc_data,add=T, xlim=c(1,0), ylim=c(0,1))
    }
  }
  
  ROC_curve_data[[1]] <- do.call(cbind, ROC_sens)
  ROC_curve_data[[2]] <- do.call(cbind, ROC_specs)
  
  #turn data into long form
  
  SENS_melt <- melt(ROC_curve_data[[1]])
  SPEC_melt <- melt(ROC_curve_data[[2]])
  
  if(all.equal(SENS_melt$Var1, SPEC_melt$Var1) & all.equal(SENS_melt$Var2, SPEC_melt$Var2)){
    SENS_melt$Value2 <- SPEC_melt$value
  }else(
    message("values dont match")
  )
  #alright so we have the sens and spec for each thing so lets generate curves from each column
  
  #calc mean values this part i'm not 100% sure how to do....
  
  
  SENs_values <- as.data.frame(cbind(ROC_curve_data[[1]], t(apply(ROC_curve_data[[1]], 1, Stats))))
  SPEC_values <- as.data.frame(cbind(ROC_curve_data[[2]], t(apply(ROC_curve_data[[2]], 1, Stats))))
  
  mean_values <- data.frame(Sens= SENs_values$Mean,
                            Specs= SPEC_values$Mean)
  
  AUC_label <- paste("Mean AUC", round(mean(AUC_data), digits=5))
  plot <- ggplot() + geom_path(data=SENS_melt, mapping=aes(x=Value2, y=value, group=Var2), alpha=0.05) + xlab("Specificity") +
    ylab("Sensitivity") + scale_x_reverse() + scale_color_manual(values = COLORS) + theme(legend.position = "none") + 
    geom_abline(intercept=1, slope=1) + geom_line(data=mean_values, aes(x=Specs, y=Sens), color="Red") + ggtitle(title) +
    annotate(geom="text", y=0, x=0.2, label=AUC_label, color="red")
  
  return(plot)
}




