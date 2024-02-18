import time

import numpy
import pandas

import sklearn
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import xgboost

import pickle

import warnings
warnings.filterwarnings('ignore')

input_features = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']

def read_data(train_file_path: str = './data/processed/train_test_data/train_data.feather',
              test_file_path: str = './data/processed/train_test_data/test_data.feather'):
    
    print("Loading files...")
    train_df = pandas.read_feather(train_file_path)
    test_df = pandas.read_feather(test_file_path)

    print('Loaded Files...\n')
    return train_df, test_df

def run_model(classifier, training_data, testing_data,
              input_features, output_feature = "TX_FRAUD", scale = True):
    
    training_data_modeling = training_data.copy()
    testing_data_modeling = testing_data.copy()

    if scale:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(training_data_modeling[input_features])
        training_data_modeling[input_features] = scaler.transform(training_data_modeling[input_features])
        testing_data_modeling[input_features] = scaler.transform(testing_data_modeling[input_features])

    print('Training Model...')
    start_time = time.time()
    classifier.fit(training_data_modeling[input_features], training_data_modeling[output_feature])
    training_execution_time = time.time() - start_time

    print('Getting Predictions...')
    # We then get the predictions on the training and test data using the `predict_proba` method
    # The predictions are returned as a numpy array, that provides the probability of fraud for each transaction
    start_time = time.time()
    predictions_test = classifier.predict_proba(testing_data_modeling[input_features])[:,1]
    prediction_execution_time = time.time() - start_time
    
    predictions_train = classifier.predict_proba(training_data_modeling[input_features])[:,1]

    # The result is returned as a dictionary containing the fitted models, 
    # and the predictions on the training and test sets
    model_and_predictions_dictionary = {'classifier': classifier,
                                        'predictions_test': predictions_test,
                                        'predictions_train': predictions_train,
                                        'training_execution_time': training_execution_time,
                                        'prediction_execution_time': prediction_execution_time
                                       }
        
    return model_and_predictions_dictionary

def performance_assessment_model_collection(fitted_models_and_predictions_dictionary, transactions_df,
                                            type_set = 'test', top_k_list = [100]):
        
        def card_precision_top_k_day(df_day, top_k):
        
            # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID, 
            # and sorts by decreasing order of fraudulent prediction
            df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)

            # Get the top k most suspicious cards
            df_day_top_k=df_day.head(top_k)
            list_detected_compromised_cards=list(df_day_top_k[df_day_top_k.TX_FRAUD==1].CUSTOMER_ID)

            # Compute precision top k
            card_precision_top_k = len(list_detected_compromised_cards) / top_k

            return list_detected_compromised_cards, card_precision_top_k

        def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards = True):
             # Sort days by increasing order
            list_days=list(predictions_df['TX_TIME_DAYS'].unique())
            list_days.sort()

            # At first, the list of detected compromised cards is empty
            list_detected_compromised_cards = []
    
            card_precision_top_k_per_day_list = []
            nb_compromised_cards_per_day = []

            # For each day, compute precision top k
            for day in list_days:
                df_day = predictions_df[predictions_df['TX_TIME_DAYS']==day]
                df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]

                # Let us remove detected compromised cards from the set of daily transactions
                df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards)==False]

                nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD==1].CUSTOMER_ID.unique()))

                detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day,top_k)

                card_precision_top_k_per_day_list.append(card_precision_top_k)

            # Let us update the list of detected compromised cards
            if remove_detected_compromised_cards:
                list_detected_compromised_cards.extend(detected_compromised_cards)

            # Compute the mean
            mean_card_precision_top_k = numpy.array(card_precision_top_k_per_day_list).mean()
        
            # Returns precision top k per day as a list, and resulting mean
            return nb_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k
             
        def performance_assessment(predictions_df, output_feature = 'TX_FRAUD',
                           prediction_feature = 'predictions', top_k_list = top_k_list,
                           rounded=True):
              
              AUC_ROC = sklearn.metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
              AP = sklearn.metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
              
              performances = pandas.DataFrame([[AUC_ROC, AP]],
                                              columns = ['AUC ROC','Average precision'])
              
              for top_k in top_k_list:
                   
                   _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
                   performances['Card Precision@'+str(top_k)]=mean_card_precision_top_k
                   
              if rounded:
                   performances = performances.round(3)
                   
              return performances

        def execution_times_model_collection(fitted_models_and_predictions_dictionary):
             
             execution_times = pandas.DataFrame()
             
             for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
                  
                  execution_times_model = pandas.DataFrame()
                  execution_times_model['Training execution time']=[model_and_predictions['training_execution_time']]
                  execution_times_model['Prediction execution time']=[model_and_predictions['prediction_execution_time']]
                  execution_times_model.index=[classifier_name]
                  
                  execution_times = pandas.concat([execution_times, execution_times_model], ignore_index=False)
                  
             return execution_times
        
        performances = pandas.DataFrame()
        execution_times = execution_times_model_collection(fitted_models_and_predictions_dictionary = fitted_models_and_predictions_dictionary)

        for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        
            predictions_df = transactions_df

            predictions_df['predictions'] = model_and_predictions['predictions_'+type_set]

            performances_model = performance_assessment(predictions_df, output_feature='TX_FRAUD', 
                                                       prediction_feature='predictions', top_k_list=top_k_list)
            performances_model.index = [classifier_name]

            performances = pandas.concat([performances, performances_model], ignore_index=False)

        return performances, execution_times

def run_time(start_time):
    print("Time to Run File: {0:.2}s".format(time.time() - start_time))

if __name__ == '__main__':
    start_time = time.time()

    train_df, test_df = read_data()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    classifiers_dictionary = {
         'logistic_regression': sklearn.linear_model.LogisticRegression(random_state = 42),                        # Runtime: ~1 Minutes
         'decision_tree_depth_2': sklearn.tree.DecisionTreeClassifier(max_depth= 2, random_state = 42),            # Runtime: ~1 Minutes
        #  'Decision tree - Unlimited Depth': sklearn.tree.DecisionTreeClassifier(random_state = 42),              # Runtime: ~30 Minutes
        #  'Random forest': sklearn.ensemble.RandomForestClassifier(random_state = 42, n_jobs = -1),               # Runtime: 1.5 Hours
         'xgboost': xgboost.XGBClassifier(random_state = 42, n_jobs = -1)                                          # Runtime: ~2 Minutes
                              }
    
    fitted_models_and_predictions_dictionary = {}

    for classifier_name in classifiers_dictionary:
        
        models_and_predictions_dictionary = run_model(classifier = classifiers_dictionary[classifier_name], training_data = train_df, testing_data = test_df,
                                                      input_features = input_features)

        print('Fininshed Training Model...\n')
        
        fitted_models_and_predictions_dictionary[classifier_name] = models_and_predictions_dictionary

        model_path = "models/"
        pickle.dump(classifier_name, open(model_path + classifier_name + '.pkl', 'wb'))


    print('Assessing Performance...')
    # performances on test set
    test_df_performances, execution_time_performance_test = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, test_df,
                                                                                                    type_set = 'test', top_k_list = [1000])

    # performances on training set
    train_df_performances, _ = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, train_df,
                                                                       type_set = 'train', top_k_list = [1000])
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    run_time(start_time)

    print('Fininshed Running File') # 2.5 Hours

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
