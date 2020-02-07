"""Find the similarity between consumption behaviors of sellers
The trained model is used to predict the probability of selling for each collector.
The result is written to table likelihood on SQL
"""
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from connLocalDB import connDB

class RfcCV(object):
    """
    Random Forest Classifer Cross Validation Class
    Methods:
        train_model: update best_parameter and best_model
        evaluate: evaluate on test-dataset, return scores
    """
    def __init__(self ):
        # parameters for grid search in the cross-validation
        self.param_grid = {
            'n_estimators': [600,800],
            'max_depth':[3,4],
            'min_samples_split':[4,6],
            'class_weight':[{0:1,1:2},{0:1,1:4}],
            'random_state':[42]
        }
        self.best_model = None
        self.best_params = {}

        # Calculate and print the total rounds to run for cross-validation
        num_test = 1
        for key, val in enumerate(self.param_grid):
            num_test *= len(val)
        print("\nStart to find the best parameters for random forest classifier using cross-validation.")
        print("Total test to run: "+str(num_test))

    def train_model(self, train_features, train_labels):
        # Train the model and update self.best_model
        rfc = RandomForestClassifier()
        # Instantiate the grid search model
        scorer = make_scorer(fbeta_score, beta=1)
        grid_search = GridSearchCV(estimator = rfc, param_grid = self.param_grid,
                                  cv = 8, n_jobs = -1, verbose = 1,scoring=scorer)   # Using k-folds with cv=10
        grid_search.fit(train_features, train_labels)
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        print('Best parameters for random forest classifier is: ')
        print(str(self.best_params))

    def evaluate(self, test_features, test_labels):
        """
        Evaluate the model performance including
        Return: ROC curve, accuracy, recall, precision, and F1
        """
        def plot_roc(test_labels):
            """
            Plot ROC curve
            """
            y_pred_grd = self.best_model.predict_proba(test_features)[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(test_labels, y_pred_grd)
            plt.plot(fpr_rf,tpr_rf)
            plt.show()
            print("fpr_rf, tpr_rf")
            print(fpr_rf)
            print(tpr_rf)

        predictions = self.best_model.predict(test_features)
        plot_roc(test_labels, predictions)
        recallScore = sklearn.metrics.recall_score(test_labels, predictions)
        f1lScore = sklearn.metrics.f1_score(test_labels, predictions)

        # Accuracy score
        errors = abs(predictions - test_labels)
        mape = 100 * np.sum(errors)/np.size(errors,0)
        accuracy = 100 - mape

        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy, recallScore, f1lScore


def get_data():
    """
    Grep data from local SQL database
    :return: DataFrame containing train and test data
    """
    engine, _ = connDB()
    features_query = """
        SELECT * From features;
    """
    # Grep and process data
    features = pd.read_sql_query(features_query, engine, index_col='userId')
    features = features.drop(columns='index')
    features = features.fillna(value=0)
    features['sumFeatures'] = features.iloc[:,:-1].sum(axis=1)

    # Remove sellers from Ebay
    print("Total sellers before removing ebay-migrant is "+ str(features[features['selling']==1].shape[0]))
    features = features[np.logical_or(features['selling']==0, np.logical_and(features['selling']==1,features['sumFeatures']>=3))]
    print("Total sellers after removing ebay-migrant is"+ str(features[features['selling']==1].shape[0]))
    features = features.drop('sumFeatures',axis=1)

    # Recent 3-month records for prediction
    features_3month_query = """
    select * 
    from featuresrecent3month
    """
    features_recent_3month = pd.read_sql_query(features_3month_query, engine, index_col='userId')
    features_recent_3month = features_recent_3month.drop(columns='index')
    return features, features_recent_3month


def write_result(result):
    """
    Grep data from local SQL database
    :return: DataFrame containing train and test data
    """
    engine, _ = connDB()
    result.to_sql('likelihood', engine, if_exists='replace')


def resample_training(X_train, y_train, oversample=False):
    """
    oversample: if true use the under-sampling+SMOLE, else only use under-sampling only
    :return: the resampled training and test sets.
    """
    train_matrix = X_train.join(y_train)
    train_resampled_neg = train_matrix[train_matrix['selling'] == 0].sample(frac=0.01, random_state=42)
    train_resampled_pos = train_matrix[train_matrix['selling'] == 1]
    print("Number of sellers in training set is", str(train_resampled_pos.shape[0]), "and", str(
        train_resampled_neg.shape[0]), "are not.")

    X_train_resampled = train_resampled_neg.append(train_resampled_pos).iloc[:, :-1]
    y_train_resampled = train_resampled_neg.append(train_resampled_pos).iloc[:, -1]
    if not oversample:
        return X_train_resampled, y_train_resampled
    else:
        X_train_resampled_oversampled, y_train_resampled_oversampled = SMOTE(kind='borderline1').fit_sample(\
                            X_train_resampled, y_train_resampled)
        return X_train_resampled_oversampled, y_train_resampled_oversampled

def main():
    """
     Train random forest classifier
     1. Create freatures
     2. Split dataset to train/test
     3. Resample data (both undersampling and oversampling)
     4. Train the model
     5. Show test results
     """
    # Split training set and test set.
    # Training set is undersanmpled and oversampled. Check function 'resampleTraining'
    features,features_recent_3month = get_data()   # Features (not split yet) from SQL
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features.iloc[:,1:-1], features.iloc[:, -1], test_size=0.30, random_state=32)
    X_train_resampled_oversampled, y_train_resampled_oversampled = resample_training(X_train, y_train, oversample=True)
    print("Number of sellers in test set is", str(sum(y_test)), "and", str(sum(y_test == 0)), "are not.")

    # Train random_forest_classifer by grid_search
    rfc = RfcCV()
    print("Training model using undersample + oversample")
    rfc.train_model(X_train_resampled_oversampled, y_train_resampled_oversampled)
    accu, recallscore, f1Score = rfc.evaluate(X_test, y_test)

    # After model train and test, predict the probability of selling in next month
    print(str(accu) + ":" + str(recallscore) + ":" + str(f1Score))
    y_pred_grd = rfc.best_model.predict_proba(features_recent_3month.iloc[:,1:])
    features_recent_3month['likelihood']=y_pred_grd[:,1]
    print('Weight for each features:')
    print(str(rfc.best_model.feature_importances_))

    # Write result to sql database
    write_result(features_recent_3month['likelihood'])


if __name__ == "__main__":
    sys.stdout = open('output.txt', 'wt')
    main()
