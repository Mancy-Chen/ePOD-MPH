# Mancy Chen 07/06/2023
#########################   1. Load the necessary packages or libraries   #############################################
# General packages
import sys
sys.path.append('/scratch/mchen/miniconda3/lib/python3.10/site-packages')
import random
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
import os
import time
from datetime import datetime
from collections import defaultdict
import glob
import shap
import statsmodels.api as sm
import seaborn as sns
# sklearn
from sklearn import model_selection, linear_model, metrics, pipeline, decomposition, \
    feature_selection, ensemble, cluster, preprocessing
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, ConfusionMatrixDisplay, \
    balanced_accuracy_score, make_scorer, accuracy_score,  classification_report, precision_recall_curve, \
    precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, StratifiedKFold, cross_validate, GridSearchCV, \
    cross_val_score, cross_val_predict, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit, learning_curve
from sklearn.feature_selection import RFE, SelectKBest, f_classif, RFECV, VarianceThreshold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
# ComBat, bayes_opt, pmtool, scipy
from neurocombat_sklearn import CombatModel
from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel
print(__doc__)
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from pmtool.AnalysisBox import AnalysisBox
from pmtool.ToolBox import ToolBox
from scipy.stats import ttest_ind, pearsonr, binomtest, spearmanr, kendalltau, pointbiserialr, median_abs_deviation
from scipy.spatial import distance
from pygam import LinearGAM, s

##################################         Whole MPH Group Analysis        ###############################################
##################################         2. Load the input data        ###############################################
# Tier 1 and 2
X = pd.read_csv('enter_your_X_csv_file_path')
y = pd.read_csv('enter_your_y_csv_file_path', header=None)

# Define the features numbers
 # Tier 1
# T1_features = 10  # 10 or 320
# MRI_features = 14   # 14 or 380
# all_features = 14 # 14 or 380 or 382
# selected_features = 14 # classification: 6(35-39), 7 (44-47) or 380 (no selection)

# # Tier 2
T1_features = 320   # 10 or 320
MRI_features = 380   # 14 or 380
all_features = 380 # 14 or 380 or 382
selected_features = 7 # classification: 6(35-39), 7 (44-47) or 380 (no selection)

Covariate_number = 2 # 2
non_MRI_number = 2 # 2 or 4
# Set the random seed
random_seed = 42
np.random.seed(random_seed)
max_display = 10 # number of features to display in the heatmap plots

# Store the feature names
feature_names = X.iloc[:, : -Covariate_number].columns.to_list()
print(" Feature's total number: ", len(feature_names), ';', "\n",
      "Selected feature's number: ", selected_features,';',  "\n",
      "Covariate number: ", Covariate_number, "\n")
# Convert to numpy arrays
X = np.float64(X.to_numpy())
y = np.ravel(y).astype(int) # Classification
print(' MPH_X shape: ', X.shape, ', dtype:', X.dtype, ';', "\n",
      'MPH_y shape: ', y.shape, ', dtype:', y.dtype, ';', "\n",
      )

# Create the output folder for the images
file_path = 'enter_your_output_folder_path'
# Open the file in "write" mode, creating it if it doesn't exist
if not os.path.exists(file_path):
    os.makedirs(file_path)
    print(f" New directory created.")
else:
    print(f" This directory already exists.")

# Define the range of T1 and DTI features for harmonization
n_samples = X.shape[0]
n_features = X.shape[1] - Covariate_number
T1_features_start = 0
MRI_features_start = 0
MRI_features_end = MRI_features
T1_features_end = T1_features
DTI_features_start = T1_features_end
DTI_features_end = MRI_features
non_MRI_features_start = MRI_features
non_MRI_features_end = X.shape[1] - Covariate_number
All_features_start = 0
All_features_end = X.shape[1] - Covariate_number
MRI_features_range = range(T1_features_start, MRI_features)
non_MRI_features_range = range(non_MRI_features_start, non_MRI_features_end)
covariates_range = range(non_MRI_features_end, X.shape[1])
print( 'n_samples: ', n_samples, '; n_features: ', n_features, ';',"\n",
       'T1 features range: ', T1_features_start, '-', T1_features_end, ';',"\n",
       'DTI features range: ', DTI_features_start, '-', DTI_features_end, ';',"\n",
       'MRI features range: ', MRI_features_start, '-', MRI_features_end, ';',"\n",
       'Non-MRI features range: ', non_MRI_features_start, '-', non_MRI_features_end, ';',"\n",
       'All features range: ', All_features_start, '-', All_features_end, ';',"\n",)

########################################     Tier 2 analysis     #######################################################
####################################      3. Define the models      ####################################################
# Custom transformer for feature selection

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, centering=True, scaling=True):
        self.centering = centering
        self.scaling = scaling
        self.scaler = RobustScaler(with_centering=self.centering, with_scaling=self.scaling)
    def fit(self, X, y=None, **kwargs):
        # Fit the scaler on the features (excluding the last two columns which are covariates)
        self.scaler.fit(X[:, : - Covariate_number])
        return self
    def transform(self, X, **kwargs):
        # Scale the features (excluding the last two columns which are covariates)
        X_scaled = self.scaler.transform(X[:, : - Covariate_number])
        # Return the scaled features WITH the last two columns appended to the end
        return np.hstack([X_scaled, X[:, - non_MRI_number: ]])

class CustomGAMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_index, end_index, lam=1.0):
        self.start_index = start_index
        self.end_index = end_index
        self.lam = lam
        self.gam_models = []
    def fit(self, X, y=None, **fit_params):
        data = X[:, self.start_index:self.end_index]
        covariate = X[:, -1].reshape(-1, 1)
        # Fit a GAM model for each feature
        self.gam_models = []
        for feature_idx in range(data.shape[1]):
            gam = LinearGAM(s(0, lam=self.lam)).fit(covariate, data[:, feature_idx])
            self.gam_models.append(gam)
        return self
    def transform(self, X, **transform_params):
        data = X[:, self.start_index:self.end_index]
        covariate = X[:, -1].reshape(-1, 1)
        # Adjust each feature using its respective GAM model
        adjusted_data = np.column_stack([
            data[:, feature_idx] - model.predict(covariate)
            for feature_idx, model in enumerate(self.gam_models)
        ])
        # Return the scaled features WITHOUT the last columns appended to the end
        return np.hstack([adjusted_data, X[:, non_MRI_features_range]])

class CustomCombatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index
        self.sites = None
        self.discrete_covariates = None
        self.continuous_covariates = None
        self.combat_model = CombatModel()
    def fit(self, X, y = None, **fit_params):
        data = X[:, self.start_index:self.end_index]
        sites = X[:, -2].reshape(-1, 1)
        self.combat_model.fit(data=data, sites=sites, discrete_covariates=None,
                              continuous_covariates=None)
        return self
    def transform(self, X, **transform_params):
        data = X[:, self.start_index:self.end_index]
        sites = X[:, -2].reshape(-1, 1)  # Use sites instead of self.sites
        transformed_data = self.combat_model.transform(data = data, sites = sites, discrete_covariates=None,
                                                       continuous_covariates = None)
        # Return the scaled features WITH the last columns appended to the end
        return np.hstack([transformed_data, X[:, - non_MRI_number: ]])

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select = selected_features, corr_th=0.8):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.corr_th = corr_th
    def fit(self, X, y=None):
        selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select, step=1)
        self.selected_features_indices_ = self.selectNonIntercorrelated(X, y, self.corr_th, selector)
        # Store selected X and y for potential later use
        self.X_selected_ = X[:, self.selected_features_indices_]
        self.y_selected_ = y
        return self
    def transform(self, X):
        return X[:, self.selected_features_indices_]
    def selectNonIntercorrelated(self, X, y, corr_th, selector):
        # Features without missing values
        non_nan_indices = np.all(~np.isnan(X), axis=0)
        X_non_nan = X[:, non_nan_indices]
        # Features with non-zero MAD variance
        mad_values = median_abs_deviation(X_non_nan, axis=0, scale='normal')
        non_zero_var_indices = mad_values > 0.001
        X_non_zero_var = X_non_nan[:, non_zero_var_indices]
        # Check if there are no non-zero MAD features
        if X_non_zero_var.shape[1] == 0:
            raise ValueError("All features have zero MAD")
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_non_zero_var, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)  # set diagonal to zero, to avoid self-correlation
        mean_absolute_corr = np.abs(corr_matrix).mean(axis=0)
        # Identify intercorrelated features
        intercorrelated_features_set = set()
        high_corrs = np.argwhere(np.abs(corr_matrix) > corr_th)
        for i, j in high_corrs:
            if mean_absolute_corr[i] > mean_absolute_corr[j]:
                intercorrelated_features_set.add(i)
            else:
                intercorrelated_features_set.add(j)
        non_intercorrelated_indices = list(set(range(X_non_zero_var.shape[1])) - intercorrelated_features_set)
        # Apply RFE on non-intercorrelated features
        X_train_non_intercorrelated = X_non_zero_var[:, non_intercorrelated_indices]
        # Check if only one feature is left, and if so, skip RFE
        if X_train_non_intercorrelated.shape[1] <= self.n_features_to_select:
            # If the number of features is less than or equal to the desired number, skip RFE
            selected_indices = np.array(non_intercorrelated_indices)  # Select the remaining feature
        else:
            # If there's more than one feature, proceed with RFE as normal
            selector = selector.fit(X_train_non_intercorrelated, y)  # Pass y here
            # Get selected feature indices
            support = selector.get_support()
            selected_indices = np.array(non_intercorrelated_indices)[support]
        return selected_indices

class CustomXGBClassifier(XGBClassifier):
    def __init__(self, random_state=random_seed,
                 eval_metric='auc', # Focusing on true negative group in imbalanced data
                 objective='binary:logistic',
                 **kwargs):
        super().__init__(random_state=random_state,
                         objective=objective,
                         eval_metric=eval_metric,
                         **kwargs)
    def fit(self, X, y, **kwargs):
        # Calculate scale_pos_weight
        class_counts = np.bincount(y.astype(int))
        scale_pos_weight = class_counts[0] / class_counts[1]
        # Set the scale_pos_weight for the current iteration
        self.set_params(scale_pos_weight=scale_pos_weight)
        # Proceed with the normal fit
        return super().fit(X, y, **kwargs)

# Create the cross-validation objects
cv_inner = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=random_seed)
cv_outer = LeaveOneOut()

# Specify the hyperparameter space for BayesSearchCV
pbounds = {
    # Hyperparameters for Scale_Regress
    'Scale__centering': Categorical([True, False]),  # Centering hyperparameter for RobustScaler
    'Scale__scaling': Categorical([True, False]),  # Scaling hyperparameter for RobustScaler
    # Hyperparameters for Harmonization
    'GAMs__lam': Real(1e-6, 1e6, prior='log-uniform'),
    # Hyperparameters for the Selector RFE
    'selector__estimator__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    'selector__estimator__max_depth': Integer(2, 7),
    'selector__estimator__subsample': Real(0.5, 1.0),
    'selector__estimator__colsample_bytree': Real(0.5, 1.0),
    'selector__estimator__min_child_weight': Integer(1, 20),
    'selector__estimator__alpha': Real(0, 2),
    'selector__estimator__lambda': Real(0, 2),
    'selector__estimator__n_estimators': Integer(10, 150),
    'selector__estimator__gamma': Real(0, 10),
    # Hyperparameters for XGBClassifier
    'clf__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    'clf__max_depth': Integer(2, 7),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__min_child_weight': Integer(1, 20),
    'clf__alpha': Real(0, 2),
    'clf__lambda': Real(0, 2),
    'clf__n_estimators': Integer(10, 150),
    'clf__gamma': Real(0, 10)
} # XGBoost hyperparameters

estimator = CustomXGBClassifier()

# Create the custom Covariate Harmonization
ComBat_transformer = CustomCombatTransformer(T1_features_start, DTI_features_end)
GAM_transformer = CustomGAMTransformer(T1_features_start, DTI_features_end)

# Create a pipeline
pipeline = Pipeline([
    ('Scale', CustomScaler()),
    ('ComBat', ComBat_transformer),
    ('GAMs', GAM_transformer),
    ('selector', CustomFeatureSelector(estimator=estimator)),
    ('clf', CustomXGBClassifier())
])

optimizer = BayesSearchCV(
    estimator=pipeline,
    search_spaces=pbounds,
    n_iter=50,
    scoring = 'roc_auc',
    n_jobs=-1,
    cv=cv_inner,
    random_state=random_seed
    )

#######################################       4. Start nested-CV!        ###############################################
# Create the empty lists to store the results
y_pred_proba_list = []
outer_fold_counter = 0
# Initialize lists to store feature importance and frequency
all_feature_importances = []
number_of_selected_features_list = []
feature_frequencies = np.zeros(n_features)
# Get the total number of outer folds
total_outer_folds = cv_outer.get_n_splits(X)
#  Initialize an empty list to store the best models
best_models = []
all_train_shap_values = []
all_test_shap_values = []
all_best_X_train = []
all_best_X_test = []
importances_weight_list = []
importances_gain_list = []
importances_cover_list = []
# Get the current starting time
current_time = datetime.now().strftime('%H:%M:%S')
print(f"[{current_time}] Progress - Outer Folds: 0.00% | Start to process the first iteration of outer folds")
elapsed_times = [] # Initialize a list to hold elapsed times

# Iterate over the test indices provided by leave-one-out
for pretrain_index, test_index in cv_outer.split(X[:, : -Covariate_number], y):
    start_time = time.time()  # Start the timer
    # Increment the outer fold counter
    outer_fold_counter += 1
    # Split the pretraining set and the test set
    X_pretrain, X_test = X[pretrain_index], X[test_index] # for numpy arrays
    y_pretrain, y_test = y[pretrain_index], y[test_index] # for numpy arrays
    # Perform Bayesian optimization in the inner loop to find the best hyperparameters and features
    optimizer.fit(X_pretrain, y_pretrain)
    print(f"Best score: {optimizer.best_score_:.4f}")
    # Get the best pipeline from the inner loop`
    best_pipeline = optimizer.best_estimator_
    best_models.append(best_pipeline)
    best_XGBoost = best_pipeline["clf"]     # get the classification step of the pipeline (XGBoost)
    # Store the SHAP values for the best classifier
    explainer = shap.TreeExplainer(best_XGBoost)
    # Predict the label y of the test set
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_proba_list.append(y_pred_proba.tolist()) #LOOCV
    # Get the feature importances
    importances = best_XGBoost.feature_importances_
    all_feature_importances.append(importances)
    count = sum(importances > 0)
    print("Non zero importances:", count)
    # Get the feature frequencies (Need to be blocked when no feature selection is performed)
    selector = best_pipeline.named_steps['selector']
    feature_frequencies[selector.selected_features_indices_] += 1
    number_of_selected_features = len(selector.selected_features_indices_ > 0)
    print("Number of selected features:", number_of_selected_features)
    number_of_selected_features_list.append(number_of_selected_features)
    # Get the feature importances: weight, gain and cover from XGBoost:
    importances_weight = best_XGBoost.get_booster().get_score(importance_type="weight")
    importances_weight_list.append(importances_weight)
    importances_gain = best_XGBoost.get_booster().get_score(importance_type="gain")
    importances_gain_list.append(importances_gain)
    importances_cover = best_XGBoost.get_booster().get_score(importance_type="cover")
    importances_cover_list.append(importances_cover)
    # Create a new pipeline without the classifier
    transform_pipeline = Pipeline(best_pipeline.steps[:-1])
    # Get the best training X
    best_X_train = transform_pipeline.transform(X_pretrain)
    # Get the best test X
    best_X_test = transform_pipeline.transform(X_test)
    # Compute the SHAP values
    shap_values_train = explainer.shap_values(best_X_train)
    shap_values_test = explainer.shap_values(best_X_test)
    # Store the SHAP values and the best X
    all_train_shap_values.append(shap_values_train)
    all_test_shap_values.append(shap_values_test)
    all_best_X_train.append(best_X_train)
    all_best_X_test.append(best_X_test)
    # Calculate the progress percentage for the outer folds
    progress_outer = outer_fold_counter / total_outer_folds * 100
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    elapsed_times.append(elapsed_time)
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    remaining_iterations = total_outer_folds - outer_fold_counter
    estimated_time_remaining = avg_elapsed_time * remaining_iterations
    # Convert elapsed_time to hours, minutes, seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Convert estimated_time_remaining to hours, minutes, seconds
    est_hours, remainder = divmod(estimated_time_remaining, 3600)
    est_minutes, est_seconds = divmod(remainder, 60)
    current_time = datetime.now().strftime('%H:%M:%S')  # Get the current real-time
    # Print the progress proportion for the outer folds
    print(
        f"[{current_time}] Progress - Outer Folds: {progress_outer:.2f}% | "
        f"Time taken for this iteration: {hours:.0f}h {minutes:.0f}m {seconds:.2f}s | "
        f"Estimated time to finish: {est_hours:.0f}h {est_minutes:.0f}m {est_seconds:.2f}s |"
    )

#####################      5. Compute the output statistics and plotting      ##########################################
# Obtain the y_pred_proba_list
y_pred = np.concatenate(y_pred_proba_list)
np.save(file_path + 'y_pred.npy', y_pred)
np.save(file_path + 'best_models.npy', best_models)

# Calculate the best theshold
precision, recall, thresholds_pr = precision_recall_curve(y, y_pred)
fpr, tpr, thresholds_roc = roc_curve(y, y_pred)
# method 1: Precision-Recall Curve
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
best_threshold_pr = thresholds_pr[ix]
# method 2: ROC Curve
j_stat = tpr - fpr
ix_roc = np.argmax(j_stat)
best_threshold_roc = thresholds_roc[ix_roc]
print("Best Threshold via Precision-Recall Curve (F1 score):", best_threshold_pr)
print("Best Threshold via ROC Curve (Youden's J statistic):", best_threshold_roc)

# Define the thresholds to evaluate
thresholds_list = [0.5, best_threshold_pr, best_threshold_roc]
# Define a dictionary to hold all results
results = {}

for threshold in thresholds_list:
    print(f"\nEvaluating threshold: {threshold}")
    outcomes = [1 if prob > threshold else 0 for prob in y_pred]
    y_accuracy_score = accuracy_score(y, outcomes)
    y_balanced_accuracy_score = balanced_accuracy_score(y, outcomes)
    y_precision_score = precision_score(y, outcomes)
    y_recall_score = recall_score(y, outcomes)
    y_f1_score = f1_score(y, outcomes)
    # Binomial test
    num_successes = sum(outcomes)  # Number of successes
    num_trials = len(outcomes)  # Total number of trials
    # Dictionary to store p-values for this threshold
    p_values_dict = {}
    for expected_proportion in [0.3617, 0.5]:
        # Perform binomial test
        p_value = binomtest(num_successes, n=num_trials, p=expected_proportion).pvalue
        # Store the p-value in the dictionary
        p_values_dict[f'p-value of {expected_proportion}'] = p_value

    # Store all metrics and p-values for this threshold
    results[f"Threshold_{threshold}"] = {
        'Accuracy': y_accuracy_score,
        'Balanced Accuracy': y_balanced_accuracy_score,
        'Precision': y_precision_score,
        'Recall': y_recall_score,
        'F1 Score': y_f1_score,
        **p_values_dict
    }

output = pd.DataFrame(results) # Convert the results dictionary to a DataFrame
output = output.T  # Transpose to have metrics as columns
output.to_csv(os.path.join(file_path, 'y_pred_accuracy.csv'), index_label="Threshold") # Save to CSV
# Print the results
pd.set_option('display.max_columns', None) # To display all columns
pd.set_option('display.max_rows', None) # To display all rows
pd.set_option('display.max_colwidth', None) # To display the entire content of each cell (no truncation)
print(output)

# Confusion Matrix
for threshold in thresholds_list:
    cm = confusion_matrix(y, y_pred > threshold)
    figname = f"confusion_matrix_{threshold:.3f}_threshold.png"
    labels = ['Non-respondents', 'Respondents']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of Threshold {threshold:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path,figname)) # Save the figure
    plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = roc_auc_score(y, y_pred)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('ROC Curve for 47 MPH Participants', fontsize = 20)
plt.legend(loc='lower right', fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(file_path,'roc_curve.png')) # Save the figure
plt.show()

# Calculate precision-recall curve values
precision, recall, _ = precision_recall_curve(y, y_pred)
pr_auc = auc(recall, precision)
# Plot Precision-Recall curve
plt.plot(recall, precision, label=f"Precision-Recall (AUC = {pr_auc:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')  # Changed the legend location for better visibility
plt.tight_layout()
plt.savefig(os.path.join(file_path, 'precision_recall_curve.png'))
plt.show()

# Calculate the feature importance and frequency
# Identify the indices of the features based on their selection frequency
sorted_feature_indices = np.argsort(feature_frequencies)
# Map the indices of features to the correct subset
selected_indices_in_each_model = [model.named_steps['selector'].selected_features_indices_
                                  for model in best_models]
# No selection of features:
# selected_indices_in_each_model = [list(range(n_features)) for _ in range(n_samples)]
# Convert list of dictionaries to matrix
def convert_importance_list_to_matrix(importance_list, n_features):
    matrix = np.zeros((n_samples, selected_features))
    for idx, imp_dict in enumerate(importance_list):
        for key, value in imp_dict.items():
            feature_idx = int(key[1:])  # Convert 'f0', 'f1', ... to 0, 1, ...
            matrix[idx][feature_idx] = value
    return matrix
# Convert list of dictionaries to matrix
importances_weight_matrix = convert_importance_list_to_matrix(importances_weight_list, n_features)
importances_gain_matrix = convert_importance_list_to_matrix(importances_gain_list, n_features)
importances_cover_matrix = convert_importance_list_to_matrix(importances_cover_list, n_features)
# Create a new array of shape (n_samples, n_samples -1, n_features) filled with zeros
new_feature_importances_array = np.zeros((n_samples, n_features))
new_feature_frequency_array = np.zeros((n_samples, n_features))
new_train_shap_array = np.zeros((n_samples, n_samples - 1, n_features))
new_train_X_array = np.zeros((n_samples, n_samples - 1, n_features))
new_test_shap_array = np.zeros((n_samples, 1,  n_features))
new_test_X_array = np.zeros((n_samples, 1,  n_features))
new_importance_weight_array = np.zeros((n_samples, n_features))
new_importance_gain_array = np.zeros((n_samples, n_features))
new_importance_cover_array = np.zeros((n_samples, n_features))
# Populate the new array with the SHAP values
for i in range(n_samples):
    num_selected_features = len(selected_indices_in_each_model[i])  # Dynamic number of selected features
    for j in range(num_selected_features):  # Loop over the last dimension of the original array
        idx = selected_indices_in_each_model[i][j]    # Get the index of the feature in the selected subset
        new_feature_importances_array[i, idx] = all_feature_importances[i][j]
        new_feature_frequency_array[i, idx] =+ 1
        new_train_shap_array[i, :, idx] = all_train_shap_values[i][:, j]
        new_train_X_array[i, :, idx] = all_best_X_train[i][:, j]
        new_test_shap_array[i, :, idx] = all_test_shap_values[i][:, j]
        new_test_X_array[i, :, idx] = all_best_X_test[i][:, j]
        new_importance_weight_array[i, idx] = importances_weight_matrix[i, j]
        new_importance_gain_array[i, idx] = importances_gain_matrix[i, j]
        new_importance_cover_array[i, idx] = importances_cover_matrix[i, j]
# Save the numpy arrays
np.save(file_path + 'new_feature_importances_array.npy', new_feature_importances_array)
np.save(file_path + 'new_feature_frequency_array.npy', new_feature_frequency_array)
np.save(file_path + 'new_train_shap_array.npy', new_train_shap_array)
np.save(file_path + 'new_train_X_array.npy', new_train_X_array)
np.save(file_path + 'new_test_shap_array.npy', new_test_shap_array)
np.save(file_path + 'new_test_X_array.npy', new_test_X_array)
np.save(file_path + 'new_importance_weight_array.npy', new_importance_weight_array)
np.save(file_path + 'new_importance_gain_array.npy', new_importance_gain_array)
np.save(file_path + 'new_importance_cover_array.npy', new_importance_cover_array)
mean_train_shap = np.mean(new_train_shap_array, axis=1)
mean_train_X = np.mean(new_train_X_array, axis=1)
# Plot the SHAP summary plot for the training set
reshaped_train_shap_array = np.reshape(new_train_shap_array, (n_samples *(n_samples -1), n_features))
reshaped_train_X_array = np.reshape(new_train_X_array, (n_samples *(n_samples -1), n_features))
shap.summary_plot(reshaped_train_shap_array, reshaped_train_X_array, feature_names=feature_names, max_display=max_display, show = False)
plt.gcf().set_size_inches(15, 7)
plt.title('SHAP Value of impact on model output (Training Set)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(file_path,'shap_summary_training.png')) # Save the figure
plt.show()
# Plot the SHAP summary plot for the test set
reshaped_test_shap_array = np.reshape(new_test_shap_array, (n_samples , n_features))
reshaped_test_X_array = np.reshape(new_test_X_array, (n_samples , n_features))
shap.summary_plot(reshaped_test_shap_array, reshaped_test_X_array, feature_names=feature_names, max_display=max_display, show = False)
plt.gcf().set_size_inches(15, 7)
plt.tight_layout()
plt.title('SHAP Value of impact on model output (Test Set)')
plt.savefig(os.path.join(file_path,'shap_summary_test.png')) # Save the figure
plt.show()

# Calculate the mean importance of each feature
mean_importances = np.mean(new_feature_importances_array, axis = 0)
num_models = len(best_models)
# Prepare the base DataFrame
df_base = pd.DataFrame({
    'Feature Index': range(1, n_features + 1),
    'Feature Name': feature_names,
    'Sum of Frequency': feature_frequencies,
    'Mean Importance': mean_importances,
    'Importances Weight': new_importance_weight_array.mean(axis=0),
    'Importances Gain': new_importance_gain_array.mean(axis=0),
    'Importances Cover': new_importance_cover_array.mean(axis=0)
})
# Create separate DataFrames for each model's metrics
dfs_to_concat = [df_base]
for model_idx in range(len(best_models)):
    df_model = pd.DataFrame({
        'Importance_Model_' + str(model_idx + 1): new_feature_importances_array[model_idx, :],
        'Frequency_Model_' + str(model_idx + 1): new_feature_frequency_array[model_idx, :],
        'Weight_Model_' + str(model_idx + 1): new_importance_weight_array[model_idx, :],
        'Gain_Model_' + str(model_idx + 1): new_importance_gain_array[model_idx, :],
        'Cover_Model_' + str(model_idx + 1): new_importance_cover_array[model_idx, :]
    })
    dfs_to_concat.append(df_model)
# Concatenate all DataFrames column-wise
features_summary = pd.concat(dfs_to_concat, axis=1)
# Sort the DataFrame by feature index
features_summary = features_summary.sort_values(by='Feature Index', ascending=True)
# Save the DataFrame to a CSV file
features_summary.to_csv(os.path.join(file_path,'features_summary.csv'), index=False)

# Create a heatmap with 3 importances
# Normalize each importance metric to be between 0 and 1
def normalize_array(array):
    max_val = np.max(array)
    min_val = np.min(array)
    if max_val == min_val:
        # Avoid division by zero in case all values are the same
        return np.zeros_like(array)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized
# Usage
norm_weight = normalize_array(new_importance_weight_array)
norm_gain = normalize_array(new_importance_gain_array)
norm_cover = normalize_array(new_importance_cover_array)

# Create RGB color for each feature and model using the normalized importance metrics
rgb_colors = np.stack([norm_weight, norm_cover, norm_gain], axis=-1)
swapped_colors = np.swapaxes(rgb_colors, 0, 1)
# Create heatmap using RGB colors
plt.figure(figsize=(15, 10))
plt.imshow(swapped_colors, aspect='auto', interpolation='none')
plt.colorbar(label='Importance Value')
# plt.yticks(np.arange(len(feature_names)), feature_names)
plt.xticks(range(1, n_samples + 1))
plt.xlabel('Models')
plt.ylabel('Features Index')
plt.title('Feature Importances Heatmap (Red=Weight, Green=Cover, Blue=Gain)')
plt.tight_layout()
plt.savefig(os.path.join(file_path,'feature_importances_heatmap_RGB.png')) # Save the figure
plt.show()

# Define a function to plot the top features for a given importance matrix
def plot_sorted_features(color_col_prefix, sort_col_prefix, title_suffix, max_display):
    # Extract the top features based on the sorting importance
    sort_col_names = [sort_col_prefix + str(i + 1) for i in range(len(best_models))]
    summed_sort_values = features_summary[sort_col_names].sum(axis=1)
    top_indices = summed_sort_values.nlargest(max_display).index
    # Get the color values for these features
    color_col_names = [color_col_prefix + str(i + 1) for i in range(len(best_models))]
    top_colors = features_summary.loc[top_indices, color_col_names]
    top_feature_names = features_summary.loc[top_indices, 'Feature Name']
    plt.figure(figsize=(16, 10))
    sns.heatmap(top_colors,
                cmap="YlGnBu",
                cbar_kws={'label': 'Feature Importance'},
                yticklabels=top_feature_names,
                xticklabels=list(range(1, top_colors.shape[1] + 1)),
                annot=False, fmt=".2f")
    plt.title(f"Feature Importances Across Models ({title_suffix})")
    plt.xlabel("Models")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'feature_importances_{title_suffix}.png'))  # Save the figure
    plt.show()
# Plot heatmap where color represents weight and features are sorted by gain
plot_sorted_features('Weight_Model_', 'Gain_Model_', 'Color: Weight | Sorted by: Gain', max_display)
plot_sorted_features('Weight_Model_', 'Cover_Model_', 'Color: Weight | Sorted by: Cover', max_display)
plot_sorted_features('Gain_Model_', 'Weight_Model_', 'Color: Gain | Sorted by: Weight', max_display)
plot_sorted_features('Gain_Model_', 'Cover_Model_', 'Color: Gain | Sorted by: Cover', max_display)
plot_sorted_features('Cover_Model_', 'Weight_Model_', 'Color: Cover | Sorted by: Weight', max_display)
plot_sorted_features('Cover_Model_', 'Gain_Model_', 'Color: Cover | Sorted by: Gain', max_display)

# Plot the histogram of feature selection frequencies and mean importances
Plot_label = ['Sum of Frequency', 'Mean Importance']
for i in Plot_label:
    # Plot the histogram of feature selection frequencies
    plt.figure(figsize=(15, 7))  # Set the figure size
    plt.bar(features_summary['Feature Index'], features_summary[i], color='steelblue')
    plt.xlabel('Feature Index')
    plt.ylabel(i)
    plt.title('Histogram of Feature ' + i)
    plt.tight_layout()
    new_string = i.replace(' ', '_')
    plt.savefig(os.path.join(file_path,'histogram_of_feature_' + new_string + '.png')) # Save the figure
    plt.show()

    # Plot the histogram of top 10 feature frequencies
    # Select the top 10 features based on Frequency
    top_10_features_summary = features_summary.nlargest(10, i)
    # Plot a horizontal bar chart for the top 10 features
    plt.figure(figsize=(15, 7))
    plt.barh(top_10_features_summary['Feature Name'],
             top_10_features_summary[i],
             color='steelblue')
    plt.xlabel(i)
    plt.ylabel('Feature Name')
    plt.title('Top 10 Features based on ' + i)
    plt.gca().invert_yaxis()  # to display the highest frequency at the top
    plt.tight_layout()
    new_string = i.replace(' ', '_')
    plt.savefig(os.path.join(file_path,'top_10_features_based_on_' + new_string + '.png')) # Save the figure
    plt.show()

# Plot the heatmap of feature importances across models
# Step 1: Sort the features based on their frequency and select the top 10
top_10_features = features_summary.nlargest(max_display, 'Sum of Frequency')
# Step 2: Extract the importances of these top 10 features for each of the best models
importance_columns = ['Importance_Model_' + str(i + 1) for i in range(num_models)]
top_10_importances = top_10_features[importance_columns]
# Step 3: Plot a heatmap for these top 10 features
plt.figure(figsize=(16, 10))
sns.heatmap(top_10_importances,
            cmap="YlGnBu",
            cbar_kws={'label': 'Feature Importance'},
            yticklabels=top_10_features['Feature Name'],
            xticklabels = list(range(1,X.shape[0]+1)),
            annot=False, fmt=".2f")
plt.xlabel("Models")
plt.ylabel("Feature Name")
plt.title('Feature Importances Across Models (Top ' + str(max_display) + ' Features by frequency)')
plt.tight_layout()
plt.savefig(os.path.join(file_path,'top_feature_importances_across_models(RFEfrequency).png')) # Save the figure
plt.show()

# Plot the heatmap of feature frequency across models
# Step 1: Sort the features based on their frequency and select the top 10
top_10_features = features_summary.nlargest(max_display, 'Mean Importance')
# Step 2: Extract the importances of these top 10 features for each of the best models
frequency_columns = ['Frequency_Model_' + str(i + 1) for i in range(num_models)]
top_10_frequency = top_10_features[frequency_columns]
# Step 3: Plot a heatmap for these top 10 features
plt.figure(figsize=(16, 10))
sns.heatmap(top_10_frequency,
            cmap="YlGnBu",
            cbar_kws={'label': 'Feature Frequency'},
            yticklabels=top_10_features['Feature Name'],
            xticklabels = list(range(1,X.shape[0]+1)),
            annot=False, fmt=".2f")
plt.xlabel("Models")
plt.ylabel("Feature Name")
plt.title("Feature Frequency Across Models (Top " + str(max_display) + " Features by mean importance)")
plt.tight_layout()
plt.savefig(os.path.join(file_path,'top_feature_frequency_across_models(RFEfrequency).png')) # Save the figure
plt.show()

# Decision Curve Analysis
def calculate_net_benefit(strategy, true_labels, pred_probs, thresholds):
    n = len(true_labels)
    net_benefit = []

    for p in thresholds:
        if strategy == 'treat_all_good':
            # Treat all patients as good responders
            tp = np.sum(true_labels == 1)
            fp = np.sum(true_labels == 0)
        elif strategy == 'treat_none_good':
            # Treat no patients as good responders
            tp = 0
            fp = 0
        elif strategy == 'prediction_model':
            # Use the prediction model to determine good responders
            tp = ((pred_probs >= p) & (true_labels == 1)).sum()
            fp = ((pred_probs >= p) & (true_labels == 0)).sum()

        # Calculate net benefit as true positives minus false positives weighted by the threshold probability
        benefit = tp - (fp * p / (1 - p))
        net_benefit.append(benefit / n - (p * np.mean(true_labels)))

    return net_benefit

# Define a range of threshold probabilities
thresholds = np.linspace(0, 1, 100)

# Calculate net benefits for different strategies
net_benefit_treat_all_good = calculate_net_benefit('treat_all_good', y, y_pred, thresholds)
net_benefit_treat_none_good = calculate_net_benefit('treat_none_good', y, y_pred, thresholds)
net_benefit_prediction_model = calculate_net_benefit('prediction_model', y, y_pred, thresholds)

# Plot Decision Curve Analysis
plt.figure(figsize=(10, 6))
plt.plot(thresholds, net_benefit_prediction_model, label='Prediction Model')
plt.plot(thresholds, net_benefit_treat_all_good, label='Treat All as Good Responders', linestyle='--')
plt.plot(thresholds, net_benefit_treat_none_good, label='Treat None as Good Responders', linestyle='--')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis for Treatment Response')
plt.legend()
plt.grid(True)
plt.show()


########################################################################################################################
#################################            Subgroup Analysis          ##############################################
##################################         2. Load the input data        ###############################################
# Tier 1 or 2
X = pd.read_csv('enter_your_X_csv_file_path')
y = pd.read_csv('enter_your_y_csv_file_path', header=None)

# Define the features numbers
# Tier 1
# T1_features =  10   #10 or 320
# MRI_features = 14   #14 or 380
# all_features = 14 # 14 or 380
# selected_features = 5 # 4 (20), 5 (23-24) or no selection(14 or 380)

# Tier 2
T1_features =  320   #10 or 320
MRI_features = 380   #14 or 380
all_features = 380 # 14 or 380
selected_features = 5 # 4 (20), 5 (23-24) or no selection(14 or 380)

Covariate_number = 1 # 1
non_MRI_number = 1 # 1
# Set the random seed
random_seed = 42
np.random.seed(random_seed)
max_display = 10 # number of features to display in the heatmap plots

# Store the feature names
feature_names = X.iloc[:, : -Covariate_number].columns.to_list()
print(" Feature's total number: ", len(feature_names), ';', "\n",
      "Selected feature's number: ", selected_features,';',  "\n",
      "Covariate number: ", Covariate_number, "\n")
# Convert to numpy arrays
X = np.float64(X.to_numpy())
y = np.ravel(y).astype(int) # Reshape y to (n_samples,) with dtype=int

print(' MPH_X shape: ', X.shape, ', dtype:', X.dtype, ';', "\n",
      'MPH_y shape: ', y.shape, ', dtype:', y.dtype, ';', "\n",
      )

# Create the output folder for the images
file_path = 'enter_your_output_folder_path'
# Open the file in "write" mode, creating it if it doesn't exist
if not os.path.exists(file_path):
    os.makedirs(file_path)
    print(f" New directory created.")
else:
    print(f" This directory already exists.")

# Define the range of T1 and DTI features for harmonization
n_samples = X.shape[0]
n_features = X.shape[1] - Covariate_number
T1_features_start = 0
MRI_features_start = 0
MRI_features_end = MRI_features
T1_features_end = T1_features
DTI_features_start = T1_features_end
DTI_features_end = MRI_features
non_MRI_features_start = MRI_features
non_MRI_features_end = X.shape[1] - Covariate_number
All_features_start = 0
All_features_end = X.shape[1] - Covariate_number
MRI_features_range = range(T1_features_start, MRI_features)
non_MRI_features_range = range(non_MRI_features_start, non_MRI_features_end)
covariates_range = range(non_MRI_features_end, X.shape[1])
print( 'n_samples: ', n_samples, '; n_features: ', n_features, ';',"\n",
       'T1 features range: ', T1_features_start, '-', T1_features_end, ';',"\n",
       'DTI features range: ', DTI_features_start, '-', DTI_features_end, ';',"\n",
       'MRI features range: ', MRI_features_start, '-', MRI_features_end, ';',"\n",
       'Non-MRI features range: ', non_MRI_features_start, '-', non_MRI_features_end, ';',"\n",
       'All features range: ', All_features_start, '-', All_features_end, ';',"\n",)

####################################      3. Define the models      ####################################################
# Custom transformer for feature selection

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, centering=True, scaling=True):
        self.centering = centering
        self.scaling = scaling
        self.scaler = RobustScaler(with_centering=self.centering, with_scaling=self.scaling)
    def fit(self, X, y=None, **kwargs):
        # Fit the scaler on the features (excluding the last two columns which are covariates)
        self.scaler.fit(X[:, : - Covariate_number])
        return self
    def transform(self, X, **kwargs):
        # Scale the features (excluding the last two columns which are covariates)
        X_scaled = self.scaler.transform(X[:, : - Covariate_number])
        # Return the scaled features WITH the last two columns appended to the end
        return np.hstack([X_scaled, X[:, - non_MRI_number: ]])

class CustomGAMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_index, end_index, lam=1.0):
        self.start_index = start_index
        self.end_index = end_index
        self.lam = lam
        self.gam_models = []
    def fit(self, X, y=None, **fit_params):
        data = X[:, self.start_index:self.end_index]
        covariate = X[:, -1].reshape(-1, 1)
        # Fit a GAM model for each feature
        self.gam_models = []
        for feature_idx in range(data.shape[1]):
            gam = LinearGAM(s(0, lam=self.lam)).fit(covariate, data[:, feature_idx])
            self.gam_models.append(gam)
        return self
    def transform(self, X, **transform_params):
        data = X[:, self.start_index:self.end_index]
        covariate = X[:, -1].reshape(-1, 1)
        # Adjust each feature using its respective GAM model
        adjusted_data = np.column_stack([
            data[:, feature_idx] - model.predict(covariate)
            for feature_idx, model in enumerate(self.gam_models)
        ])
        # Return the scaled features WITHOUT the last columns appended to the end
        return np.hstack([adjusted_data, X[:, non_MRI_features_range]])

class CustomCombatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index
        self.sites = None
        self.discrete_covariates = None
        self.continuous_covariates = None
        self.combat_model = CombatModel()
    def fit(self, X, y = None, **fit_params):
        data = X[:, self.start_index:self.end_index]
        sites = X[:, -2].reshape(-1, 1)
        self.combat_model.fit(data=data, sites=sites, discrete_covariates=None,
                              continuous_covariates=None)
        return self
    def transform(self, X, **transform_params):
        data = X[:, self.start_index:self.end_index]
        sites = X[:, -2].reshape(-1, 1)  # Use sites instead of self.sites
        transformed_data = self.combat_model.transform(data = data, sites = sites, discrete_covariates=None,
                                                       continuous_covariates = None)
        # Return the scaled features WITH the last columns appended to the end
        return np.hstack([transformed_data, X[:, - non_MRI_number: ]])

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select = selected_features, corr_th=0.8):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.corr_th = corr_th
    def fit(self, X, y=None):
        selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select, step=1)
        self.selected_features_indices_ = self.selectNonIntercorrelated(X, y, self.corr_th, selector)
        # Store selected X and y for potential later use
        self.X_selected_ = X[:, self.selected_features_indices_]
        self.y_selected_ = y
        return self
    def transform(self, X):
        return X[:, self.selected_features_indices_]
    def selectNonIntercorrelated(self, X, y, corr_th, selector):
        # Features without missing values
        non_nan_indices = np.all(~np.isnan(X), axis=0)
        X_non_nan = X[:, non_nan_indices]
        # Features with non-zero MAD variance
        mad_values = median_abs_deviation(X_non_nan, axis=0, scale='normal')
        non_zero_var_indices = mad_values > 0.001
        X_non_zero_var = X_non_nan[:, non_zero_var_indices]
        # Check if there are no non-zero MAD features
        if X_non_zero_var.shape[1] == 0:
            raise ValueError("All features have zero MAD")
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_non_zero_var, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)  # set diagonal to zero, to avoid self-correlation
        mean_absolute_corr = np.abs(corr_matrix).mean(axis=0)
        # Identify intercorrelated features
        intercorrelated_features_set = set()
        high_corrs = np.argwhere(np.abs(corr_matrix) > corr_th)
        for i, j in high_corrs:
            if mean_absolute_corr[i] > mean_absolute_corr[j]:
                intercorrelated_features_set.add(i)
            else:
                intercorrelated_features_set.add(j)
        non_intercorrelated_indices = list(set(range(X_non_zero_var.shape[1])) - intercorrelated_features_set)
        #Check if there are no non-intercorrelated features
        if len(non_intercorrelated_indices) == 0:
            # If all features are highly correlated, select the feature with the highest correlation
            # with the target variable
            corr_with_target = np.abs(np.corrcoef(X_non_zero_var, y, rowvar=False)[:-1, -1])
            selected_index = np.argmax(corr_with_target)
            return [selected_index]
        # Apply RFE on non-intercorrelated features
        X_train_non_intercorrelated = X_non_zero_var[:, non_intercorrelated_indices]
        # Check if only one feature is left, and if so, skip RFE
        if X_train_non_intercorrelated.shape[1] <= self.n_features_to_select:
            # If the number of features is less than or equal to the desired number, skip RFE
            selected_indices = np.array(non_intercorrelated_indices)  # Select the remaining feature
        else:
            # If there's more than one feature, proceed with RFE as normal
            selector = selector.fit(X_train_non_intercorrelated, y)  # Pass y here
            # Get selected feature indices
            support = selector.get_support()
            selected_indices = np.array(non_intercorrelated_indices)[support]
        return selected_indices

class CustomXGBClassifier(XGBClassifier):
    def __init__(self, random_state=random_seed,
                 eval_metric='auc', # Focusing on true negative group in imbalanced data
                 objective='binary:logistic',
                 **kwargs):
        super().__init__(random_state=random_state,
                         objective=objective,
                         eval_metric=eval_metric,
                         **kwargs)
    def fit(self, X, y, **kwargs):
        # Calculate scale_pos_weight
        class_counts = np.bincount(y.astype(int))
        scale_pos_weight = class_counts[0] / class_counts[1]
        # Set the scale_pos_weight for the current iteration
        self.set_params(scale_pos_weight=scale_pos_weight)
        # Proceed with the normal fit
        return super().fit(X, y, **kwargs)


# Create the cross-validation objects
cv_inner = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=random_seed)
cv_outer = LeaveOneOut()

# Specify the hyperparameter space for BayesSearchCV
pbounds = {
    # Hyperparameters for Scale_Regress
    'Scale__centering': Categorical([True, False]),  # Centering hyperparameter for RobustScaler
    'Scale__scaling': Categorical([True, False]),  # Scaling hyperparameter for RobustScaler
    # Hyperparameters for Harmonization
    'GAMs__lam': Real(1e-6, 1e6, prior='log-uniform'),
    # Hyperparameters for the Selector RFE
    'selector__estimator__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    'selector__estimator__max_depth': Integer(2, 7),
    'selector__estimator__subsample': Real(0.5, 1.0),
    'selector__estimator__colsample_bytree': Real(0.5, 1.0),
    'selector__estimator__min_child_weight': Integer(1, 20),
    'selector__estimator__alpha': Real(0, 2),
    'selector__estimator__lambda': Real(0, 2),
    'selector__estimator__n_estimators': Integer(10, 150),
    'selector__estimator__gamma': Real(0, 10),
    # Hyperparameters for XGBClassifier
    'clf__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    'clf__max_depth': Integer(2, 7),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__min_child_weight': Integer(1, 20),
    'clf__alpha': Real(0, 2),
    'clf__lambda': Real(0, 2),
    'clf__n_estimators': Integer(10, 150),
    'clf__gamma': Real(0, 10)
} # XGBoost hyperparameters

estimator = CustomXGBClassifier()

# Create the custom Covariate Harmonization
GAM_transformer = CustomGAMTransformer(T1_features_start, DTI_features_end)

# Create a pipeline
pipeline = Pipeline([
    ('Scale', CustomScaler()),
    ('GAMs', GAM_transformer),
    ('selector', CustomFeatureSelector(estimator)),
    ('clf', CustomXGBClassifier())
])

optimizer = BayesSearchCV(
    estimator=pipeline,
    search_spaces=pbounds,
    n_iter=50,
    scoring = 'roc_auc',
    n_jobs=-1,
    cv=cv_inner,
    random_state=random_seed
    )


#######################################       4. Start nested-CV!        ###############################################
# Create the empty lists to store the results
y_pred_proba_list = []
outer_fold_counter = 0
# Initialize lists to store feature importance and frequency
all_feature_importances = []
number_of_selected_features_list = []
feature_frequencies = np.zeros(n_features)
# Get the total number of outer folds
total_outer_folds = cv_outer.get_n_splits(X)
#  Initialize an empty list to store the best models
best_models = []
all_train_shap_values = []
all_test_shap_values = []
all_best_X_train = []
all_best_X_test = []
importances_weight_list = []
importances_gain_list = []
importances_cover_list = []
# Get the current starting time
current_time = datetime.now().strftime('%H:%M:%S')
print(f"[{current_time}] Progress - Outer Folds: 0.00% | Start to process the first iteration of outer folds")
elapsed_times = [] # Initialize a list to hold elapsed times

# Iterate over the test indices provided by leave-one-out
for pretrain_index, test_index in cv_outer.split(X[:, : -Covariate_number], y):
    start_time = time.time()  # Start the timer
    # Increment the outer fold counter
    outer_fold_counter += 1
    # Split the pretraining set and the test set
    X_pretrain, X_test = X[pretrain_index], X[test_index] # for numpy arrays
    y_pretrain, y_test = y[pretrain_index], y[test_index] # for numpy arrays
    # Perform Bayesian optimization in the inner loop to find the best hyperparameters and features
    optimizer.fit(X_pretrain, y_pretrain)
    # TODO: we do 5-fold CV ,so we expect 5 pipelines with 5 best validation scores, of which we should take the mean
    print(f"Best score: {optimizer.best_score_:.4f}")
    # Get the best pipeline from the inner loop`
    best_pipeline = optimizer.best_estimator_
    best_models.append(best_pipeline)
    best_XGBoost = best_pipeline["clf"]     # get the classification step of the pipeline (XGBoost)
    # Store the SHAP values for the best classifier
    explainer = shap.TreeExplainer(best_XGBoost)
    # Predict the label y of the test set
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_proba_list.append(y_pred_proba.tolist()) #LOOCV
    # Get the feature importances
    importances = best_XGBoost.feature_importances_
    all_feature_importances.append(importances)
    count = sum(importances > 0)
    print("Non zero importances:", count)
    # Get the feature frequencies (Need to be blocked when no feature selection is performed)
    selector = best_pipeline.named_steps['selector']
    feature_frequencies[selector.selected_features_indices_] += 1
    number_of_selected_features = len(selector.selected_features_indices_ > 0)
    print("Number of selected features:", number_of_selected_features)
    number_of_selected_features_list.append(number_of_selected_features)
    # Get the feature importances: weight, gain and cover from XGBoost:
    importances_weight = best_XGBoost.get_booster().get_score(importance_type="weight")
    importances_weight_list.append(importances_weight)
    importances_gain = best_XGBoost.get_booster().get_score(importance_type="gain")
    importances_gain_list.append(importances_gain)
    importances_cover = best_XGBoost.get_booster().get_score(importance_type="cover")
    importances_cover_list.append(importances_cover)
    # Create a new pipeline without the classifier
    transform_pipeline = Pipeline(best_pipeline.steps[:-1])
    # Get the best training X
    best_X_train = transform_pipeline.transform(X_pretrain)
    # Get the best test X
    best_X_test = transform_pipeline.transform(X_test)
    # Compute the SHAP values
    shap_values_train = explainer.shap_values(best_X_train)
    shap_values_test = explainer.shap_values(best_X_test)
    # Store the SHAP values and the best X
    all_train_shap_values.append(shap_values_train)
    all_test_shap_values.append(shap_values_test)
    all_best_X_train.append(best_X_train)
    all_best_X_test.append(best_X_test)
    # Calculate the progress percentage for the outer folds
    progress_outer = outer_fold_counter / total_outer_folds * 100
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    elapsed_times.append(elapsed_time)
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    remaining_iterations = total_outer_folds - outer_fold_counter
    estimated_time_remaining = avg_elapsed_time * remaining_iterations
    # Convert elapsed_time to hours, minutes, seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Convert estimated_time_remaining to hours, minutes, seconds
    est_hours, remainder = divmod(estimated_time_remaining, 3600)
    est_minutes, est_seconds = divmod(remainder, 60)
    current_time = datetime.now().strftime('%H:%M:%S')  # Get the current real-time
    # Print the progress proportion for the outer folds
    print(
        f"[{current_time}] Progress - Outer Folds: {progress_outer:.2f}% | "
        f"Time taken for this iteration: {hours:.0f}h {minutes:.0f}m {seconds:.2f}s | "
        f"Estimated time to finish: {est_hours:.0f}h {est_minutes:.0f}m {est_seconds:.2f}s |"
    )

#####################      5. Compute the output statistics and plotting      ##########################################
# Obtain the y_pred_proba_list
y_pred = np.concatenate(y_pred_proba_list)
np.save(file_path + 'y_pred.npy', y_pred)
np.save(file_path + 'best_models.npy', best_models)

# Calculate the best theshold
precision, recall, thresholds_pr = precision_recall_curve(y, y_pred)
fpr, tpr, thresholds_roc = roc_curve(y, y_pred)
# method 1: Precision-Recall Curve
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
best_threshold_pr = thresholds_pr[ix]
# method 2: ROC Curve
j_stat = tpr - fpr
ix_roc = np.argmax(j_stat)
best_threshold_roc = thresholds_roc[ix_roc]
print("Best Threshold via Precision-Recall Curve (F1 score):", best_threshold_pr)
print("Best Threshold via ROC Curve (Youden's J statistic):", best_threshold_roc)

# Define the thresholds to evaluate
thresholds_list = [0.5, best_threshold_pr, best_threshold_roc]
# Define a dictionary to hold all results
results = {}

for threshold in thresholds_list:
    print(f"\nEvaluating threshold: {threshold}")
    outcomes = [1 if prob > threshold else 0 for prob in y_pred]
    y_accuracy_score = accuracy_score(y, outcomes)
    y_balanced_accuracy_score = balanced_accuracy_score(y, outcomes)
    y_precision_score = precision_score(y, outcomes)
    y_recall_score = recall_score(y, outcomes)
    y_f1_score = f1_score(y, outcomes)
    # Binomial test
    num_successes = sum(outcomes)  # Number of successes
    num_trials = len(outcomes)  # Total number of trials
    # Dictionary to store p-values for this threshold
    p_values_dict = {}
    for expected_proportion in [0.3617, 0.5]:
        # Perform binomial test
        p_value = binomtest(num_successes, n=num_trials, p=expected_proportion).pvalue
        # Store the p-value in the dictionary
        p_values_dict[f'p-value of {expected_proportion}'] = p_value

    # Store all metrics and p-values for this threshold
    results[f"Threshold_{threshold}"] = {
        'Accuracy': y_accuracy_score,
        'Balanced Accuracy': y_balanced_accuracy_score,
        'Precision': y_precision_score,
        'Recall': y_recall_score,
        'F1 Score': y_f1_score,
        **p_values_dict
    }

output = pd.DataFrame(results) # Convert the results dictionary to a DataFrame
output = output.T  # Transpose to have metrics as columns
output.to_csv(os.path.join(file_path, 'y_pred_accuracy.csv'), index_label="Threshold") # Save to CSV
# Print the results
pd.set_option('display.max_columns', None) # To display all columns
pd.set_option('display.max_rows', None) # To display all rows
pd.set_option('display.max_colwidth', None) # To display the entire content of each cell (no truncation)
print(output)

# Confusion Matrix
for threshold in thresholds_list:
    cm = confusion_matrix(y, y_pred > threshold)
    figname = f"confusion_matrix_{threshold:.3f}_threshold.png"
    labels = ['Non-respondents', 'Respondents']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of Threshold {threshold:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path,figname)) # Save the figure
    plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = roc_auc_score(y, y_pred)
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Receiver Operating Characteristic Curve', fontsize = 20)
plt.legend(loc='lower right', fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(file_path,'roc_curve.png')) # Save the figure
plt.show()

# Calculate precision-recall curve values
precision, recall, _ = precision_recall_curve(y, y_pred)
pr_auc = auc(recall, precision)
# Plot Precision-Recall curve
plt.plot(recall, precision, label=f"Precision-Recall (AUC = {pr_auc:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')  # Changed the legend location for better visibility
plt.tight_layout()
plt.savefig(os.path.join(file_path, 'precision_recall_curve.png'))
plt.show()

# Calculate the feature importance and frequency
# Identify the indices of the features based on their selection frequency
sorted_feature_indices = np.argsort(feature_frequencies)
# Map the indices of features to the correct subset
selected_indices_in_each_model = [model.named_steps['selector'].selected_features_indices_
                                  for model in best_models]
# No selection of features:
# selected_indices_in_each_model = [list(range(n_features)) for _ in range(n_samples)]
# Convert list of dictionaries to matrix
def convert_importance_list_to_matrix(importance_list, n_features):
    matrix = np.zeros((n_samples, selected_features))
    for idx, imp_dict in enumerate(importance_list):
        for key, value in imp_dict.items():
            feature_idx = int(key[1:])  # Convert 'f0', 'f1', ... to 0, 1, ...
            matrix[idx][feature_idx] = value
    return matrix
# Convert list of dictionaries to matrix
importances_weight_matrix = convert_importance_list_to_matrix(importances_weight_list, n_features)
importances_gain_matrix = convert_importance_list_to_matrix(importances_gain_list, n_features)
importances_cover_matrix = convert_importance_list_to_matrix(importances_cover_list, n_features)
# Create a new array of shape (n_samples, n_samples -1, n_features) filled with zeros
new_feature_importances_array = np.zeros((n_samples, n_features))
new_feature_frequency_array = np.zeros((n_samples, n_features))
new_train_shap_array = np.zeros((n_samples, n_samples - 1, n_features))
new_train_X_array = np.zeros((n_samples, n_samples - 1, n_features))
new_test_shap_array = np.zeros((n_samples, 1,  n_features))
new_test_X_array = np.zeros((n_samples, 1,  n_features))
new_importance_weight_array = np.zeros((n_samples, n_features))
new_importance_gain_array = np.zeros((n_samples, n_features))
new_importance_cover_array = np.zeros((n_samples, n_features))
# Populate the new array with the SHAP values
for i in range(n_samples):
    num_selected_features = len(selected_indices_in_each_model[i])  # Dynamic number of selected features
    for j in range(num_selected_features):  # Loop over the last dimension of the original array
        idx = selected_indices_in_each_model[i][j]    # Get the index of the feature in the selected subset
        new_feature_importances_array[i, idx] = all_feature_importances[i][j]
        new_feature_frequency_array[i, idx] =+ 1
        new_train_shap_array[i, :, idx] = all_train_shap_values[i][:, j]
        new_train_X_array[i, :, idx] = all_best_X_train[i][:, j]
        new_test_shap_array[i, :, idx] = all_test_shap_values[i][:, j]
        new_test_X_array[i, :, idx] = all_best_X_test[i][:, j]
        new_importance_weight_array[i, idx] = importances_weight_matrix[i, j]
        new_importance_gain_array[i, idx] = importances_gain_matrix[i, j]
        new_importance_cover_array[i, idx] = importances_cover_matrix[i, j]
# Save the numpy arrays
np.save(file_path + 'new_feature_importances_array.npy', new_feature_importances_array)
np.save(file_path + 'new_feature_frequency_array.npy', new_feature_frequency_array)
np.save(file_path + 'new_train_shap_array.npy', new_train_shap_array)
np.save(file_path + 'new_train_X_array.npy', new_train_X_array)
np.save(file_path + 'new_test_shap_array.npy', new_test_shap_array)
np.save(file_path + 'new_test_X_array.npy', new_test_X_array)
np.save(file_path + 'new_importance_weight_array.npy', new_importance_weight_array)
np.save(file_path + 'new_importance_gain_array.npy', new_importance_gain_array)
np.save(file_path + 'new_importance_cover_array.npy', new_importance_cover_array)
mean_train_shap = np.mean(new_train_shap_array, axis=1)
mean_train_X = np.mean(new_train_X_array, axis=1)
# Plot the SHAP summary plot for the training set
reshaped_train_shap_array = np.reshape(new_train_shap_array, (n_samples *(n_samples -1), n_features))
reshaped_train_X_array = np.reshape(new_train_X_array, (n_samples *(n_samples -1), n_features))
shap.summary_plot(reshaped_train_shap_array, reshaped_train_X_array, feature_names=feature_names, max_display=max_display, show = False)
plt.gcf().set_size_inches(15, 7)
plt.title('SHAP Value of impact on model output (Training Set)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(file_path,'shap_summary_training.png')) # Save the figure
plt.show()
# Plot the SHAP summary plot for the test set
reshaped_test_shap_array = np.reshape(new_test_shap_array, (n_samples , n_features))
reshaped_test_X_array = np.reshape(new_test_X_array, (n_samples , n_features))
shap.summary_plot(reshaped_test_shap_array, reshaped_test_X_array, feature_names=feature_names, max_display=max_display, show = False)
plt.gcf().set_size_inches(15, 7)
plt.tight_layout()
plt.title('SHAP Value of impact on model output (Test Set)')
plt.savefig(os.path.join(file_path,'shap_summary_test.png')) # Save the figure
plt.show()

# Calculate the mean importance of each feature
mean_importances = np.mean(new_feature_importances_array, axis = 0)
num_models = len(best_models)
# Prepare the base DataFrame
df_base = pd.DataFrame({
    'Feature Index': range(1, n_features + 1),
    'Feature Name': feature_names,
    'Sum of Frequency': feature_frequencies,
    'Mean Importance': mean_importances,
    'Importances Weight': new_importance_weight_array.mean(axis=0),
    'Importances Gain': new_importance_gain_array.mean(axis=0),
    'Importances Cover': new_importance_cover_array.mean(axis=0)
})
# Create separate DataFrames for each model's metrics
dfs_to_concat = [df_base]
for model_idx in range(len(best_models)):
    df_model = pd.DataFrame({
        'Importance_Model_' + str(model_idx + 1): new_feature_importances_array[model_idx, :],
        'Frequency_Model_' + str(model_idx + 1): new_feature_frequency_array[model_idx, :],
        'Weight_Model_' + str(model_idx + 1): new_importance_weight_array[model_idx, :],
        'Gain_Model_' + str(model_idx + 1): new_importance_gain_array[model_idx, :],
        'Cover_Model_' + str(model_idx + 1): new_importance_cover_array[model_idx, :]
    })
    dfs_to_concat.append(df_model)
# Concatenate all DataFrames column-wise
features_summary = pd.concat(dfs_to_concat, axis=1)
# Sort the DataFrame by feature index
features_summary = features_summary.sort_values(by='Feature Index', ascending=True)
# Save the DataFrame to a CSV file
features_summary.to_csv(os.path.join(file_path,'features_summary.csv'), index=False)

# Create a heatmap with 3 importances
# Normalize each importance metric to be between 0 and 1
def normalize_array(array):
    max_val = np.max(array)
    min_val = np.min(array)
    if max_val == min_val:
        # Avoid division by zero in case all values are the same
        return np.zeros_like(array)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized
# Usage
norm_weight = normalize_array(new_importance_weight_array)
norm_gain = normalize_array(new_importance_gain_array)
norm_cover = normalize_array(new_importance_cover_array)

# Create RGB color for each feature and model using the normalized importance metrics
rgb_colors = np.stack([norm_weight, norm_cover, norm_gain], axis=-1)
swapped_colors = np.swapaxes(rgb_colors, 0, 1)
# Create heatmap using RGB colors
plt.figure(figsize=(15, 10))
plt.imshow(swapped_colors, aspect='auto', interpolation='none')
plt.colorbar(label='Importance Value')
# plt.yticks(np.arange(len(feature_names)), feature_names)
plt.xticks(range(1, n_samples + 1))
plt.xlabel('Models')
plt.ylabel('Features Index')
plt.title('Feature Importances Heatmap (Red=Weight, Green=Cover, Blue=Gain)')
plt.tight_layout()
plt.savefig(os.path.join(file_path,'feature_importances_heatmap_RGB.png')) # Save the figure
plt.show()

# Define a function to plot the top features for a given importance matrix
def plot_sorted_features(color_col_prefix, sort_col_prefix, title_suffix, max_display):
    # Extract the top features based on the sorting importance
    sort_col_names = [sort_col_prefix + str(i + 1) for i in range(len(best_models))]
    summed_sort_values = features_summary[sort_col_names].sum(axis=1)
    top_indices = summed_sort_values.nlargest(max_display).index
    # Get the color values for these features
    color_col_names = [color_col_prefix + str(i + 1) for i in range(len(best_models))]
    top_colors = features_summary.loc[top_indices, color_col_names]
    top_feature_names = features_summary.loc[top_indices, 'Feature Name']
    plt.figure(figsize=(16, 10))
    sns.heatmap(top_colors,
                cmap="YlGnBu",
                cbar_kws={'label': 'Feature Importance'},
                yticklabels=top_feature_names,
                xticklabels=list(range(1, top_colors.shape[1] + 1)),
                annot=False, fmt=".2f")
    plt.title(f"Feature Importances Across Models ({title_suffix})")
    plt.xlabel("Models")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'feature_importances_{title_suffix}.png'))  # Save the figure
    plt.show()
# Plot heatmap where color represents weight and features are sorted by gain
plot_sorted_features('Weight_Model_', 'Gain_Model_', 'Color: Weight | Sorted by: Gain', max_display)
plot_sorted_features('Weight_Model_', 'Cover_Model_', 'Color: Weight | Sorted by: Cover', max_display)
plot_sorted_features('Gain_Model_', 'Weight_Model_', 'Color: Gain | Sorted by: Weight', max_display)
plot_sorted_features('Gain_Model_', 'Cover_Model_', 'Color: Gain | Sorted by: Cover', max_display)
plot_sorted_features('Cover_Model_', 'Weight_Model_', 'Color: Cover | Sorted by: Weight', max_display)
plot_sorted_features('Cover_Model_', 'Gain_Model_', 'Color: Cover | Sorted by: Gain', max_display)

# Plot the histogram of feature selection frequencies and mean importances
Plot_label = ['Sum of Frequency', 'Mean Importance']
for i in Plot_label:
    # Plot the histogram of feature selection frequencies
    plt.figure(figsize=(15, 7))  # Set the figure size
    plt.bar(features_summary['Feature Index'], features_summary[i], color='steelblue')
    plt.xlabel('Feature Index')
    plt.ylabel(i)
    plt.title('Histogram of Feature ' + i)
    plt.tight_layout()
    new_string = i.replace(' ', '_')
    plt.savefig(os.path.join(file_path,'histogram_of_feature_' + new_string + '.png')) # Save the figure
    plt.show()

    # Plot the histogram of top 10 feature frequencies
    # Select the top 10 features based on Frequency
    top_10_features_summary = features_summary.nlargest(10, i)
    # Plot a horizontal bar chart for the top 10 features
    plt.figure(figsize=(15, 7))
    plt.barh(top_10_features_summary['Feature Name'],
             top_10_features_summary[i],
             color='steelblue')
    plt.xlabel(i)
    plt.ylabel('Feature Name')
    plt.title('Top 10 Features based on ' + i)
    plt.gca().invert_yaxis()  # to display the highest frequency at the top
    plt.tight_layout()
    new_string = i.replace(' ', '_')
    plt.savefig(os.path.join(file_path,'top_10_features_based_on_' + new_string + '.png')) # Save the figure
    plt.show()

# Plot the heatmap of feature importances across models
# Step 1: Sort the features based on their frequency and select the top 10
top_10_features = features_summary.nlargest(max_display, 'Sum of Frequency')
# Step 2: Extract the importances of these top 10 features for each of the best models
importance_columns = ['Importance_Model_' + str(i + 1) for i in range(num_models)]
top_10_importances = top_10_features[importance_columns]
# Step 3: Plot a heatmap for these top 10 features
plt.figure(figsize=(16, 10))
sns.heatmap(top_10_importances,
            cmap="YlGnBu",
            cbar_kws={'label': 'Feature Importance'},
            yticklabels=top_10_features['Feature Name'],
            xticklabels = list(range(1,X.shape[0]+1)),
            annot=False, fmt=".2f")
plt.xlabel("Models")
plt.ylabel("Feature Name")
plt.title('Feature Importances Across Models (Top ' + str(max_display) + ' Features by frequency)')
plt.tight_layout()
plt.savefig(os.path.join(file_path,'top_feature_importances_across_models(RFEfrequency).png')) # Save the figure
plt.show()

# Plot the heatmap of feature frequency across models
# Step 1: Sort the features based on their frequency and select the top 10
top_10_features = features_summary.nlargest(max_display, 'Mean Importance')
# Step 2: Extract the importances of these top 10 features for each of the best models
frequency_columns = ['Frequency_Model_' + str(i + 1) for i in range(num_models)]
top_10_frequency = top_10_features[frequency_columns]
# Step 3: Plot a heatmap for these top 10 features
plt.figure(figsize=(16, 10))
sns.heatmap(top_10_frequency,
            cmap="YlGnBu",
            cbar_kws={'label': 'Feature Frequency'},
            yticklabels=top_10_features['Feature Name'],
            xticklabels = list(range(1,X.shape[0]+1)),
            annot=False, fmt=".2f")
plt.xlabel("Models")
plt.ylabel("Feature Name")
plt.title("Feature Frequency Across Models (Top " + str(max_display) + " Features by mean importance)")
plt.tight_layout()
plt.savefig(os.path.join(file_path,'top_feature_frequency_across_models(RFEfrequency).png')) # Save the figure
plt.show()

# Decision Curve Analysis
def calculate_net_benefit(strategy, true_labels, pred_probs, thresholds):
    n = len(true_labels)
    net_benefit = []

    for p in thresholds:
        if strategy == 'treat_all_good':
            # Treat all patients as good responders
            tp = np.sum(true_labels == 1)
            fp = np.sum(true_labels == 0)
        elif strategy == 'treat_none_good':
            # Treat no patients as good responders
            tp = 0
            fp = 0
        elif strategy == 'prediction_model':
            # Use the prediction model to determine good responders
            tp = ((pred_probs >= p) & (true_labels == 1)).sum()
            fp = ((pred_probs >= p) & (true_labels == 0)).sum()

        # Calculate net benefit as true positives minus false positives weighted by the threshold probability
        benefit = tp - (fp * p / (1 - p))
        net_benefit.append(benefit / n - (p * np.mean(true_labels)))

    return net_benefit

# Define a range of threshold probabilities
thresholds = np.linspace(0, 1, 100)

# Calculate net benefits for different strategies
net_benefit_treat_all_good = calculate_net_benefit('treat_all_good', y, y_pred, thresholds)
net_benefit_treat_none_good = calculate_net_benefit('treat_none_good', y, y_pred, thresholds)
net_benefit_prediction_model = calculate_net_benefit('prediction_model', y, y_pred, thresholds)

# Plot Decision Curve Analysis
plt.figure(figsize=(10, 6))
plt.plot(thresholds, net_benefit_prediction_model, label='Prediction Model')
plt.plot(thresholds, net_benefit_treat_all_good, label='Treat All as Good Responders', linestyle='--')
plt.plot(thresholds, net_benefit_treat_none_good, label='Treat None as Good Responders', linestyle='--')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis for Treatment Response')
plt.legend()
plt.grid(True)
plt.show()

####################################      2.5 Correlation test      ####################################################
# Define the correlation method and ploting method
def compute_correlation(data, target, method):
    """
    Compute correlations between each column of data and target using the specified method.

    Parameters:
    - data (np.ndarray): Input data with each column treated as a feature.
    - target (np.ndarray): Target values.
    - method (str): Correlation method ('pearson', 'spearman', 'kendall', 'pointbiserial').

    Returns:
    - (correlations, p_values): Tuple of correlation values and p-values.
    """
    methods = {
        'spearman': spearmanr,

    }
    corr_func = methods[method]
    results = [corr_func(data[:, i], target) for i in range(data.shape[1])]
    correlations = np.array([item[0] for item in results])
    p_values = np.array([item[1] for item in results])
    return correlations, p_values
def plot_correlations(ax, data, p_values, title):
    """
    Plot correlation values with significance indicated by color.

    Parameters:
    - ax (matplotlib axis): Axis on which to plot.
    - data (np.ndarray): Correlation values.
    - p_values (np.ndarray): p-values associated with the correlations.
    - title (str): Title for the plot.
    """
    ax.bar(range(len(data)), data, color=np.where(p_values < 0.05, 'r', 'b'))
    ax.set_title('Spearman Rank Correlation Test of ' + title )
    ax.set_ylabel('Correlation Coefficient Values')
    ax.set_xlabel("Feature Index")

X_scaled = CustomScaler().fit_transform(X)
print("X_scaled shape: ", X_scaled.shape)
# Unprocessed X
A = X_scaled[:, : - Covariate_number]
# ComBat harmonization
model = CombatModel()
model.fit(X_scaled[:, : - Covariate_number],
                        sites = Group,
                        # continuous_covariates = Motion_Score
                        )
A = model.transform(X_scaled[:, : - Covariate_number],
                   sites = Group,
                   # continuous_covariates = Motion_Score
                   )

ComBat_transformer = CustomCombatTransformer(T1_features_start, DTI_features_end)
GAM_transformer = CustomGAMTransformer(T1_features_start, DTI_features_end)


pipeline1 = Pipeline([
    ('Scale', CustomScaler()),
    ('ComBat', ComBat_transformer),
    ('GAMs', GAM_transformer),
    # ('selector', CustomFeatureSelector(estimator=estimator)),
    ('clf', CustomXGBClassifier())
])

# pbounds = {
#     # Hyperparameters for Scale_Regress
#     'Scale__with_centering': Categorical([True, False]),  # Centering hyperparameter for RobustScaler
#     'Scale__with_scaling': Categorical([True, False]),  # Scaling hyperparameter for RobustScaler
#     # Hyperparameters for Harmonization (T1)
#     'Harmonization__T1_features__lam': Real(1e-6, 1e6, prior='log-uniform'),
#     # Hyperparameters for Harmonization (DTI)
#     'Harmonization__DTI_features__lam': Real(1e-6, 1e6, prior='log-uniform'),
#     # Hyperparameters for XGBClassifier
#     'clf__learning_rate': Real(0.01, 0.2, 'log-uniform'),
#     'clf__max_depth': Integer(2, 7),
#     'clf__subsample': Real(0.5, 1.0),
#     'clf__colsample_bytree': Real(0.5, 1.0),
#     'clf__min_child_weight': Integer(1, 20),
#     'clf__alpha': Real(0, 2),
#     'clf__lambda': Real(0, 2),
#     'clf__n_estimators': Integer(10, 150),
#     'clf__gamma': Real(0, 10)
# }

# Specify the hyperparameter space for BayesSearchCV
pbounds = {
    # Hyperparameters for Scale_Regress
    'Scale__centering': Categorical([True, False]),  # Centering hyperparameter for RobustScaler
    'Scale__scaling': Categorical([True, False]),  # Scaling hyperparameter for RobustScaler
    # Hyperparameters for Harmonization
    'GAMs__lam': Real(1e-6, 1e6, prior='log-uniform'),
    # Hyperparameters for the Selector RFE
    # 'selector__estimator__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    # 'selector__estimator__max_depth': Integer(2, 7),
    # 'selector__estimator__subsample': Real(0.5, 1.0),
    # 'selector__estimator__colsample_bytree': Real(0.5, 1.0),
    # 'selector__estimator__min_child_weight': Integer(1, 20),
    # 'selector__estimator__alpha': Real(0, 2),
    # 'selector__estimator__lambda': Real(0, 2),
    # 'selector__estimator__n_estimators': Integer(10, 150),
    # 'selector__estimator__gamma': Real(0, 10),
    # Hyperparameters for XGBClassifier
    'clf__learning_rate': Real(0.01, 0.2, 'log-uniform'),
    'clf__max_depth': Integer(2, 7),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__min_child_weight': Integer(1, 20),
    'clf__alpha': Real(0, 2),
    'clf__lambda': Real(0, 2),
    'clf__n_estimators': Integer(10, 150),
    'clf__gamma': Real(0, 10)
} # XGBoost hyperparameters

optimizer1 = BayesSearchCV(
    estimator=pipeline1,
    search_spaces=pbounds,
    n_iter=50,
    scoring = 'roc_auc',
    n_jobs=-1,
    cv=cv_inner,
    random_state=random_seed
    )


pipeline1 = pipeline1.fit(X,y)
X1 = pipeline1.steps[0][1].transform(X)
print("X shape in step 1: ", X1.shape)
X2 = pipeline1.steps[1][1].transform(X1)
print("X shape in step 2: ", X2.shape)
X3 = pipeline1.steps[2][1].transform(X2)
print("X shape in step 3: ", X3.shape)
X4 = pipeline1.steps[3][1].transform(X3)
print("X shape in step 4: ", X4.shape)

# Show the whole dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

optimizer1.fit(X,y)
model_best = optimizer1.best_estimator_.fit(X,y)
X1 = model_best.steps[0][1].transform(X)
print("X1 shape: ", X1.shape)
X2 = model_best.steps[1][1].transform(X1)
print("X2 shape: ", X2.shape)
X3 = model_best.steps[2][1].transform(X2)
print("X3 shape: ", X3.shape)

# histogram of each feature
# Setting up the figure size and grid for the subplots
plt.figure(figsize=(20, 10))
n_rows = 4
n_cols = 4
# Looping through each feature
for i in range(X2.shape[1]):
    plt.subplot(n_rows, n_cols, i+1)
    plt.hist(X2[:, i], bins=10)  # You can adjust the number of bins
    plt.title(feature_labels[i])
plt.tight_layout()
plt.show()

# Define the file path
file_path = 'enter_your_input_file_path'
# Ensure the directory exists
os.makedirs(file_path, exist_ok=True)
# transfer numpy to dataframe
df = pd.DataFrame(X3)
# df1 = df.iloc[:, : - Covariate_number]
df1 = df
df1.columns = feature_names
# Save the DataFrame to a CSV file in the specified directory
df1.to_csv(os.path.join(file_path, 'X_Scale_ComBat_GAMs_44.csv'), index=False)

# neuroHarmonize
# load your data and all numeric covariates
my_data = pd.read_csv('enter_your_input_data_X_csv_file_path')
my_data = np.array(my_data)
covars = pd.read_csv('enter_your_covariates_csv_file_path')

# run harmonization with NONLINEAR effects of age
my_data = RobustScaler().fit_transform(my_data)
my_model_T1, my_data_T1, s_data_T1 = harmonizationLearn(my_data[:, T1_features_start: T1_features_end],
                                          covars, smooth_terms=['Coefficient of Joint Variation'],
                                          return_s_data = True
                                          )
my_model_DTI, my_data_DTI, s_data_DTI = harmonizationLearn(my_data[:, DTI_features_start: DTI_features_end],
                                          covars, smooth_terms=['DTI Motion Score'],
                                          return_s_data = True
                                          )

A = np.concatenate((s_data_T1, s_data_DTI), axis=1)

# A = np.concatenate((my_data_T1, my_data_DTI), axis=1)

print("A shape: ", A.shape)

# Load the covariates file
covars = pd.read_csv('enter_your_covariates_csv_file_path')
file_path = 'enter_the_output_file_path'
# Define the column names to convert
columns_to_convert = ['Group', 'age_data', 'Motion_Score', 'CJV']
# A dictionary to store the converted arrays
converted_arrays = {}
for column in columns_to_convert:
    converted_arrays[column] = np.float64(covars[column].to_numpy())
# Now you can access each array using its column name
Group = converted_arrays['Group']
age_data = converted_arrays['age_data']
Motion_Score = converted_arrays['Motion_Score']
CJV = converted_arrays['CJV']

# Define your target variables and their corresponding labels, grouped as desired
target_groups = [
    [
        {'B': Group, 'ylabel': 'Age Group - Dichotomous'},
        {'B': age_data, 'ylabel': 'Age Group - Continuous'}
    ],
    [
        {'B': Motion_Score, 'ylabel': 'Motion Score'},
        {'B': CJV, 'ylabel': 'Coefficient of Joint Variation'}
    ]
]
# Loop over each target group
for target_group in target_groups:
    # Create a new figure for each group
    fig, axs = plt.subplots(len(target_group), 1, figsize=(15, 8))  # Adjust the size as needed
    # If there is only one target in the group, axs will not be a list. We need to convert it to a list.
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    # Loop over each target in the group
    for i, target in enumerate(target_group):
        B = target['B']
        ylabel = target['ylabel']
        # Compute correlations for 'spearman' method
        spearman_correlations, spearman_p_values = compute_correlation(A, B, 'spearman')
        # Plot the correlations in the subplot
        plot_correlations(axs[i], spearman_correlations, spearman_p_values, ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, 'correlation_test.png')) # Save the figure
    plt.show()


# BEST analysis
import best
from best import analyze_two, plot_all
df_X = pd.DataFrame(X3)
# save the dataframe
df_X.to_csv(os.path.join(file_path,'X3.csv'), index=False)
for column_name in df_X.columns:
    # Select the column corresponding to the current feature
    feature_data = df_X[column_name]
    # Split the feature data into responders and non-responders based on y
    responders = feature_data[y == 1]
    non_responders = feature_data[y == 0]
    # Perform BEST analysis
    best_analysis = best.analyze_two(non_responders, responders)
    # Plot the analysis
    best.plot_all(best_analysis)
    # Save the plot with the feature name as the filename
    plt.savefig(os.path.join(file_path,f'{column_name}.png'))
    plt.close()


# Violin plot of all features
df = pd.read_csv('enter_the_input_data_X_csv_file_path')
groups = pd.read_csv('enter_the_y_csv_file_path', header=None)
file_path_1 = 'enter_the_output_folder_path'
# Add the group information to the dataframe
df['group'] = groups
# Rename the group names
df['group'] = df['group'].map({0: 'Poor Responder', 1: 'Good Responder'})
# Melt the dataframe to long-form for use with seaborn
df_long = df.melt(id_vars='group', var_name='feature', value_name='value')
# Initialize the matplotlib figure
plt.figure(figsize=(20, 10))
# Create a violinplot with seaborn
sns.violinplot(x='feature', y='value', hue='group', data=df_long, split=True, inner="quartile",hue_order = ['Good Responder', 'Poor Responder'],
               palette= {"Poor Responder": "#3687ca", "Good Responder": "#f05d8d"}, satuation = 1)
plt.axhline(0, color='gray', linestyle='--')
# plt.title('Features Grouped by On-medication Treatment Response', fontsize = 24)
# Improve the legibility of the plot
plt.xticks(rotation=90, fontsize = 20)  # Rotate the x labels for better readability
plt.xlabel('')
plt.ylabel('Feature Value', fontsize = 18)
plt.legend(title='Response', loc='upper center', fontsize = 18, title_fontsize=20)
plt.tight_layout()  # Adjust layout to fit all the x labels
# Display the plot
# plt.savefig(os.path.join(file_path_1,'violin_plot_total.png')) # Save the figure
plt.show()

# Violin plot of selected features
# Load the feature data
file_path_1 = 'Enter_the_output_folder_path'
# Load the feature data
df = pd.read_csv('enter_the_X_csv_file_path')
# Load the group data
groups_6 = pd.read_csv('enter_the_y_CGI6_csv_file_path', header=None)
groups_7 = pd.read_csv('enter_the_y_CGI7_csv_file_path', header=None)
# Assume the first column is the feature of interest
# feature_of_interest = 'Left_Accumbens_Area_VoxelVolume'  # Replace with your actual feature name
# feature_of_interest = 'Right_Putamen_VoxelVolume'
feature_of_interest = 'ATR_Left_MeanFA'
groups_6['Feature Value'] = df[feature_of_interest]
groups_7['Feature Value'] = df[feature_of_interest]
# Now create a 'Group' column in both dataframes
# Rename the response values
groups_6[0] = groups_6[0].map({0: 'Poor Responder', 1: 'Good Responder'})
groups_7[0] = groups_7[0].map({0: 'Poor Responder', 1: 'Good Responder'})
# Now create a 'Group' column in both dataframes and rename
groups_6['Treatment Response Evaluation'] = 'On-medication'
groups_7['Treatment Response Evaluation'] = 'Off-medication'
# Concatenate the group dataframes
combined_groups = pd.concat([groups_6, groups_7], ignore_index=True)
combined_groups.columns = ['Response', 'Feature Value', 'Treatment Response Evaluation']  # Rename columns for clarity
# Initialize the matplotlib figure
plt.figure(figsize=(10, 7))
# Create a violin plot with seaborn
sns.violinplot(x='Treatment Response Evaluation', y='Feature Value',
               hue='Response', data=combined_groups, split=True, inner='quartile',
               hue_order = ['Good Responder', 'Poor Responder'],
               palette= {"Poor Responder": "#3687ca", "Good Responder": "#f05d8d"}, saturation=1)
# Improve the legibility of the plot
plt.axhline(0, color='#5a595b', linestyle='--')
# plt.title(f'ROI volume in left accumbens area', fontsize = 30)
# plt.title(f'ROI volume in right putamen', fontsize = 30)
plt.title(f'Mean FA in left ATR', fontsize = 30)
plt.xlabel('Medication status', fontsize = 25)
plt.ylabel('Feature value [AU]', fontsize = 25)
plt.tick_params(labelsize=24)  # Tick labels
plt.legend(title='Response group', loc= 'upper center', fontsize = 15, title_fontsize = 15)
# Add annotation "A" to the upper left corner
# plt.text(-0.75, 5.6, 'A', fontsize=80, fontweight='bold', va='top', ha='left')
# plt.text(-0.75, 2.9, 'B', fontsize=80, fontweight='bold', va='top', ha='left')
plt.text(-0.75, 2.9, 'C', fontsize=80, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig(os.path.join(file_path_1, 'violin_plot' + feature_of_interest + '.png'))
# Display the plot
plt.show()

########################################################################################################################
# Number of features calculation
print ('Number of samples in training dataset: ', X.shape[0])
print ('Number of features to select according to Abu-Mostafa: ', int(X.shape[0]/10))
print ('Number of features to select according to Hua: ', X.shape[0] ** 0.5)

# Harmonization
X = pd.read_csv('enter_the_X_csv_file_path')
# Store the feature names
feature_names = X.columns.to_list()
# Convert to numpy arrays
X = np.float64(X.to_numpy())
# Load the covariates
covariates = pd.read_csv('enter_the_covariates_csv_file_path')
# Group = (np.ravel(covariates['Group']).astype(int)).reshape(1, -1).T
Outcome = (np.ravel(covariates['CGI 7']).astype(int)).reshape(1, -1).T
# Motion_Score = (np.float64(covariates['TP1_DTI_Motion_Score'].to_numpy())).reshape(1, -1).T
# Apply ComBat to harmonize the data
combat = CombatModel()
# fit_transform is not available for continuous covariates, so I had to split the fit and transform steps
# X_T1 = combat.fit(data=X, sites=Group, discrete_covariates=None, continuous_covariates=None)
# X_T1 = combat.transform(data=X, sites=Group, discrete_covariates=None, continuous_covariates=None)
X_T1 = combat.fit(data=X, sites=Outcome, discrete_covariates=None, continuous_covariates=None)
X_T1 = combat.transform(data=X, sites=Outcome, discrete_covariates=None, continuous_covariates=None)
# X_DTI = combat.fit(data=X, sites=Group,discrete_covariates=None, continuous_covariates=Motion_Score)
# X_DTI = combat.transform(data=X, sites=Group,discrete_covariates=None, continuous_covariates= Motion_Score)
df = pd.DataFrame(X_T1, columns=feature_names)
df.to_csv('enter_the_output_folder_path', index=False)

# Feature Basic statistics plots
X = pd.read_csv('enter_the_X_csv_file_path')
feature_names = X.columns.to_list()
parameters = {
   'feature_path': "enter_the_X_csv_file_path", # path to csv/xls file with features
    'outcome_path': "enter_the_y_csv_file_path", #path to csv/xls file with outcome (in our case - the same file)
    'patient_column': 'Patients_ID', # name of column with patient ID
    'patient_in_outcome_column': 'Patients_ID', # name of column with patient ID in clinical data file
    'outcome_column': 'Outcome', # name of outcome column
    'feature_column': feature_names # name of column with features
}

fs = AnalysisBox(**parameters)
print('Initial amount of features: ', len(parameters['feature_column']))
fs.handle_nan(axis=1)
print('Amount of features after exclusion of NaN values: ', len(parameters['feature_column']))
fs.plot_distribution(fs._feature_column)
fs.plot_correlation_matrix(fs._feature_column)
fs.plot_MW_p(fs._feature_column)
fs.plot_univariate_roc(fs._feature_column, auc_threshold = 0.70)
fs.calculate_basic_stats()
print('Basic statistics for each feature')


file_path = "enter_the_output_folder_path"

X = X.drop(X.columns[0], axis=1)
Left = ['Left_Caudate', 'Left_Hippocampus', 'Left_Accumbens_Area', 'Left_Pallidum', 'Left_Putamen']
Right = ['Right_Caudate', 'Right_Hippocampus', 'Right_Accumbens_Area', 'Right_Pallidum', 'Right_Putamen']
correlation_matrix_left_right = pd.DataFrame(index=Left, columns=Right)
for left_col in Left:
    for right_col in Right:
        correlation_matrix_left_right.loc[left_col, right_col] = X[left_col].corr(X[right_col])
correlation_matrix_left_right = correlation_matrix_left_right.astype(float)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_left_right, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Features Correlation Matrix')
plt.ylabel('Left Hemisphere')
plt.xlabel('Right Hemisphere')
plt.xticks(rotation=0)  # Rotate x-axis labels to horizontal
plt.tight_layout()
plt.savefig(os.path.join(file_path,'Features Correlation Matrix.png'))
plt.show()

#######################################################################################################################
# Function to compute Mahalanobis distance for a subgroup
def compute_mahalanobis(group):
    # Extract CJV and Motion Score columns
    values = group[['Coefficient of Joint Variation', 'DTI Motion Score']].values

    # Compute mean and inverse covariance matrix
    mean = np.mean(values, axis=0)
    inv_cov = np.linalg.inv(np.cov(values, rowvar=False))

    # Calculate Mahalanobis distance for each row
    group['Mahalanobis'] = [distance.mahalanobis(row, mean, inv_cov) for row in values]
    return group


# Group by Age and Treatment response, then apply the function
result = covariates.groupby(['Group', 'CGI 6']).apply(compute_mahalanobis)
result.to_csv('enter_the_output_csv_path', index=False)
print(result)

# Plot the results
sns.catplot(x="Group", y="Mahalanobis", data=result, kind="violin", col="CGI 6")
plt.show()

g = sns.FacetGrid(result, col="CGI 6", row="Group", margin_titles=True)
g.map(sns.boxplot, "Mahalanobis")
plt.show()
########################################################################################################################
# Load the top3 features
Feature_top3_csv = pd.read_csv('enter_the_X_csv_file_path')
# Feature_top3_label = 'Right_Accumbens_Area_shape_Sphericity'
# Feature_top3_label = 'Right_Putamen_shape_SurfaceVolumeRatio'
Feature_top3_label = 'Right_Accumbens_Area_shape_Flatness'
Feature_top3 = (np.float64(Feature_top3_csv[Feature_top3_label].to_numpy())).reshape(1, -1).T
# Load the quality check
Quality_check_csv = pd.read_csv('enter_the_covariate_csv_file_path')
# Quality_check_label = 'TP1_DTI_Motion_Score'
Quality_check_label = 'Coefficient of Joint Variation'
Quality_check = (np.float64(Quality_check_csv[Quality_check_label].to_numpy())).reshape(1, -1).T
# Compute correlation coefficient and p-value
slope, intercept, r_value, p_value, _ = linregress(Feature_top3.squeeze(), Quality_check.squeeze())

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(Feature_top3, Quality_check, label='Data Points')
plt.plot(Feature_top3, intercept + slope * Feature_top3, color='red', label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
plt.title(f'Correlation Plot (R={r_value:.2f}, P-value={p_value:.4f})')
plt.xlabel(Feature_top3_label)
plt.ylabel(Quality_check_label)
plt.legend()
plt.grid(True)
plt.show()
########################################################################################################################
# Box Plot
df = pd.read_csv("enter_the_X_csv_file_path")
# df = pd.read_csv("enter_the_covariate_csv_file_path")
column_label = "Coefficient of Joint Variation"
# column_label = "DTI Motion Score"
group_a = df[df['Group'] == 1][column_label]
group_b= df[df['Group'] == 2][column_label]
# group_a = df[df['CGI 6'] == 1][column_label]
# group_b= df[df['CGI 6'] == 0][column_label]
# group_a = df[df['Children'] == 1][column_label]
# group_b= df[df['Children'] == 0][column_label]
fig, ax = plt.subplots()
ax.boxplot([group_a, group_b], labels=['Children', 'Adults'])
# ax.boxplot([group_a, group_b], labels=['Respondents', 'Non-respondents'])
# Perform the independent t-test
t_stat, t_p_value = ttest_ind(group_a, group_b)
# Display the p-value on the plot
ax.text(0.4, 0.9, f'p-value = {t_p_value:.4f}', transform=ax.transAxes)
plt.title('Box plot of ' + column_label + ' by age group')
plt.xlabel('Age Group')
# plt.title('Box plot of ' + column_label + ' by outcome (CGI 6)')
# plt.xlabel('Treatment Outcome (CGI 6)')
plt.ylabel(column_label)
plt.show()
########################################################################################################################
# Scatter plot
df_complex_updated = pd.read_csv("enter_your_covariate.csv")

# Initialize the plot again
plt.figure(figsize=(12, 8))

# Initialize a dictionary to hold the correlation coefficients
correlation_dict_updated = {}

# Function to plot and calculate correlation for each subgroup
def plot_and_calculate_correlation_updated(group, cgi, color, marker, label):
    x_data = df_complex_updated.loc[
        (df_complex_updated['Group'] == group) & (df_complex_updated['CGI 7'] == cgi), 'Coefficient of Joint Variation']
    y_data = df_complex_updated.loc[
        (df_complex_updated['Group'] == group) & (df_complex_updated['CGI 7'] == cgi), 'DTI Motion Score']
    plt.scatter(x_data, y_data, color=color, marker=marker)

    # Calculate Pearson correlation
    corr, _ = pearsonr(x_data, y_data)
    label += f' (Corr: {corr:.2f})'
    correlation_dict_updated[label] = corr
    return label

# Plotting and calculating correlation for each subgroup
labels_updated = []
labels_updated.append(plot_and_calculate_correlation_updated(1, 2, 'r', 'x', 'Children, Poor Respondents'))
labels_updated.append(plot_and_calculate_correlation_updated(1, 1, 'r', 'o', 'Children, Good Respondents'))
labels_updated.append(plot_and_calculate_correlation_updated(2, 2, 'b', 'x', 'Adults, Poor Respondents'))
labels_updated.append(plot_and_calculate_correlation_updated(2, 1, 'b', 'o', 'Adults, Good Respondents'))

# Add labels and title
plt.xlabel('Coefficient of Joint Variation', fontsize = 20)
plt.ylabel('DTI Motion Score', fontsize = 20)
# plt.title('Scatter Plot with Multiple Groupings and Correlation Analysis with On-medication Evaluation', fontsize = 20)
plt.legend(labels_updated, loc='lower right', fontsize = 12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Create a separate plot for the labels
plt.figure(figsize=(8, 6))
plt.axis('off')  # Turn off the axis

# Display the legend in a separate plot
plt.legend(labels_updated, loc='center', fontsize=14)
plt.show()


########################################################################################################################
# Confusion matrix
# Load the data
df = pd.read_csv("Enter_your_covariate.csv")
group_a = df['CGI 6']
group_b= df['Group']
# Compute the confusion matrix
cm = confusion_matrix(group_a, group_b)
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", yticklabels=["Respondents", "Non-respondents"], xticklabels=["Children", "Adults"])
plt.title("Confusion Matrix of CGI 6 and Age Group")
plt.show()

########################################################################################################################
# Statistical comparison of different model (LOOCV)
import numpy as np
from scipy.stats import chi2, binomtest
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# read npy file of y
y = pd.read_csv('enter_the_y_csv_file_path', header=None)
y = y.to_numpy().flatten()
y_pred_tier1 = np.load('enter_your_model_1_y_pred.npy')
# y_pred=y_pred_tier1
y_pred_tier2 = np.load('enter_your_model_2_y_pred.npy')

i = 1
for y_tier in [y_pred_tier1, y_pred_tier2]:
    y_pred = y_tier
    y_pred1= (y_pred >= 0.5).astype(int)
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y, y_pred)
    # Calculate Precision-Recall AUC
    prc_auc = average_precision_score(y, y_pred)
    print("Tier ", i, " results:")
    balanced_accuracy = balanced_accuracy_score(y, y_pred1)
    precision = precision_score(y, y_pred1)
    recall = recall_score(y, y_pred1)
    f1 = f1_score(y,y_pred1)
    print(f"Balanced Accuracy: {round(balanced_accuracy,2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"f1: {round(f1, 2)}")
    print(f"ROC AUC: {round(roc_auc, 2)}")
    print(f"PRC AUC: {round(prc_auc, 2)}")
    i = i + 1

y_pred_tier1 = (y_pred_tier1 >= 0.5).astype(int)
y_pred_tier2 = (y_pred_tier2 >= 0.5).astype(int)
# Constructing the contingency table
a = np.sum((y_pred_tier1 == y) & (y_pred_tier2 == y))
b = np.sum((y_pred_tier1 == y) & (y_pred_tier2 != y))
c = np.sum((y_pred_tier1 != y) & (y_pred_tier2 == y))
d = np.sum((y_pred_tier1 != y) & (y_pred_tier2 != y))

n = b + c

# Constructing the contingency table
contingency_table = [[0, 0], [0, 0]]

for true, pred1, pred2 in zip(y, y_pred_tier1, y_pred_tier2):
    if pred1 == pred2:
        if pred1 == true:
            contingency_table[0][0] += 1  # Both models correct
        else:
            contingency_table[1][1] += 1  # Both models incorrect
    else:
        if pred1 == true:
            contingency_table[0][1] += 1  # Only Model 1 correct
        else:
            contingency_table[1][0] += 1  # Only Model 2 correct

if n >= 25: # If b and c are large enough, calculating p-value using McNemar's statistic
    mcnemar_result = mcnemar(contingency_table, exact=False)
    print (f" b + c = ", n, " >= 25, McNemar's test")
    print(f"Chi-squared: {mcnemar_result.statistic}")
    print(f"P-value: {mcnemar_result.pvalue}")
    chi2_stat = ((b - c) ** 2) / (b + c)
    p_value = chi2.sf(chi2_stat, 1)
    print(f"Chi-squared: {chi2_stat}")
    print(f"P-value of Tier 1 vs Tier 2: {round(p_value, 3)}")
elif n < 25: # If either b or c is small, calculating p-value using binomial test
    mcnemar_result = mcnemar(contingency_table, exact=True)
    print(f" b + c = ", n, " < 25, Exact binomial test")
    print(f"Chi-squared: {mcnemar_result.statistic}")
    print(f"P-value: {round(mcnemar_result.pvalue, 3)}")
    # Counting successes for each model
    successes_1 = np.sum(y == y_pred_tier1)
    successes_2 = np.sum(y == y_pred_tier2)
    # Benchmark probability (e.g., random guessing)
    benchmark_probability = 0.5
    # Apply the Exact Binomial Test for each model
    p_value_1 = binomtest(successes_1, len(y), benchmark_probability)
    p_value_2 = binomtest(successes_2, len(y), benchmark_probability)
    print(f"P-value of Tier 1 vs random guessing: {p_value_1}")
    print(f"P-value of Tier 2 vs random guessing: {p_value_2}")

###############################################################################################################
# Impute the missing values of treatment outcome
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Assuming 'df' is your DataFrame
scaler = StandardScaler()
df = pd.read_csv('enter_the_input_y_csv_file_path', na_values = [' '])
# Create indicator for missing values
df1 = df.isna().sum()
print(df1)

df2 = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df2, columns=df.columns)

# Initialize the KNN Imputer with k=1 for nearest neighbor
imputer = KNNImputer(n_neighbors=1)

# Apply the imputer within each group
df3 = imputer.fit_transform(df2)
df_imputed = pd.DataFrame(df3, columns=df.columns)

# Reverse the standardization/normalization process
df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df.columns)
print(df_imputed)

df_imputed.to_csv('enter_the_output_y_csv_file_path', index=False)
