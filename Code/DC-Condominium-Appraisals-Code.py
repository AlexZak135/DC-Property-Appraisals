# Title: DC Condominium Appraisals Analysis
# Author: Alexander Zakrzeski
# Date: March 4, 2025

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data
import numpy as np
import pandas as pd

# Load to produce data visualizations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Load to run statistical tests
from scipy.stats import pearsonr, pointbiserialr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load to train, test, and evaluate machine learning models
from catboost import CatBoostRegressor, Pool
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error 
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Define a function for control flow using the conditions and choices
def conditional_map(*conditional_choice_pairs): 
    # Map conditions to corresponding choices 
    mapped_array = np.select( 
      condlist = conditional_choice_pairs[0::2],
      choicelist = conditional_choice_pairs[1::2], 
      default = pd.NA  
      )  

    # Return the array
    return mapped_array

# Define a function to remove trailing ".0" from numeric strings
def remove_dot_zero(series):
    # Convert the series to strings and remove any trailing ".0"
    series = series.astype(str)
    series = series.str.replace(r"\.0$", "", regex = True)
    
    # Return the series
    return series

# Define a function to plot a distribution using a histogram
def generate_histogram(column, value):   
    # Create a histogram to display the distribution
    sns.histplot(appraisals, x = column, bins = 20, kde = True, 
                 color = "#0078ae") 
    plt.title(label = f"Distribution of {value}", fontsize = 17)
    plt.xlabel(xlabel = value, fontsize = 14)
    plt.ylabel(ylabel = "Frequency", fontsize = 14)
    sns.despine(left = True)
    plt.gca().xaxis.grid(visible = False)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter( 
      lambda x, _: f"{x:,.1f}".rstrip("0").rstrip("."))  
      )
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter( 
      lambda x, _: f"{x:,.1f}".rstrip("0").rstrip(".")) 
      )
    
    # Display the plot
    plt.show()

# Define a function to plot distributions using a box plot
def generate_box_plot(column, value):
    # Create a box plot to display the distributions
    sns.boxplot(appraisals, x = column, y = "log_price", 
                order = sorted(appraisals[column].unique()), color = "#0078ae")
    plt.title(label = f"Distribution of the Log of Price by {value}", 
              fontsize = 13)
    plt.xlabel(xlabel = "")
    plt.ylabel(ylabel = "")
    sns.despine(left = True, bottom = True)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter( 
      lambda x, _: str(x).rstrip(".0")) 
      )
    
    # Display the plot
    plt.show() 

# Define a function to prepare the training and test data 
def process_and_split(model):
    # Drop the columns that will no longer be used
    processed = (appraisals. 
      drop(columns = ["saledate", "saledate_y", "rmdl", "bedrms", "ac", 
                      "price"])) 
    
    # Appropriately create dummy variables from the categorical variables
    if model == "lr": 
        processed = pd.get_dummies(processed, drop_first = True)
    elif model in ["knn", "svm", "rf", "xgb"]:  
        processed = pd.get_dummies(processed, drop_first = False) 
        
    # Modify the values of the column to display integers instead of booleans
    if model != "cb":
        processed[processed.select_dtypes(include = "bool").columns] = ( 
          processed.select_dtypes(include = "bool").astype(int) 
          )  
        
    # Perform a train-test split and reset the index of certain dataframes
    x_train, x_test, y_train, y_test = train_test_split( 
      processed.drop(columns = "log_price"), processed["log_price"], 
      test_size = 0.2, shuffle = False 
      )
    x_test, y_test = [ 
      dataframe.reset_index(drop = True) for dataframe in [x_test, y_test] 
      ]
    
    # Return the dataframes
    return x_train, x_test, y_train, y_test

# Part 2: Data Preprocessing

# Load the data from the CSV file
appraisals = pd.read_csv("DC-Condominium-Appraisals-Data.csv")

# Make all column names lowercase, rename specific ones, and drop columns
appraisals = (appraisals.
  rename(columns = str.lower).  
  rename(columns = {"bedrm": "bedrms",
                    "bathrm": "bathrms", 
                    "hf_bathrm": "hf_bathrms",
                    "fireplaces": "fireplace",
                    "living_gba": "unit_gba"}).
  drop(columns = ["bldg_num", "cmplx_num", "eyb", "heat", "gis_last_mod_dttm", 
                  "objectid"]))

# Filter based on multiple conditions to create a subsetted dataframe
appraisals = appraisals[  
  (appraisals["ayb"].between(1900, 2024)) &
  (appraisals["ayb"] <= pd.to_datetime(appraisals["saledate"]).dt.year) & 
  (((appraisals["yr_rmdl"].between(appraisals["ayb"], 2024)) & 
    (appraisals["yr_rmdl"] <= pd.to_datetime(appraisals["saledate"]). 
                              dt.year)) | 
   (appraisals["yr_rmdl"].isna())) & 
  (appraisals["rooms"].between(1, 6)) &
  (appraisals["rooms"] > appraisals["bedrms"]) & 
  (appraisals["bedrms"] <= 3) &
  (appraisals["bathrms"].between(1, 3)) &
  (appraisals["hf_bathrms"] <= 1) &
 ~((appraisals["heat_d"].isin(["Air Exchng", "Evp Cool", "Ind Unit", 
                               "No Data"])) | 
   ((appraisals["heat_d"] == "Warm Cool") & (appraisals["ac"] == "N"))) & 
  (appraisals["ac"] != "0") & 
  (appraisals["fireplace"] <= 1) &
  (pd.to_datetime(appraisals["saledate"]).dt.normalize(). 
   between(pd.Timestamp(2020, 1, 1, tz = "UTC"), 
           pd.Timestamp(2024, 12, 31, tz = "UTC"))) &
  (appraisals["price"].between(150000, 1750000)) & 
  (appraisals["qualified"] == "Q") &
  (appraisals["sale_num"] <= 8) &
  (appraisals["unit_gba"].between(500, 1800)) &
  (appraisals["usecode"].isin([16, 17])) &
  (appraisals["landarea"].between(0, 1500)) 
  ]
  
# Modify the values of existing columns and create new columns
appraisals["ssl"] = appraisals["ssl"].str.replace(r"\s{2,}", " ", regex = True) 
appraisals["rmdl"] = conditional_map( 
 ~appraisals["yr_rmdl"].isna(), "Yes",
  True, "No" 
  )
appraisals["rooms"] = conditional_map(   
  appraisals["rooms"] <= 2, "2 or Fewer",
 (appraisals["rooms"] >= 3) & (appraisals["rooms"] <= 4),
  remove_dot_zero(appraisals["rooms"]),
  appraisals["rooms"] >= 5, "5 or More"  
  ) 
appraisals["bedrms"] = conditional_map( 
  appraisals["bedrms"] <= 1, "1 or Fewer", 
 (appraisals["bedrms"] >= 2) & (appraisals["bedrms"] <= 3), 
  remove_dot_zero(appraisals["bedrms"])
  )
appraisals["ttl_bathrms"] = conditional_map( 
  appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5) <= 2,
  remove_dot_zero(appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5)),
  True, "2.5 or More"  
  )
appraisals["heat_d"] = conditional_map( 
  appraisals["heat_d"] == "Forced Air", "Forced Air",
  appraisals["heat_d"] == "Ht Pump", "Heat Pump",
  appraisals["heat_d"] == "Hot Water Rad", "Hot Water",
  True, "Other"
  )
appraisals["ac"] = appraisals["ac"].replace({"Y": "Yes", "N": "No"})
appraisals["fireplace"] = appraisals["fireplace"].map({1: "Yes", 0: "No"})
appraisals["age"] = conditional_map(
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 10, 
  "10 or Fewer",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 11) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 40)),
  "11 to 40",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 41) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 60)), 
  "41 to 60",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 61) & 
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 80)), 
  "61 to 80",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 81) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 90)), 
  "81 to 90",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 91) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 100)), 
  "91 to 100",  
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 101, 
  "101 or More" 
  )
appraisals["saledate"] = pd.to_datetime(appraisals["saledate"]).dt.date
appraisals["saledate_ym"] = (pd.to_datetime(appraisals["saledate"]). 
                             dt.to_period("M").dt.start_time.dt.date)
appraisals["saledate_y"] = conditional_map( 
  pd.to_datetime(appraisals["saledate"]).dt.year.isin([2023, 2024]), "2023+", 
  True, pd.to_datetime(appraisals["saledate"]).dt.year.astype(str) 
  )
appraisals["log_price"] = np.log(appraisals["price"])
appraisals["log_unit_gba"] = np.log(appraisals["unit_gba"])

# Load the data from the CSV file
addresses = pd.read_csv("DC-Addresses-Data.csv", usecols = ["MAR_ID", "WARD"])

# Rename columns and modify values of columns
addresses = addresses.rename(columns = str.lower)
addresses["mar_id"] = addresses["mar_id"].astype(str)
addresses["ward"] = addresses["ward"].str.replace(r"^Ward ", "", regex = True)

# Load the data from the CSV file
units = pd.read_csv("DC-Residential-Units-Data.csv", usecols = ["CONDO_SSL", 
                                                                "MAR_ID"])

# Rename columns, drop rows with missing values, and modify values of columns
units = (units. 
  rename(columns = {"CONDO_SSL": "ssl", 
                    "MAR_ID": "mar_id"}). 
  dropna())
units["ssl"] = units["ssl"].str.replace(r"\s{2,}", " ", regex = True)
units["mar_id"] = units["mar_id"].astype(str)

# Drop duplicates, reset the index, and reorder the column
units = units.drop_duplicates().reset_index(drop = True)
units.insert(0, "mar_id", units.pop("mar_id"))

# Perform an inner join and reorder the column
addresses = (addresses.
  merge(units, on = "mar_id", how = "inner").
  drop(columns = "mar_id"))
addresses.insert(0, "ssl", addresses.pop("ssl"))

# Perform a left join, drop rows with missing values, and drop columns
appraisals = (appraisals.
  merge(addresses, on = "ssl", how = "left"). 
  dropna(subset = "ward"). 
  drop(columns = ["ssl", "ayb", "yr_rmdl", "bathrms", "hf_bathrms", 
                  "qualified", "sale_num", "unit_gba", "usecode", 
                  "saledate_ym"])) 

# Reorder the columns, sort the rows in ascending order, and reset the index
appraisals.insert(0, "saledate", appraisals.pop("saledate"))
appraisals.insert(1, "saledate_y", appraisals.pop("saledate_y"))
appraisals.insert(2, "ward", appraisals.pop("ward"))
appraisals.insert(3, "age", appraisals.pop("age"))
appraisals.insert(4, "rmdl", appraisals.pop("rmdl"))
appraisals.insert(7, "ttl_bathrms", appraisals.pop("ttl_bathrms"))
appraisals.insert(13, "price", appraisals.pop("price"))
appraisals.insert(14, "log_price", appraisals.pop("log_price"))
appraisals.insert(11, "log_unit_gba", appraisals.pop("log_unit_gba"))
appraisals = appraisals.sort_values(by = "saledate").reset_index(drop = True)

# Part 3: Exploratory Data Analysis

# Globally set the size and theme for visualizations
plt.rcParams["figure.figsize"] = (8, 6) 
sns.set_style("whitegrid")

# Generate summary statistics
print(appraisals.select_dtypes("number").describe().map(lambda x: f"{x:.2f}"))

# Create histograms with overlaid KDEs to display distributions 
generate_histogram("price", "Price")
generate_histogram("log_price", "Log of Price")

# Select columns and modify the values of existing columns
sr_corr_inputs = (appraisals            
  [["log_price", "log_unit_gba", "ttl_bathrms", "bedrms", "rooms", "age", 
    "landarea", "saledate_y"]].copy())
sr_corr_inputs["ttl_bathrms"] = sr_corr_inputs["ttl_bathrms"].map({ 
  "1": 1, "1.5": 2, "2": 3, "2.5 or More": 4 
  })
sr_corr_inputs["bedrms"] = sr_corr_inputs["bedrms"].map({ 
  "1 or Fewer": 1, "2": 2, "3": 3
  })
sr_corr_inputs["saledate_y"] = sr_corr_inputs["saledate_y"].map({ 
  "2020": 1, "2021": 2, "2022": 3, "2023+": 4 
  }) 
sr_corr_inputs["rooms"] = sr_corr_inputs["rooms"].map({ 
  "2 or Fewer": 1, "3": 2, "4": 3, "5 or More": 4
  })
sr_corr_inputs["age"] = sr_corr_inputs["age"].map({ 
  "10 or Fewer": 1, "11 to 40": 2, "41 to 60": 3, "61 to 80": 4, "81 to 90": 5,
  "91 to 100": 6, "101 or More": 7
  })  

# Rename columns and calculate Spearman's rank correlation coefficients 
sr_corr_inputs = (sr_corr_inputs.
  rename(columns = {"log_price": "Log of Price",
                    "log_unit_gba": "Log of Unit GBA", 
                    "ttl_bathrms": "Bathrooms",
                    "bedrms": "Bedrooms",
                    "rooms": "Rooms",
                    "age": "Age",
                    "landarea": "Land Area",
                    "saledate_y": "Year"}).
  corr(method = "spearman").round(2))

# Create a correlation matrix to display the correlation coefficients 
corr_matrix = sns.heatmap( 
  sr_corr_inputs, vmin = -1, vmax = 1, 
  cmap = sns.diverging_palette(h_neg = 10, h_pos = 240, as_cmap = True), 
  annot = True, annot_kws = {"color": "#000000"}, linewidths = 0.5, 
  linecolor = "#000000",
  cbar_kws = {"format": ticker.FuncFormatter( 
    lambda x, _: f"{int(x)}" if x in [-1, 0, 1] else f"{x:.2f}" 
    )}, 
  xticklabels = True, yticklabels = True 
  )
corr_matrix.set_title(label = "Spearman's Rank Correlation Matrix", 
                      fontdict = {"fontsize": 17}, pad = 22) 
corr_matrix.tick_params(bottom = False, left = False)
plt.xticks(rotation = 45, ha = "right")
plt.show()

# Select columns and modify values in columns
pb_corr_inputs = appraisals[["rmdl", "ac", "fireplace", "log_price"]].copy()
pb_corr_inputs[["rmdl", "ac", "fireplace"]] = ( 
  (pb_corr_inputs[["rmdl", "ac", "fireplace"]] == "Yes").astype(int) 
  ) 
   
# Generate Pearson and point-biserial correlation coefficients
print("\nThe correlation between log_unit_gba and log_price:", 
      pearsonr(appraisals["log_unit_gba"], 
               appraisals["log_price"]).statistic.round(2), "\n")
for column in ["rmdl", "ac", "fireplace"]: 
    print(f"The correlation between {column} and log_price:", 
          pointbiserialr(pb_corr_inputs[column], 
                         pb_corr_inputs["log_price"]).statistic.round(2), "\n")

# Create a scatter plot to display the relationship between the variables
sns.lmplot(appraisals, x = "log_unit_gba", y = "log_price", 
           scatter_kws = {"color": "#0078ae"}, line_kws = {"color": "#000000"})
plt.title(label = "Log of Price vs. Log of Unit GBA", fontsize = 17)
plt.xlabel(xlabel = "Log of Unit GBA", fontsize = 14)
plt.ylabel(ylabel = "Log of Price", fontsize = 14)
sns.despine(left = True, bottom = True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
  lambda x, _: str(x).rstrip(".0")) 
  )
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter( 
  lambda x, _: str(x).rstrip(".0"))
  )
plt.show()

# Create box plots to display distributions
generate_box_plot("ward", "Ward")
generate_box_plot("heat_d", "Heating System")

# Perform one-way ANOVAs and Tukey HSD post-hoc tests
for column in ["ward", "heat_d"]:    
    print(f"\nOutputs for {column}:\n\n", 
          sm.stats.anova_lm(ols(f"log_price ~ {column}",    
                                data = appraisals).fit(), typ = 1), "\n\n", 
          pairwise_tukeyhsd(appraisals["log_price"], appraisals[column]), "\n")

# Part 4: Machine Learning Models

# Perform the train-test split for the model 
x1_train, x1_test, y1_train, y1_test = process_and_split("lr")

# Fit the model to the training data
lr_fit = LinearRegression().fit(x1_train, y1_train)

# Create a dataframe containing the performance and error metrics
model_metrics = pd.DataFrame({  
  "Model": "Linear Regression",
  "R\u00b2": format(lr_fit.score(x1_test, y1_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(    
    np.exp(y1_test), 
    np.exp(lr_fit.predict(x1_test)) *  
    np.mean(np.exp(y1_train - lr_fit.predict(x1_train))) 
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error(  
    np.exp(y1_test),
    np.exp(lr_fit.predict(x1_test)) *  
    np.mean(np.exp(y1_train - lr_fit.predict(x1_train)))  
    ), ",.0f"),  
  }, index = [0]) 

# Perform the train-test split for the model, which will be reusable for others
x2_train, x2_test, y2_train, y2_test = process_and_split("knn")

# Tune hyperparameters with cross-validation to find the best hyperparameters
knn_best_hp = GridSearchCV(
  estimator = KNeighborsRegressor(),
  param_grid = {"n_neighbors": [3, 4, 5], 
                "weights": ["distance", "uniform"]}, 
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False) 
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
knn_fit = KNeighborsRegressor( 
  n_neighbors = knn_best_hp["n_neighbors"], 
  weights = knn_best_hp["weights"] 
  ).fit(x2_train, y2_train)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "k-Nearest Neighbors",
  "R\u00b2": format(knn_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(knn_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - knn_fit.predict(x2_train)))  
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error(  
    np.exp(y2_test),
    np.exp(knn_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - knn_fit.predict(x2_train)))  
    ), ",.0f"),    
  }, index = [0])], ignore_index = True) 

# Tune hyperparameters with cross-validation to find the best hyperparameters
svm_best_hp = GridSearchCV(   
  estimator = SVR(),
  param_grid = {"kernel": ["rbf"],
                "C": [4000, 5000], 
                "epsilon": [0.001, 0.1, 1]},
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
svm_fit = SVR(
  kernel = svm_best_hp["kernel"],
  C = svm_best_hp["C"],
  epsilon = svm_best_hp["epsilon"] 
  ).fit(x2_train, y2_train)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({  
  "Model": "Support Vector Machine",
  "R\u00b2": format(svm_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(  
    np.exp(y2_test), 
    np.exp(svm_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - svm_fit.predict(x2_train)))
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(svm_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - svm_fit.predict(x2_train)))  
    ), ",.0f"),  
  }, index = [0])], ignore_index = True)   

# Tune hyperparameters with cross-validation to find the best hyperparameters
rf_best_hp = GridSearchCV( 
  estimator = RandomForestRegressor(random_state = 123), 
  param_grid = {"n_estimators": [500],
                "max_depth": [20, 21, 22],
                "min_samples_split": [2, 3], 
                "min_samples_leaf": [1]},
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)  
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
rf_fit = RandomForestRegressor(  
  n_estimators = rf_best_hp["n_estimators"],
  max_depth = rf_best_hp["max_depth"],
  min_samples_split = rf_best_hp["min_samples_split"],
  min_samples_leaf = rf_best_hp["min_samples_leaf"], 
  random_state = 123
  ).fit(x2_train, y2_train)
  
# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "Random Forest",
  "R\u00b2": format(rf_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(rf_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - rf_fit.predict(x2_train)))  
    ), ",.0f"), 
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(rf_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - rf_fit.predict(x2_train)))  
    ), ",.0f"),     
  }, index = [0])], ignore_index = True)   

# Perform the train-test split for the model and then create the pool objects
x3_train, x3_test, y3_train, y3_test = process_and_split("cb")
train_pool = Pool( 
  data = x3_train, label = y3_train, 
  cat_features = x3_train.drop(columns = "log_unit_gba").columns.tolist() 
  )
test_pool = Pool( 
  data = x3_test, label = y3_test, 
  cat_features = x3_test.drop(columns = "log_unit_gba").columns.tolist() 
  )

# Tune hyperparameters with cross-validation to find the best hyperparameters
cb_best_hp = pd.DataFrame(
  [CatBoostRegressor(loss_function = "RMSE", silent = True, 
                     random_state = 123). 
   grid_search(param_grid = {"iterations": [500],
                             "learning_rate": [0.1, 0.2, 0.3],
                             "depth": [6, 7, 8], 
                             "l2_leaf_reg": [6, 7, 8]}, 
               X = train_pool,
               cv = 5,
               shuffle = False, 
               verbose = False)["params"]],  
  index = [0])

# Fit the model to the training data
cb_fit = CatBoostRegressor(
  iterations = cb_best_hp["iterations"].iloc[0],
  learning_rate = cb_best_hp["learning_rate"].iloc[0],
  depth = cb_best_hp["depth"].iloc[0],
  l2_leaf_reg = cb_best_hp["l2_leaf_reg"].iloc[0], 
  silent = True,
  random_state = 123 
  ).fit(train_pool)

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "CatBoost",
  "R\u00b2": format(cb_fit.score(test_pool), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error( 
    np.exp(y3_test),
    np.exp(cb_fit.predict(test_pool)) *
    np.mean(np.exp(y3_train - cb_fit.predict(train_pool))) 
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y3_test),
    np.exp(cb_fit.predict(test_pool)) *
    np.mean(np.exp(y3_train - cb_fit.predict(train_pool))) 
    ), ",.0f"),
  }, index = [0])], ignore_index = True)

# Tune hyperparameters with cross-validation to find the best hyperparameters
xgb_best_hp = GridSearchCV( 
  estimator = XGBRegressor(random_state = 123), 
  param_grid = {"n_estimators": [500],
                "max_depth": [10, 11, 12],
                "learning_rate": [0.01, 0.02, 0.03],
                "min_child_weight": [1, 2, 3], 
                "subsample": [1],
                "colsample_bytree": [0.6, 0.7, 0.8]}, 
  scoring = "neg_root_mean_squared_error",
  cv = KFold(n_splits = 5, shuffle = False)  
  ).fit(x2_train, y2_train).best_params_

# Fit the model to the training data
xgb_fit = XGBRegressor( 
  n_estimators = xgb_best_hp["n_estimators"],  
  max_depth = xgb_best_hp["max_depth"], 
  learning_rate = xgb_best_hp["learning_rate"], 
  min_child_weight = xgb_best_hp["min_child_weight"],
  subsample = xgb_best_hp["subsample"],
  colsample_bytree = xgb_best_hp["colsample_bytree"],
  random_state = 123
  ).fit(x2_train, y2_train) 

# Append performance and error metrics to an existing dataframe
model_metrics = pd.concat([model_metrics, pd.DataFrame({ 
  "Model": "XGBoost",
  "R\u00b2": format(xgb_fit.score(x2_test, y2_test), ".3f"),
  "RMSE": "$" + format(root_mean_squared_error(   
    np.exp(y2_test), 
    np.exp(xgb_fit.predict(x2_test)) * 
    np.mean(np.exp(y2_train - xgb_fit.predict(x2_train)))  
    ), ",.0f"),
  "MAE": "$" + format(mean_absolute_error( 
    np.exp(y2_test),
    np.exp(xgb_fit.predict(x2_test)) *  
    np.mean(np.exp(y2_train - xgb_fit.predict(x2_train)))  
    ), ",.0f"),   
  }, index = [0])], ignore_index = True)

# Save the trained model that uses an XGBoost algorithm in a pickle file
joblib.dump(xgb_fit, "DC-Condominium-Appraisals-Model.pkl")
