import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


def task1_sklearn_linear_regression(X, y):
    """
    Given the asked for features X and the target regression
    values y, fit a sklearn linear regression model to the data
    and return the fitted model.

    Parameters
    ----------
    X - The expected set of features to train with in order to get the expected
        fitted regression model for Task 1
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1

    Returns
    -------
    model, intercept, slope, mse, rmse, rsquared - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slope and
        intercept, mse, rmse and r2 score

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slope, mse, rmse, rsquared = task1_sklearn(X, y)
    >>> isclose(intercept, 0.37578175021210747)
    True
    >>> isclose(slope, 0.3354845860060065)
    True
    >>> isclose(mse, 3.5473465427798607)
    True
    >>> isclose(rmse, 1.8834400820784984)
    True
    >>> isclose(rsquared, 0.5008050204985712)
    True
    """
    # Create the linear regression model
    model = LinearRegression()

    # Fit the model to the provided data
    model.fit(X, y)

    # Extract model parameters
    intercept = model.intercept_  # Intercept of the regression line
    slope = model.coef_[0]  # Slope (coefficient) of the regression line

    # Make predictions using the trained model
    y_pred = model.predict(X)

    # Compute evaluation metrics
    mse = mean_squared_error(y, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    rsquared = model.score(X, y)  # R² score

    return model, intercept, slope, mse, rmse, rsquared


def task1_statsmodel_linear_regression(y, X):
    """Given the asked for features X (with a dummy intercept constant
    already added), and the target regression values y, fit a
    statsmodel OLS (ordinary least squares) regression model to the
    data and return the fitted model along with important fit parameters.

    Parameters
    ----------
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1
    X - The expected set of features to train with, with an already added
        dummy intercept constant, in order to get the expected
        fitted regression model for Task 1

    Returns
    -------
    model, intercept, slope, mse, rmse, r2score - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slope and
        intercept, mse, rmse and r2 score

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slope, mse, rmse, r2score = task1_statsmodel(y, X)

    # the params of a statsmodel OLS model contains the [intercept, slope1, slope2...]
    >>> isclose(intercept, 0.37578175021210747)
    True
    >>> isclose(slope, 0.3354845860060065)
    True
    >>> isclose(mse, 3.5473465427798607)
    True
    >>> isclose(rmse, 1.8834400820784984)
    True
    >>> isclose(rsquared, 0.5008050204985712)
    True

    """
    # Add an intercept column to X
    X_with_intercept = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X_with_intercept).fit()                                       

    # Extract intercept and slope
    # intercept = model.params[0]
    intercept = model.params[0]
    slope = model.params[1]                      
                                                                                    
    # Calculate predictions
    y_pred = model.predict(X_with_intercept)

    # Calculate MSE and RMSE
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Get R-squared value
    rsquared = model.rsquared

    return model, intercept, slope, mse, rmse, rsquared

def task2_label_encoding(df_rain_tomorrow):
    """
    At the start of task 2, you first need to encode the RainTommorow as
    a categorical variable, and correctly map No to the 0 encoding and 
    Yes to the 1 encoding.  The encoded labels are passed in and tested by
    this function.  The encoded binary targets `y` are returned from calling
    this function.

    Parameters
    ----------
    df_rain_tomorrow - The dataframe/series of the RainTomorrow column that you
       need to encode as binary categorical labels.

    Returns
    -------
    y, ndim, shape, num_no, num_yes - The array should be 1 dimensional with 366 values, and
       if encoded correctly, there are 300 No and 66 Yes.  These are extractred and
       returned so we can test.

    Tests
    -----
    # these tests assume y is already defined in envrionment where
    # the doctests are called, and that it contains the correctly
    # encoded categorical labels for Task 2
    >>> y, ndim, shape, num_no, num_yes = task2_label_tests(df.RainTomorrow)
    >>> ndim
    1
    >>> shape
    (366,)
    >>> num_no
    300
    >>> num_yes
    66
    """
    # implement your label encoding for the binary classification labels here, make sure you
    # replace the return statement and return the asked for labels and test values

    # Ensure the input is a DataFrame (convert if it's a Series)
    if isinstance(df_rain_tomorrow, pd.Series):
        df_rain_tomorrow = df_rain_tomorrow.to_frame()

    # Convert column to string type (to avoid dtype issues)
    df_rain_tomorrow['RainTomorrow'] = df_rain_tomorrow['RainTomorrow'].astype(str)

    # Clean the 'RainTomorrow' column: Map '1' to 'Yes' and '0' to 'No'
    df_rain_tomorrow['RainTomorrow'] = df_rain_tomorrow['RainTomorrow'].replace({
        '1': 'Yes',
        '0': 'No'
    })

    # Validate that the column contains only 'No' and 'Yes'
    unique_values = df_rain_tomorrow['RainTomorrow'].unique()
    if not set(unique_values).issubset({'No', 'Yes'}):
        raise ValueError(f"Unexpected values in 'RainTomorrow': {unique_values}")

    # Create OrdinalEncoder instance with explicit category order
    ordinal_encoder = OrdinalEncoder(categories=[['No', 'Yes']])

    # Fit and transform the 'RainTomorrow' column
    y = ordinal_encoder.fit_transform(df_rain_tomorrow[['RainTomorrow']]).astype(int).flatten()

    # Compute required statistics
    ndim = y.ndim
    shape = y.shape
    num_no = np.sum(y == 0)  # Count of 'No'
    num_yes = np.sum(y == 1)  # Count of 'Yes'

    return y, ndim, shape, num_no, num_yes
    
def task2_impute_missing_data(df):
    """
    Next in task 2, you need to extract and encode the Sunshine and Pressure3pm features,
    and perform some data cleaning to impute some missing features.  The whole
    dataframe is passed in as input to this function.  You need to extract the needed
    features, and build a imputer to fill in the missing data.  You are also expected
    to extract some information after creating the cleaned features dataframe to be
    returned for testing.

    Parameters
    ----------
    df - The whole original dataframe, from which you will extract and clearn the needed features.

    Returns
    -------
    X, ndim, shape, columns, na_sum - The feature dataframe X should be 2 dimensional
       with 366 values and 2 features.  There should not be any missing values, and the returned 
       ndim, shape, columns, and na_sum will be used to test that data has been cleaned and
       no missing data remains.

    Tests
    -----
    # these tests assume y is already defined in envrionment where
    # the doctests are called, and that it contains the correctly
    # encoded categorical labels for Task 2
    >>> X, ndim, shape, columns, na_sum = task2_impute_missing_data(df)
    >>> ndim
    2
    >>> shape
    (366, 2)
    >>> columns
    Index(['Sunshine', 'Pressure3pm'], dtype='object')
    >>> na_sum
    Sunshine       0
    Pressure3pm    0
    dtype: int64
    >>> description
             Sunshine  Pressure3pm
    count  366.000000   366.000000
    mean     7.909366  1016.810383
    std      3.467180     6.469422
    min      0.000000   996.800000
    25%      6.000000  1012.800000
    50%      8.600000  1017.400000
    75%     10.500000  1021.475000
    max     13.600000  1033.200000
    """
    # put your implementation of imputing missing data for the binary classification
    # task 2 here
    # put your implementation of imputing missing data for the binary classification
    """Imputes missing data for Sunshine and Pressure3pm."""

    required_cols = ['Sunshine', 'Pressure3pm']  # Define required_cols INSIDE the function (BEFORE the if)

    if not all(col in df.columns for col in required_cols):
        return None, 0, (0, 0), None, pd.Series()

    X = df[required_cols].copy()
    imputer = SimpleImputer(strategy='mean')
    X[:] = imputer.fit_transform(X)

    ndim = X.ndim
    shape = X.shape
    columns = X.columns
    na_sum = X.isna().sum()

    return X, ndim, shape, columns, na_sum

def task2_sklearn_logistic_regression(X, y):
    """
    Given the asked for features X and the target regression
    values y, fit a sklearn logistic regression model to perform
    a classification task.

    Parameters
    ----------
    X - The expected set of features to train with in order to get the expected
        fitted regression model for Task 1
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1

    Returns
    -------
    model, intercept, slopes, accuracy - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slopes and
        intercept, and accuracy on the training data

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slopes, accuracy = task2_sklearn_logistic_regression(X, y)
    >>> isclose(intercept[0], 186.590648)
    True
    >>> isclose(slopes[0][0], -0.320885)
    True
    >>> isclose(slopes[0][1], -0.183120)
    True
    >>> isclose(accuracy, 0.863388)
    True
    """
    # put your implementation of task2 logistic regression using scikit-learn here
    # make sure you return your model and the expected test values in the return statement
    # Create and fit the logistic regression model
    model = LogisticRegression(solver='lbfgs', C=500.0)
    model.fit(X, y)
    
    # Extract the intercept and slopes
    intercept = model.intercept_[0]
    slopes = model.coef_[0]
    
    # Calculate accuracy
    accuracy = accuracy_score(y, model.predict(X))
    
    return model, intercept, slopes, accuracy

def task2_statsmodel_logistic_regression(y, X):
    """Given the asked for features X (with a dummy intercept constant
    already added), and the target regression values y, fit a
    statsmodel Logit  (logitsic regression) regression model to the
    data and return the fitted model along with important fit parameters.

    Parameters
    ----------
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1
    X - The expected set of features to train with, with an already added
        dummy intercept constant, in order to get the expected
        fitted regression model for Task 1

    Returns
    -------
    model, intercept, slopes, accuracy - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the
        fitted model intercept and slopes and the final model accuracy

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slopes, accuracy = task2_statsmodel_logistic_regression(y, X)
    Optimization terminated successfully.
             Current function value: 0.324586
             Iterations 7

    # the params of a statsmodel OLS model contains the [intercept, slope1, slope2...]
    >>> isclose(intercept['const'], 186.59040174670466)
    True
    >>> isclose(slopes['Sunshine'], -0.3208828310913976)
    True
    >>> isclose(slopes['Pressure3pm'], -0.18311988277396155)
    True
    >>> isclose(accuracy, 0.8633879781420765)
    True

    """
    # Add a constant (dummy intercept) term to X for statsmodels
    X_with_intercept = sm.add_constant(X)

    # Fit a logistic regression model using statsmodels
    model = sm.Logit(y, X_with_intercept).fit(disp=False)  # Fit model, suppress output

    # Extract intercept and slope coefficients
    intercept = model.params['const']
    slopes = model.params.drop('const')

    # Predict probabilities and classify based on threshold 0.5
    y_pred_probs = model.predict(X_with_intercept)
    y_pred = (y_pred_probs >= 0.5).astype(int)  # Convert probabilities to class labels

    # Compute accuracy
    accuracy = accuracy_score(y, y_pred)

    return model, intercept, slopes, accuracy