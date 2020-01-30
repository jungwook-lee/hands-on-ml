# Chapter 2. End-to-End Machine Learning Project
- We are hired at a real estate company!
- Steps of an End-to-End Project:
  - Look at big picture
  - Get the data
  - Discover, visualize the data to gain insight
  - Prepare the data for ML algorithms
  - Select a Model and train
  - Fine-tune the model
  - Present your solution
  - Launch, monitor and maintain the system

## Working with Real Data
- For the chapter, use `California Housing Prices` dataset from StatLib.
- Based on data from `1990` California Census

## Looking at the Big Picture
- `Frame` the problem:
> What do we want to do with the model?
> What is the business objective?
> How does the company use and benefit from the model?
- Building a model is not the end goal.
- For the given chapter, the model's output (prediction of housing price) will be fed to another ML system.
- The system will determine whether it is worth investing in a given area or not.
- The decision affects `revenue`.


- `Pipelines`: Sequence of data proessing components that handle data manipultaions and transforms.
  - components run `asynchronously`
  - each piece takes data, processes it and spits it out as result
  - `self-contained`: the interface between components is data store
  - `data flow graph` helps explain the flow of the system
  - different teams can work on components
  - make the whole system `robust`
  - however, proper `monitoring` is required to make sure a single component is degrading


- What is the current situation?
  - manual expert estimation of housing prices are not efficient
  - human labor is expensive
  - error rate is high
  - uses complex rule that are hard to maintain
- Use ML to automate the process of estimation of prices


- `Frame` the problem again:
  - supervised? unsupervised? or reinforcement learning?
  - classification? regression? or something else?
  - batch learning? or online learning?
- `Univariate` vs. `Multivariate`: # of input features for prediction


- Select a `Performance Measure`:
  - Typical regression measure for regression problem is RMSE
  - `RMSE` measures the `Standard Deviation`
  - From the `68-95-99.7` rule, 68% of the data falls within the first sigma
- If there are many outliers, sometimes `MAE`, mean absolute error is better.
- Note that `RMSE` is *euclidean norm*, also known as `l2-norm`
- Note that `MAE` is *manhattan norm*, also known as `l1-norm`
- For the `l-norms` higher the norm index means giving more focus on larger values and neglect smaller ones
  - Therefore, `RMSE` is more sensitive to outliers than `MAE`
- When the errors are exponentially rare, use `RMSE`.


- Check your `assumptions`!
  - list and verify the assumptions with other teams who are going to use your outputs!

## Get the Data

- `jupyter` is handy because it allows:
  - Drawing Figures
  - Markdown with `Latex` support
  - Interactive python
- Good for `Scientific/Educational Purposes!`
- make sure to have `%matplotlib inline` for figures


- Use `pandas` for data manipulating purposes
  - `DataFrame.info()` shows features, total rows, types
  - `DataFrame.head()` shows the first top 5 rows
  - `DataFrame.decribe()` shows count, mean, std, max, percentiles
  - `DataFrame.hist()` to get histograms of features


- `Exploring` the data:
  - Check the `units` of the data
  - Check whether values are `capped`
    - you are introducing non-linearity
    - model might not predict any values outside the capped range
    - to fix, either get the `proper value` or `remove`
  - Check the `scales` of the data (feature scaling)
  - Check the `shape` of the distributions:
    - `tail heavy` distribution might need some transformation


- Create a `test set`
  - using the `test set` to measure generalization may lead to `data snooping` bias
  - when using `random`, set a `np.random.seed(42)` to ensure you have same results
  - for dealing with future proofing test sets, you can hash the instances
  - use stable features to make unique identifiers
  - `Random` sampling isn't always best to get representative `train/test` split
  - sometimes `stratified` sampling is better
  - `strata` are homogenous subgroups
  - if a certain `feature` is quite important, it is important that the `test` set contains enough

## Discover and Visualize the Data to Gain Insight
- Use a scatterplot (since the data is based on 2d locations)
- overlay features such as populations as circles with radius
- color values based on output features (i.e. price)


- `correlations`, compute standard correlation coefficient (Pearson's r) between every pair of attributes
  - note that `correlation` is a `linear` relationship
  - ranges from -1 to 1
  - 1 or -1 is strong relation
  - strong correlation means similar linear growth with variable increase
  - pandas allow `scatter_matrix`
- Binning to categories sometimes cause some quicks in the data
- Fuse features to bring new combinations `feature engineering`
  - Some fused features might correlate better!
  - `Experimentation` with features is key.

## Prepare the Data for Machine Learning Algorithms
- Why should we write functions to prepare the data?
  - reproduce the transformation on any dataset
  - build a library of transformation functions for the future
  - use them on live system to transform new data
  - makes it easy to try various transformation


#### Data Cleaning
  - Dealing with `Missing` Features
  - You can:
    - get rid of data with missing features `DataFrame.dropna()`
    - get rid of the feature itself `DataFrame.drop()`
    - fill missing values `DataFrame.fillna(value)`
  - Often `median` value is used to fill
  - you can use `sklearn.impute.SimpleImputer`
  - you can output calculated values using `SimpleImputer.statistcs_`


#### Scikit-Learn's API Design
  - `Estimators`: estimate some kind of parameters using `fit()` with `hyperparameters`
  - `Transformers`: transform a dataset, using `transform()`, also has a handy `fit_transform()` to do both at same time
  - `Predictors`: some estimators can output predictions, such as linear regression model. usese `predict()` and `score()` to measure of the quality of prediction
  - `Inspection`: all of estimators's hyperparameters are directly accessible by public instance variables
  - `Nonproliferation of clsses`: datasets are `numpy` or `scipy` sparse matrices, hyperparameters are python strings or numbers
  - `Composition`: existing functions are used as much as possible, thus `Pipeline` is easy to make
  - `Sensible defaults`: provides good default values for most parameters


#### Handling Text and Categorical Attributes:
  - problem with mapping text categories that they won't represent the continuous relationship
  - for example, red=0, blue=1, green=2 doesn't make much sense on a number line
  - Thus use `sklearn.preprocessing.OneHotEncoder`
  - for text->int->one-hot, use `sklearn.preprocessing.LabelBinarizer`


#### Custom Transfomers
  - why write custom transformers?
    - sometimes you will need to write your own custom cleanup/combining attributes
    - want to make transformer to work seamlessly with scikit-learn functionalities (i.e. pipelines)
    - scikit relies on `duck-typing`
    - the more you automate, the more you can try different hyperparams and such ...

#### Feature Scaling
- Machine Learning algorithms don't behave well, if the values are all over the place! (i.e. scales)
- Two common methods `min-max scaling` and `standardization`
- min-max scaling is basically normalization (0 to 1)
- `standardization` is different
  - subtracts the mean value, `zero-mean`
  - divide by variance `unit-variance`
  - does not bound values to a specific range, problematic for `ReLu` units or neural networks
  - however, not affected by outliers that much
- use `sklearn.preprocessing.StandardScaler` for standardization
