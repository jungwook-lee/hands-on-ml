# Chapter 1. The Machine Learning Landscape

## Supervised Learning Algorithms
- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVM's)
- Decision Trees and Random Forests
- Neural Networks


- **Supervised Neural Networks**
  - autoencoders
  - restricted Boltzmann Machines
- **Semisupervised Neural Networks**
  - deep belief networks
  - unsupervised pretraining

**logistic regression**: regresses probability values of a given class for classification

## Un-supervised Learning Algorithms
- **Clustering**
  - K-menas
  - Hierarchical Cluster Analysis (HCA)
  - Expectation Maximization
- **Visualization and dimensionality reduction**
  - PCA (principal component analysis)
  - Kernal PCA
  - Locally-linear embedding
  - t-distributed stochastic neighbor embedding (t-SNE)
- **Association rule Learning**
  - Apriori
  - Eclat

**Dimensionality reduction**: reduce the data, without too much loss of information
  - correlate features into one

**Feature Extraction**: merge/fuse features together that correlates well

*Why use dim. reduction?*: helps the ML methods to run faster, take up less
space and memory space.

**Anomaly Detection**: detecting unusal events, removing outliers

**Association Rule Learning**: goal is to dig into large amouts of data to discover interesting relations
  between attributes

## Semi-supervised Learning
- deal with partially labeled training set
- usually alot of unlabeled data, little bit of labeled data
- example: `Google Photos`
  - detects faces
  - clusters similar faces
  - label one, and the system can cluster for a single person
- most are combination of unsupervised and Supervised
  - DBN (deep belief network) -> supervised (RMB's)

## Reinforcement Learning
- Learning system: Agent
- Agents can:
  - observer environment
  - select, perform action
  - get rewards/penalties in return
- It learns the best strategy, called a `policy` to maximize maximize reward over time
- `Deepmind`'s `AlpahGo`
  - Learned how to play by analyzing millions of games
  - Then playing by itself alot

## Batch and Online Learning
- Can the system learn incrementally from a *stream of data*?

**Batch Learning**: Can't be trained incrementally, trained on all the data
- Takes alot of data/computing resources
- Typically done as `offline learning`
- Then launched to production, and runs without learning
- To use new data, the model needs to be stopped to be trained
- Training with full set of data can take many hours
- Therefore, update only every 24 hours or even just weekly
- Not ideal for rapidly changing data (e.g. stock prices)
- Cost lots of resource (CPU, memory space, disk space, I/O)

**Online Learning**: Train the system incrementally by feeding it data instances sequentially, in groups called *mini-batches*.
- On-the-fly learning
- Good for rapid changing dataset, limited computing resources
- Also called `out-of-core` learning (if outside main memory)

`Learning Rate`: key hyperparameter that affects how fast it should adapt to chaning data.
- High learning rate -> model rapidly adapts to new data, forget old data
- Low learning rate -> system will have more inertia

**Note on Online Learning**: Potential problem lies if your model is fed bad data. The model performance will decline rapidely. Therefore, it is important to monitor closely and switch learning off if necessary. Monitoring input data for abnormal data is also recommened.

## Instance-Based Versus Model-Based Learning
- How do the ML system generalize?

**Instance-based learning**: The system *memorizes* the data and uses *measures of similarities* to generalize to new cases.

**Model-based learning**: Build a model of the examples, use the model to make predictions. (i.e. learn the boundaries)

**Typical Machine Learning Project**:
- Study the data
- Select a model
- Train it on a training data
  - The model is trained to find the best parameter that minimize a cost function
- Apply the prediction on new cases (**inference**), hope that it generalize well

## Main Challenges of Machine Learning
- 2 ways things can go wrong: `bad data` or `bad algorithm`

**Insufficient Quantity of Training Data**:
- ML algorithms need `lots of data`
- `Banko and Brill [2001]` shows that even simple algorithms do well given enough data
- `Norvig et al. [2009]` "The Unreasonable Effectiveness of Data" shows that data matters more than algorithms for complex problems.

**Nonrepresentative Training Data**:
- Crucial that your data is `representative` of the new cases to be generalized
- `non-representative` data can affect the model's accuracy
  - if the sample is too small, `sampling noise`
  - If the sampling method is flawed, `sampling bias`

**Poor-Quality Data**:
- Full of errors, outliers, noise (from measurement)
- Hard for models to learn, thus `data cleaning`
  - discard, fix the erroneous data (outliers)
  - missing data, features, ignore values

**Irrelevant Features**:
- Training data should have most relevant features.
- `Feature Engineering`:
  - `Feature Selection`, selecting/unselecting most helpful features
  - `Feature Extraction`, combining existing feature to produce more useful ones (think dim. reduction)
  - Creating new features

**Overfitting the Training Data**:
- `Overgeneralizing` is bad, so is in Machine Learning.
- Performs well on training data, but bad on test data
- Models with high capacity for learning tends to ovefit
- Stick too close to the noise
- Solutions are :
  - pick a simpler model with fewer parameters
  - get more data
  - reduce noise in the training data
- `regularization` (contraining a model to reduce overfitting
- `hyperparameter`: parameters of learning algorithms (not the model)
- if a model is heavily regularized, it might not learn fast enough

**Underfitting the Training Data**:
- `Underfitting` is when your model is too simple to learn the structure of data
- Solutions are:
  - pick a more complex model with more parameters
  - feed better relevant features
  - reduce the constraints (i.e. tweak hyperparameters)

## Testing and Validating
- How do know if a model is generalizing well to new data?
- How do we test before serving it live?
- Split data in to 2: `training set` and `test set`
- ration of `80:20`
- error rate on new cases i called `generalization error`
- low training error and high generalization error: overfitting
- measuring performance from `test set` can lead to overfitting to itself
- use `validation set` to measure performance after training
- to avoid wasting too much data for validation/testing, use `cross-validation`
  - split into complementary subsets
  - each model is trained against different combinations
  - validated with remaining parts
  - best one is picked and measured on test set

**No Free Lunch Theorem**: `Wolpert [1996]` demonstrated that there is no model that is *a priori* guaranteed to work better. Only way to know is to evaluate them all. Therefore, you make an assumption about the data and evaluate a few models.
