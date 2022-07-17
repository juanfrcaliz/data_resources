import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, recall_score, classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def pca_variance_analysis(data: pd.DataFrame, step_size=10):
    """
    Simple function to explore how much of the data variance is explained
    when grouped by principal components.
    The results will be shown in a line graph.
    :param data: pd.DataFrame with the data to analyze.
    :param step_size: int, default = 10. Number of variables reduced in each step.
    """
    variance_explained = []
    for n_features in tqdm.tqdm(range(len(data.columns), 0, -step_size)):
        pca = PCA(n_components=n_features)
        pca.fit_transform(data)
        var = pca.explained_variance_ratio_.sum()
        variance_explained.append({'n_features': n_features,
                                   'variance': var})

    pd.DataFrame(variance_explained).set_index('n_features').plot(figsize=(7, 5))
    plt.title('Variance explained by number of dimensions with PCA', fontsize=18)
    plt.show()


def calculateVIF(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the degree of multicollinearity existing among
    different variables in a dataset.
    As a reference, significance levels of multicollinearity are:
    - VIF < 5: Not significant
    - 5 < VIF < 10: Moderately significant
    - 10 < VIF: Highly significant

    Args:
    - data_: pandas DataFrame containing the variables.

    returns: pandas DataFrame containing the multicollinearity level for each
    variable.
    """
    features_ = list(data_.columns)
    num_features = len(features_)

    model = LinearRegression()

    result = pd.DataFrame(index=['VIF'], columns=features_)
    result = result.fillna(0)

    for ite in range(num_features):
        x_features = features_[:]
        y_feature = features_[ite]
        x_features.remove(y_feature)

        model.fit(data_[x_features], data_[y_feature])
        result[y_feature] = 1 / (1 - model.score(data_[x_features], data_[y_feature]))

    return result


def quick_score(x_, y_):
    """
    Quickly tests the score for a dataset using a simple
    Logistic Regression model.
    The function shows the results on screen.
    """
    x_train_, x_test_, y_train_, y_test_ = train_test_split(
        x_, y_, test_size=0.2
    )
    # Invoke classifier
    clf = LogisticRegression(n_jobs=-1)

    # Make a scoring callable from recall_score
    recall = make_scorer(recall_score)

    # Cross-validate on the train data
    train_cv = cross_val_score(X=x_train_, y=y_train_, estimator=clf, scoring=recall, cv=3, n_jobs=-1)
    print("TRAIN GROUP")
    print("\nCross-validation recall scores:", train_cv)
    print("Mean recall score:", train_cv.mean())

    # Now predict on the test group
    print("\nTEST GROUP")
    y_pred = clf.fit(x_train_, y_train_).predict(x_test_)
    print("\nRecall:", recall_score(y_test_, y_pred))

    # Classification report
    print('\nClassification report:\n')
    print(classification_report(y_test_, y_pred, zero_division=0))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.copper)
    plt.show()


def score_optimization(params_, clf_, train_test: list, performance: pd.DataFrame):
    """
    This function receives a list of parameters and a classifier object and optimizes
    a predictive model based on them.
    The model returns the best performing trained model.

    :arg: params_: list of dicts with parameters to try for the model.
    :arg: clf_: model object.
    :arg: train_test: list containing train and test data in order [x_train, y_train, x_test, y_test].
    :arg: performance: pd.DataFrame where the performance of the model is stored.
    """
    x_train, y_train, x_test, y_test = [i for i in train_test]
    precision = make_scorer(precision_score)

    # Load GridSearchCV
    search = GridSearchCV(
        estimator=clf_,
        param_grid=params_,
        n_jobs=-1,
        scoring=precision
    )

    # Train search object
    search.fit(x_train, y_train)

    # Heading
    print('\n', '-' * 40, '\n', clf_.__class__.__name__, '\n', '-' * 40)

    # Extract best estimator
    best = search.best_estimator_
    print('Best parameters: \n\n', search.best_params_, '\n')

    # Cross-validate on the train data
    print("TRAIN GROUP")
    train_cv_ = cross_val_score(X=x_train, y=y_train,
                                estimator=best, scoring=precision, cv=3)
    print("\nCross-validation precision scores:", train_cv_)
    print("Mean precision score:", train_cv_.mean())

    # Now predict on the test group
    print("\nTEST GROUP")
    y_pred_ = best.fit(x_train, y_train).predict(x_test)
    print("\nPrecision:", precision_score(y_test, y_pred_))

    # Get classification report
    print(classification_report(y_test, y_pred_))

    # Print confusion matrix
    conf_matrix_ = confusion_matrix(y_test, y_pred_)
    sns.heatmap(conf_matrix_, annot=True, fmt='d', cmap=plt.cm.copper)
    plt.show()

    # Store results
    performance.loc[clf_.__class__.__name__ + '_optimize', :] = [
        train_cv_.mean(),
        precision_score(y_test, y_pred_),
        conf_matrix_[0, 0] / conf_matrix_[0, :].sum(),
        # precision_optim(y_test,y_pred_)
    ]
    # Look at the parameters for the top best scores
    display(pd.DataFrame(search.cv_results_).iloc[:, 4:].sort_values(by='rank_test_score').head())
    display(performance)

    return best, performance
