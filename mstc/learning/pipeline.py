"""Components for cross-validation and model evaluation."""
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline


def generate_cross_validation_pipeline(
    classifier, parameter_grid,
    folds=5, repeats=1, random_state=12345,
    number_of_jobs=1, scoring=None, refit=True
):
    """
    Evaluate a classifier trained with cross validation.

    Args:
        classifier (sklearn.base.ClassifierMixin): a classifier.
        parameter_grid (dict): grid of parameter.
        folds (int): number of stratified cross validation folds,
            defaults to 5.
        repeats (int): number of cross validation repeats,
            defaults to 1.
        random_state (int): random state, defaults to 12345.
        number_of_jobs (int): number of jobs to run in parallel, defaults to 1.
            -1 means using all processors.
        scoring (string, callable, list/tuple, dict or None): socring function
            or functions to evaluate predictions on the test set.
            Defaults to None to use the classifier default score method.
        refit (bool, string): whether to refit with best estimator. For
            multiple metric evaluation, this needs to be a string denoting the
            scorer is used to find the best parameters
            for refitting the estimator at the end.

    Returns:
        an evaluation report.
    """
    # ensure reproducibility in the classifier and log seed via parameter
    parameter_grid['random_state'] = [random_state]
    # generate the pipeline
    return make_pipeline(
        VarianceThreshold(),
        MinMaxScaler(),
        GridSearchCV(
            classifier,
            param_grid=parameter_grid,
            cv=RepeatedStratifiedKFold(
                n_splits=folds,
                n_repeats=repeats,
                random_state=random_state
            ),
            refit=refit,
            n_jobs=number_of_jobs,
            scoring=scoring,
            return_train_score=True
        )
    )
