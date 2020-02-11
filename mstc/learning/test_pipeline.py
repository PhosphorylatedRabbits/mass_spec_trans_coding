"""Tests learning pipeline."""
import unittest
import pandas as pd
from functools import reduce
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from .pipeline import generate_cross_validation_pipeline


class LearningPipelineTestCase(unittest.TestCase):
    """Test learning pipeline."""

    random_state = 12345
    test_size = .10
    classifier = LogisticRegression(solver='liblinear')
    parameter_grid = {'C': [0.1, 10.], 'penalty': ['l1', 'l2']}
    folds = 5
    repeats = 2
    scoring = {
        'Accuracy': 'accuracy',
        'F1': 'f1'
    }
    refit = 'Accuracy'

    def setUp(self):
        """
        Create a classification dataset and expected sizes for the report.
        """
        self.combinations = reduce(
            lambda a, b: a*b,
            (len(parameters) for _, parameters in self.parameter_grid.items())
        )
        self.report_columns = (
            4 +  # timings columns
            len(self.parameter_grid) +  # parameters from the grid
            1 +  # added random state
            1 +  # parameters dictionary
            len(self.scoring)*(
                self.folds*self.repeats + 3 +  # test splits and scores
                self.folds*self.repeats + 2  # train splits and scores
            )
        )
        (
            self.X, self.y
        ) = make_classification(random_state=self.random_state)

    def test_cross_validation_pipeline(self):
        """Test the cross validation pipeline."""
        pipeline = generate_cross_validation_pipeline(
            self.classifier, self.parameter_grid,
            folds=self.folds, repeats=self.repeats,
            random_state=self.random_state,
            scoring=self.scoring, refit=self.refit
        )
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        pipeline.fit(X_train, y_train)
        self.assertEqual(
            (self.combinations, self.report_columns),
            pd.DataFrame(pipeline.steps[2][1].cv_results_).shape
        )
        # in the following one could validate with the test data
