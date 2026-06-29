"""Classical ML model implementations."""

from forge.training.classical.catboost_model import CatBoostModel
from forge.training.classical.distance_models import KNNModel, NaiveBayesModel, SVMModel
from forge.training.classical.lightgbm_model import LightGBMModel
from forge.training.classical.linear_models import ElasticNetModel, LassoModel, RidgeModel, SGDModel
from forge.training.classical.logistic_regression import LogisticRegressionModel
from forge.training.classical.random_forest import RandomForestModel
from forge.training.classical.tree_models import DecisionTreeModel, ExtraTreesModel
from forge.training.classical.xgboost_model import XGBoostModel

BASE_MODELS = [
    LogisticRegressionModel,
    RidgeModel,
    LassoModel,
    ElasticNetModel,
    SGDModel,
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    KNNModel,
    SVMModel,
    NaiveBayesModel,
]

FAST_MODELS = [
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
]

__all__ = [
    "BASE_MODELS",
    "FAST_MODELS",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
]
