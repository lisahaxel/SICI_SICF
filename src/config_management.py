"""
Configuration management for EEG classification analysis.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PathsConfig:
    """File paths configuration."""
    features_path: str
    model_savepath: str
    permutation_test_savepath: str


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation configuration."""
    n_repetitions: int
    k_out: int
    k_in: int
    random_state_multiplier: int


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Feature selection configuration."""
    max_mRMR_features: int
    min_features_to_select: int


@dataclass(frozen=True)
class OptimizationConfig:
    """Hyperparameter optimization configuration."""
    method: str
    n_trials: int
    early_stopping: bool
    startup_trials: int


@dataclass(frozen=True)
class SVMConfig:
    """SVM hyperparameter ranges."""
    C_range: Tuple[float, float]
    kernels: List[str]


@dataclass(frozen=True)
class LogRegConfig:
    """Logistic Regression hyperparameter ranges."""
    C_range: Tuple[float, float]
    penalties: List[str]
    solvers: List[str]


@dataclass(frozen=True)
class RFConfig:
    """Random Forest hyperparameter ranges."""
    n_estimators_range: Tuple[int, int]
    max_depth_range: Tuple[int, int]
    min_samples_split_range: Tuple[int, int]
    min_samples_leaf_range: Tuple[int, int]


@dataclass(frozen=True)
class BayesianRangesConfig:
    """Bayesian optimization ranges for all models."""
    SVM: SVMConfig
    LogReg: LogRegConfig
    RF: RFConfig


@dataclass(frozen=True)
class PermutationTestConfig:
    """Permutation testing configuration."""
    num_permutations: int


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    paths: PathsConfig
    feature_names: List[str]
    cross_validation: CVConfig
    feature_selection: FeatureSelectionConfig
    optimization: OptimizationConfig
    bayesian_ranges: BayesianRangesConfig
    classifiers: List[str]
    permutation_test: PermutationTestConfig


def get_config() -> AppConfig:
    """Create and return the centralized application configuration."""
    config = AppConfig(
        paths=PathsConfig(
            features_path="/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/features_extracted",
            model_savepath="./results/models",
            permutation_test_savepath="./results/permutation_tests"
        ),
        feature_names=["psd", "wpli", "pac", "phase"],
        cross_validation=CVConfig(
            n_repetitions=1,
            k_out=3,
            k_in=3,
            random_state_multiplier=100
        ),
        feature_selection=FeatureSelectionConfig(
            max_mRMR_features=200,
            min_features_to_select=2
        ),
        optimization=OptimizationConfig(
            method="bayesian",
            n_trials=40,
            early_stopping=True,
            startup_trials=8
        ),
        bayesian_ranges=BayesianRangesConfig(
            SVM=SVMConfig(
                C_range=(1e-3, 1e2), 
                kernels=["linear", "rbf"]
            ),
            LogReg=LogRegConfig(
                C_range=(1e-3, 1e2), 
                penalties=["l1", "l2"], 
                solvers=["liblinear"]
            ),
            RF=RFConfig(
                n_estimators_range=(30, 100),
                max_depth_range=(3, 6),
                min_samples_split_range=(2, 5),
                min_samples_leaf_range=(1, 3)
            )
        ),
        classifiers=["SVM", "LogReg", "RF"],
        permutation_test=PermutationTestConfig(num_permutations=500)
    )

    # Create output directories
    os.makedirs(config.paths.model_savepath, exist_ok=True)
    os.makedirs(config.paths.permutation_test_savepath, exist_ok=True)

    return config