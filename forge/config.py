"""Global configuration for FORGE pipeline runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ForgeConfig:
    """Runtime configuration for a FORGE experiment."""

    target_column: str
    task_description: str = ""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    hpo_trials_per_model: int = 30
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "forge"
    enable_llm: bool = True
    enable_feature_selection: bool = True
    enable_shap: bool = True
    enable_lime: bool = True
    enable_pdp: bool = True
    enable_ensembles: bool = True
    enable_deep_learning: bool = True
    enable_error_analysis: bool = True
    enable_fairness: bool = True
    enable_llm_report: bool = True
    fast_mode: bool = False

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
