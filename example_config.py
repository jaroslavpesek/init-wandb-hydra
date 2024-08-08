from dataclasses import dataclass

def flatten_dict(d, parent_key='', sep='.'):
    return {f'{parent_key}{sep}{k}' if parent_key else k: v
            for kk, vv in d.items()
            for k, v in (flatten_dict(vv, f'{parent_key}{sep}{kk}' if parent_key else kk, sep=sep) if isinstance(vv, dict) else {kk: vv}).items()}


@dataclass
class ExperimentConfig:
    project_name: str
    dataset: str
    batch_size: int
    learning_rate: float
    epochs: int
    layer1_size: int
    layer2_size: int