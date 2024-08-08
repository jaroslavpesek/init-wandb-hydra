from keras import layers, models
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os
from example_config import ExperimentConfig, flatten_dict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from wandb.integration.keras import WandbMetricsLogger
from wandb_reporter import wandb_report

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="conf")
def main(cfg : DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = ExperimentConfig(**cfg_dict)

    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    os.environ["WANDB_HTTP_TIMEOUT"] = "300"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    run = wandb.init(
        project=f"{config.project_name}",
        dir=os.path.expanduser("~"),
        config=flatten_dict(cfg_dict)
    )

    config = dict(cfg_dict)
    config.update(wandb.config)
    config = ExperimentConfig(**config)

    logging.info(f"Config: {config}")

    # You can access the config values like this:
    # timestamp = run.start_time
    # experiment_nickname = run.name

    if config.dataset == "iris":
        data = load_iris()
        X = data.data
        y = data.target.reshape(-1, 1)
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")
    
    # Definition of your experiment goes here:
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Build the neural network model
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(layers.Dense(config.layer1_size, activation='relu'))
    model.add(layers.Dense(config.layer2_size, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    wandb.log({"eval/loss": loss, "eval/accuracy": accuracy})

    wandb_report(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), data.target_names)

    # wandb.alert(title="Experiment completed", text="The experiment has completed successfully")

    run.finish()

if __name__ == "__main__":
    main()

