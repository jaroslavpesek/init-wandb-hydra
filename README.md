Boilerplate code for setup ML project with Wandb and Hydra

## Run without sweeper
```bash
python example.py --config-name example-conf
```

## Initialize sweeper
```bash
wandb sweep --project example-for-boys conf/example-sweeper.yaml
```

# Run agent
```bash
wandb agent <SWEEP_ID>
```