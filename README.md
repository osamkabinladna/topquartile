![Cool logo](assets/img.png)

# How to optimize hyperparameters using wandb sweeps

1. cd to where u have the yaml file
2. run this command
```bash
wandb sweep <name of yaml file>
```
3. copy the wandb agent command output
4. cd to top level topquartile folder (the one that does not have __init__ files)
5. run the wandb agent command, for example
```bash
wandb agent repres/topquartile/qkr7gyoq
``` 
