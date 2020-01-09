# ML_ROM_Closures
Using sequential learning techniques for closing reduced-order models

## Runs
These are results using the hand-tuned hyperparameters

## HPS-Runs
These are results using deephyper for hyperparameter optimization

## Note
Results can be visualized using `deployment_mode = 'test'` in the run directories and retrained using `deployment_mode = 'train'`. For running LSTM deployments use `python Burgers_POD_ROM.py` and for NODE deployments use `python NODE_Burgers.py`. This repo will have further documentation added to it slowly - in the meanwhile please contact rmaulik@anl.gov for any questions.
