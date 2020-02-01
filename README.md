# ML_ROM_Closures
Time-series learning of latent-space dynamics for reduced-order model closure

R. Maulik, A. Mohan, B. Lusch, S. Madireddy, P. Balaprakash, D. Livescu

https://doi.org/10.1016/j.physd.2020.132368

## Runs
These are results using the hand-tuned hyperparameters

## HPS
These are results using deephyper for hyperparameter optimization

## Note
For visualizing results from LSTM deployments use `python Burgers_POD_ROM.py` and for NODE deployments use `python NODE_Burgers.py`. Change `deployment_mode = 'test'` to `deployment_mode = 'train'` within these files to retrain the framework. This repo will have further documentation added to it slowly - in the meanwhile please contact rmaulik@anl.gov for any questions.
