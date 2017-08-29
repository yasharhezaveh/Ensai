========
Overview of the code
========

Here you will find a quick summary of the codes provided.

You can start your python environment and execute "init.py". This file sets up the necessary variables to perform various neural network tasks (e.g., training or testing). Once this code is executed, nothing happens. You can now run "train.py" to train a model, or run "single_model_predictions.py" to get the predictions of a particular neural net for the test/validation samples. 

If instead of "single_model_predictions.py", you run "combo_prediction.py", you will get the predictions of 4 different networks, combined together. This will require that you have the weight files properly stored in your data/trained_weights folder (see installation). 


The models (with the exception of Inception.v4) are defined in "ensai_model.py". It is quite easy to modify these or to add new models. 


We use "get_data.py" to produce simulated images. This file contains a number of functions (e.g., apply_psf, add_poisson_noise) which are applied to previously saved simulated images of gravitational lenses. the function "read_data_batch" reads the noise-free images from the disk, and applies these effects to them. 


