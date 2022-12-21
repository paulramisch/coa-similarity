# Convolutional Autoencoder: Code Structure

- `torch_data.py` contains dataset class and feature extraction to load data.
- `torch_create_embeddings.py` contains the embedding code.
- `torch_model.py` contains the model code.
- `torch_engine.py` contains training and validation steps. It also contains code to create embeddings.
- `utils.py` contains utility functions.
- `torch_train.py` contains training script. It makes use of `torch_data.py`, `torch_model.py`, `torch_engine.py` and `utils.py`.
- `torch_infer.py` contains code for inference.
- `automated_test.py` contains code for an automated test of the models performance.

# Usage
## Train
To train simply tweak the hyperparamets and set the data location in `config.py`, after that you just need to run `torch_train.py`, the model and the embeddings will be saved to the model folder set in the `config.py`.

## Create Embedding
To create a embedding of a whole folder, you can use `torch_create_embeddings.py`, the code doesn't use the values from the config-file but the model_name, model_path and img_path can be set right in the script.

The best performing model `_transformed20`is set as default.

## Inference/Usage
For inference / use of the model, you can load the package in python by
`from cnn_model.torch_infer import compute_similar_images, set_vars, plot_similar_cnn`

After that load the model with the following code (the model that is currently set in `config.py` will be loaded)
`encoder, img_dict, embedding, device = set_vars()`

To get a plot of similar coat of arms, use:
`plot_similar_cnn('path/to/image.jpg', embedding, encoder, device, img_dict)`

To get a list of similar coat of arms, use:
`compute_similar_images('path/to/image.jpg', 10, embedding, encoder, device, img_dict, 'image.jpg', angle_dict)`

## Server usage
To use the server just navigate to this directory in the terminal, then run:
`python python serve_waitress.py`

You can now either send a post request with an input file with the name "image" to `http://localhost:9001/similar_coa`
or you can just navigate to that url and use the form.

The server will return a list of similar files and an similarity score.

# Reference
The code is based on a model by Aditya Oke, that can be found here (Apache License 2.0s): 
https://github.com/oke-aditya/image_similarity
