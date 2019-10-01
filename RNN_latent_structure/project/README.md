# Research Documentation


I have summarized most of my research goals, progress, and shortcomings in the linked [document](https://github.com/zackmcnulty/Shea-Brown-Lab/blob/master/RNN_latent_structure/project/research_goals_and_notes.pdf). Below,
I list some documentation for a variety of the code found in this repository. I highly suggest any training of the PredNet model
be on a remote server from the AMATH department or Hyak. The video clips and recurrent structure make these models
pretty slow to train, so I utilized the GPUs available on Hyak and tensorflow-gpu. If you are new to Hyak, I made
some [notes](../learning_materials/Hyak_Notes.md) documenting my experience. On Hyak, I have a miniconda environment
set up that you can activate and you should be able to run all this code. Just run:

```
module load contrib/keras_tensorflow-gpu_opencv_miniconda
source activate
```


## Main Python Files

### make_movie.py

This was the file I used to generate short .mp4 video files containing my moving blocks. It uses the well-known
basic python graphics library, [graphics.py](https://github.com/jminz/graphics.py). Each video contains some basic shape (or several)
like a block, triangle, or circle which moves around throughout a dynamical system. You must specify the properties
of the shape (see the graphics.py documentation for some examples) as well as a set of differential equations that
specify the dyanmics (the first two variables of the dynamical system should be x and y coordinate). Some examples of
these like the spring-mass system I used can be found in [dynamical_systems.py](./dynamical_systems.py](./dynamical_systems.py). Many of the arguments
you can pass to make_movie.py are just to help expedite the process of making a bunch of training movies. Most are
pretty specific to the movies I were making (i.e. spring-mass systems), but others are more general. Their uses can be seen by running ```python3 make_movie.py -h```
to bring up the help menu.

When run, the program saves the created movie file to the provided folder (default of ```./movie_files```). Here
is an example function call:

``` python3 make_movie.py --random_points --name training_movie  ```


### dynamical_systems.py

This file just stores some example dynamical systems (their system's of equations) and is used by the make_movie.py program for convenience.

### get_trajectory.py

This simply numerically integrates the given dynamical systems. It is used by make_movie.py to calculate the position of each object over time.
It has certain settings to help choose how to normalize the motion which are described further in the function itself.


### graphics.py

A basic graphics library for python. Documentation and usage can be found [here](https://github.com/jminz/graphics.py)

### generate_test/training.sh

These are convenience shell scripts that call the make_movie.py program many times to generate a large number of training/testing
datasets. They are tailored to generate movies for my spring-mass system.

### rnn_predictor_prednet.py

Trains a prednet model to do t+1 sequence-to-sequence prediction on the provided training/testing movie clips. At the end, outputs a model summary
and a video of the models prediction on the testing and training datasets. For more information on the parameters this
program accepts, run ```python3 rnn_predictor_prednet.py -h``` for the help log. This function can also be used to train
pre-trained models even further and this makes it suitable for checkpointing on Hyak backfill. An example method call would be:

```python3 rnn_predictor_prednet.py --epochs 100 --batch_size 10 --name prednet --show_movie ```

### analysis_prednet.py

This program is for analyzing a trained prednet model. Provided some testing data, it runs principal component analysis
on the recurrent units in prednet (layer specified in the program itself). It will plot the neural representation
in principal component space for each of the given test movies and color the points based on the relevant latent
parameters. This file is fine-tuned to analyze the latent parameters for my specific movies (spring-mass systems) but
it should be easy to adapt for any setting. Again, the function has a help log if you
want to see what each of its parameters does. Simply run ```python3 analysis_prednet.py -h```.
An example call to the program would be

```python3 analysis_prednet.py --load my_prednet.h5 --folder ./test_movies --make_figs```


### rnn_predictor_prednet_multistep

A bit of a work in progress. Takes a PredNet modeled trained for t+1 sequence to sequence prediction and 
fine-tunes the network for multistep (t + dt) prediction.

### models

Default location where trained models are saved.

### movie_files

Default location where movies from make_movies.py are saved.

### analysis_plots

Default location where plots from analysis_prednet.py are saved.

### test_movies

Stores a bunch of test movies from analysis_prednet.py. There is a labeling file, labels.csv, the describes
some of the details of each specific movie.


### old_files

A collection of old movie files from when I was doing a simpler system (block oscillating through origin of screen) and using
a simpler network (a convolutional autoencoder with an RNN stuck in the middle). I did not have much success with
these networks so I don't know if they have any uses.
