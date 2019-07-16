#!/bin/bash

# I used this file to generate a bunch of testing data

# usage:
# ./generate_uniform.sh


num_movies=10
# number of movies to make with angles uniformly between [start, stop]
folder="random" # foldername in ./test_movies to save movies



touch "./test_movies/$folder/labels.csv"

# generate training movies
for i in $( seq $num_movies)
do
       filename=`printf random%03d $i`
       points=`python3 make_movie.py --name $filename --folder "./test_movies/$folder" --random_points`

       # add the given label to the labels file
       printf "$i,$points\n" >> "./test_movies/$folder/labels.csv"
done

#printf '%s\n' '$' 's/.$//' wq | ex './test_movies/uniform/labels.csv' 
