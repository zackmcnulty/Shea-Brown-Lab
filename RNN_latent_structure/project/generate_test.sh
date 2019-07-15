#!/bin/bash

# I used this file to generate a bunch of testing data

# usage:
# ./generate_uniform.sh


num_movies=180
# number of movies to make with angles uniformly between [start, stop]
folder="random" # foldername in ./test_movies to save movies



diff=$(expr $stop_angle - $start_angle)
step_size=$(expr $diff / $num_movies)

touch "./test_movies/$folder/labels.csv"

# generate training movies
for i in $( seq $num_movies)
do
	angle=$(($start_angle + $(($i * $step_size))))
       filename=`printf uniform%03d $angle`
       python3 make_movie.py --name $filename --theta $angle --folder "./test_movies/$folder"

       # add the given label to the labels file
       printf $angle >> "./test_movies/$folder/labels.csv"
       printf ',' >> "./test_movies/$folder/labels.csv"
done

printf '%s\n' '$' 's/.$//' wq | ex './test_movies/uniform/labels.csv' 
