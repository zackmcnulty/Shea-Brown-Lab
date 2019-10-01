#!/bin/bash

# I used this file to generate a bunch of testing data

# usage:
# ./generate_uniform.sh


folder="uniform_angle_random_points" # foldername in ./test_movies to save movies

num_movies=10 # number of movies to make for each angle

# make "num_movies" movies for each angle in [min_theta, max_theta] taking step sizes of "theta_step"
min_theta=1 # minimum angle to make movies for
max_theta=351 # max angle to make movies for
theta_step=10 



touch "./test_movies/$folder/labels.csv"

movie_num=1

# generate training movies
for i in $( seq $num_movies)
do

	for next_theta in $( seq $min_theta $theta_step $max_theta)
	do
		filename=`printf random%03d $movie_num`
		points=`python3 make_movie.py --name $filename --folder "./test_movies/$folder" --random_points --theta $next_theta`

		# add the given label to the labels file
		printf "$movie_num,$points\n" >> "./test_movies/$folder/labels.csv"

		movie_num=$(($movie_num + 1))

	done
done

#printf '%s\n' '$' 's/.$//' wq | ex './test_movies/uniform/labels.csv' 
