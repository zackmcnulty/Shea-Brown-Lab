'''
make_movie.py

Given a series of objects that are influenced by some dynamical system, creates a movie (frame by frame) of those
objects moving (and possibly rotating) throughout that dynamical system.

Creates a movie file in the directory './movie_files/move_file_name.mpeg4' within the current working directory.

Usage:

    python3 make_movie.py movie_filename [, frames_per_second = 10]

'''

# NOTE: To support rotation, all objects will have to be regular polygons. We can then calculate rotation
#       manually and redraw the polygon
# see: https://stackoverflow.com/questions/45508202/how-to-rotate-a-polygon

from graphics import * # allows for drawing
import get_trajectory # calculates the trajectory of an object given a dynamical system
import dynamical_systems as ds # a few function definitions for common dynamical systems
import numpy as np
import time
from PIL import Image as NewImage
import os
import sys


# Define objects and their motion  ==============================================================================================

# each object will be specified by (motion = (dynamics, initial_condition), shape)
# Shapes will be specified as graphics objects

# NOTE: The depth of objects in the image is specified by the order they appear in "objects" below.
#       Shapes earlier in "objects" appear deeper in the image
objects = (ds.f_horz_spring(initial_condition=[0.5,0.5,1]), \
           ds.f_vert_spring(k=10,initial_condition=[0,0,10]), \
           ds.f_both_spring(initial_condition = [0,0,10,-10], k1=10, k2=1) ) 


# NOTE: set all shapes to start at centered (0,0)
shapes =  [Circle(Point(0,0), radius=10), Rectangle(Point(-5,-5), Point(5,5)), Circle(Point(0,0), radius=20)]

# Randomly choose colors for the objects.
fill_colors = np.random.choice(['red', 'blue', 'green', 'black'], len(objects))
outline_colors = np.random.choice(['red', 'blue', 'green', 'black'], len(objects))

if len(objects) != len(shapes) or len(objects) != len(fill_colors) or len(objects) != len(outline_colors):
    raise ValueError("The number of objects, shapes, and colors specified must all match.")



# Define Properties of Graphics window (Video Properties) ===================================================================
try:
    movie_filename = sys.argv[1]
    if not os.path.exists('./movie_files'):
       os.system('mkdir ./movie_files') 
    elif os.path.exists('./movie_files/' + movie_filename + ".mp4"):
        answer = input("Filename " + "./movie_files/" + movie_filename + ".mp4 already exists. Overwrite? [y/n] : ")
        if 'n' in answer.lower():
            raise ValueError()
except:
    raise ValueError("\n\tmake_movies.py expects an argument - the output movie file name. This name should not include spaces or be a pre-existing filename.")
     

# Number of frames to generate for the movie and the frames per second for the movie respectively.
num_frames = 100
movie_fps = 10

# Specify properties of the graphics window; Below allows you to control the size in pixels of
# the window (and hence the resolution in the video)
# The padding feature generates a an area of white space above/below and to the left/right of the window.
# I found this helpful for keeping the objects from running off the edge of the screen. As their motion
# moves around their center so if the object is too big it will go over the edge.
win_height = 200
win_width = 100 
padding = 50 




# ===============================================================================================================================

(t_vals, x_vals, y_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames)

# create temporary directories to save the frames/images that are generated
try:
    # clear folders if they are there from a previous stopped run
    os.system('rm -rf tmp_images')
    os.system('rm -rf tmp_frames')
except:
    pass

os.system("mkdir tmp_images/")
os.system("mkdir tmp_frames/")

def save_frame(win, frame):
    '''
    Saves as an image the current graphics window
    '''
    # saves the current TKinter object in postscript format
    win.postscript(file="tmp_frames/image.eps", colormode='color')

    # Convert from eps format to gif format using PIL
    img = NewImage.open("tmp_frames/image.eps")
    img.save('tmp_images/frame{:05d}.png'.format(frame), "png")




win = GraphWin( width = win_width + 2*padding, height=win_height + 2*padding)


# draw the initial state and color the objects 
for i, shape in enumerate(shapes):
    x = x_vals[i, 0]
    y = y_vals[i, 0]
    shape._move(padding + x * win_width, padding + y * win_height)
    shape.draw(win)

    shape.setFill(fill_colors[i])
    shape.setOutline(outline_colors[i])

    save_frame(win, 1)


for frame in range(1, len(t_vals)):
    for i, shape in enumerate(shapes):
        dx = win_width * (x_vals[i, frame] - x_vals[i, frame-1]) 
        dy = win_height * (y_vals[i, frame] - y_vals[i, frame - 1])
        shape._move(dx, dy)

    win.redraw()
    save_frame(win, frame + 1)


# Convert all these frames into a movie
movie_filename = './movie_files/' + movie_filename + ".mp4"
os.system("ffmpeg -r {:d} -i ./tmp_images/frame%05d.png -vcodec mpeg4 -y {:s}".format(movie_fps, movie_filename))

# delete the temporary folders created
os.system('rm -rf tmp_images')
os.system('rm -rf tmp_frames')


