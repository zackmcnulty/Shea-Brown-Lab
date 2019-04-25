'''
make_movie.py

Given a series of objects that are influenced by some dynamical system, creates a movie (frame by frame) of those
objects moving (and possibly rotating) throughout that dynamical system.

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


# Define objects and their motion  ==============================================================================================

# each object will be specified by (motion = (dynamics, initial_condition), shape)
# Shapes will be specified as graphics objects

# NOTE: The depth of objects in the image is specified by the order they appear in "objects" below.
#       Shapes ealier in "objects" appear deeper in the image
objects = (ds.f_horz_spring(initial_condition=[0.5,0.5,1]),  ds.f_vert_spring(k=10,initial_condition=[0,0,10])) 

# NOTE: set all shapes to start at centered (0,0)
shapes =  [Circle(Point(0,0), radius=10), Rectangle(Point(-5,-5), Point(5,5))]
fill_colors = np.random.choice(['red', 'blue', 'green', 'black'], len(objects))
outline_colors = np.random.choice(['red', 'blue', 'green', 'black'], len(objects))

if len(objects) != len(shapes) or len(objects) != len(fill_colors) or len(objects) != len(outline_colors):
    raise ValueError("The number of objects, shapes, and colors specified must all match.")


(t_vals, x_vals, y_vals) = get_trajectory.get_trajectories(objects, num_steps=100)

# ===============================================================================================================================

# create temporary directories to save the frames/images that are generated
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
    img.save('tmp_images/frame{:05d}.gif'.format(frame), "gif")



win_height = 200
win_width = 100 # specify graphics window dimensions
padding = 50 # how much white space to add around the edge of the window to avoid objects running off edge
frame_rate = 20 # speed at which to play the frames

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
#    time.sleep(1/frame_rate)
    save_frame(win, frame + 1)


# Convert all these frames into a movie
movie_fps = 10;
movie_filename = 'movie_files/movie.mp4'
os.system("ffmpeg -r {:d} -i ./tmp_images/frame%05d.png -vcodec mpeg4 -y {:s}".format(movie_fps, movie_filename))

# delete the temporary folders created
os.system('rm -rf tmp_images')
os.system('rm -rf tmp_frames')


