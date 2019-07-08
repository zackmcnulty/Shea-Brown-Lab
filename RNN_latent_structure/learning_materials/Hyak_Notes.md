# HYAK Notes

## What is Hyak?

Hyak is a supercomputer owned by UW. It consists of a collection of high-throughput nodes (each of which is roughly equivalent to a high-end server)
that are capable of parallel computation. Even if you do not plan to parallelize any work, the nodes are still incredibly fast to work on and have
top of the line equipment (large memory, fast CPUs, and some have GPUs as well). Nodes are bought by individual groups to use, but Hyak has a cool
feature called "Backfill" which allows users access to all available nodes (although this use is a bit restricted). See the section below on Backfill for more
information on it.

Lots of information on Hyak can be found on their [wiki](https://wiki.cac.washington.edu/display/hyakusers/WIKI+for+Hyak+users). For further questions,
the Research Computing Club runs a [Slack](https://uw-rcc.slack.com/messages) channel where smart and useful people reside. Also, there appears to be some sort
of question [forum](https://talk.uw-hp.cc/login) where many common questions are answered.

If you are a student at UW (graduate or undergraduate) you can apply for free access to the Student Technology Fund group. STF purchased several
nodes for use by the Research Computing Club at UW. To get access to these nodes, follow the instructions found [here](http://depts.washington.edu/uwrcc/getting-started-2/getting-started/).
This is what I have been using and it works pretty well. 

## Logging In and Navigating Hyak

Just like a remote server, Hyak is accessed through SSH (and it also uses 2-factor verification). When you SSH into Hyak with (either Mox or Ikt; depends on where your group's nodes are):

```python
ssh -X yourUWnetid@mox.hyak.uw.edu  # for Mox
ssh -X yourUWnetid@ikt.hyak.uw.edu  # for Ikt (older branch of Hyak)
```

you will find yourself in your users home directory. These directories have a small memory capacity (10GB) and are NOT meant to hold any significantly large files.
Only really store files here that are meant to be completely private like SSH keys or possibly logon scripts. For all data from experiments and what not, these files
should be stored in the gscratch directories (see GSCRATCH section below).  

Like most servers, Hyak runs on the Linux operating system. Thus, you will interact with it through the command line by running bash commands
(if you have a Mac, this is pretty much the same as the terminal on your own computer). For more information on the types of commands available
on the Linux OS, check out this [tutorial](https://www.tutorialspoint.com/unix/index.htm).


## Types of Nodes

- **Log-on Nodes**: When you initially SSH into hyak when you log on, this is the node you start with. It handles all the basic operations allowing you to navigate through Hyak,
submit jobs, and request access to other nodes. These do NOT have a lot of computing power, and should not be used for anything beyond these simple tasks

- **Build Nodes**: These are nodes that all users have access to. They are set aside to help install needed software packages and compile code. Because they are often quick to obtain, they are also helpful
for testing (in interactive mode; see below) whether your code runs on small samples and do some possible debugging. However, no large computations should be performed on these nodes. These nodes have access to the internet and thus are helpful for file transfers, using git, etc.
    - To get access to a build node, use the command: `srun -p build --nodes 1 --pty /bin/bash`

- **Compute Nodes**: This is what you are really paying for. These are the nodes with a ton of computational power that you want to be running your large jobs on. You only have unrestricted access to the nodes your group
buys directly. However, Hyak has a cool feature called Backfill that allows you to run on the unused nodes owned by other people. However, there are some restrictions that come with this (see Backfill section below). These
nodes do NOT have access to the internet, so nothing can be downloaded outside Hyak from here.



## GSCRATCH

While you have a home directory, the directory you start in when you log in, this location has a small memory capacity allocated to it. Beyond private information (i.e. SSH keys, logon scripts, etc)
most files should be stored instead in your gscratch folder. The location of these directories are:

```/gscratch/group_name/user_name```


For me under the STF group, its:

```/gscratch/stf/zmcnulty```

This is where all data/code files will be stored for anything you are trying to do on a compute node. Use the scp command (while on a build node) in bash to transfer files locally (or from the Lolo archives?) to this folder
before running jobs on some of the compute nodes. Think of this as your main workspace for anything you are trying to do on Hyak. Here is an example of how to transfer files with scp. This is run from my local machine (NOT logged onto hyak):

```
scp -r local_folder/ user_id@mox.hyak.uw.edu:path/to/folder/on/server
scp -r project/ zmcnulty@mox.hyak.uw.edu:/gscratch/stf/zmcnulty
```

The second command copies the entire project/ folder from my local machine (-r stands for recursive; copy folder and all its subfolders, etc) to my
scratch folder on Hyak (/gscratch/stf/zmcnulty)



### Loading Modules and Downloading Software

The modules are different types of software that are avaible on Hyak pre-downloaded (A full list is available [here](https://wiki.cac.washington.edu/display/hyakusers/Hyak+Software)). This software is pre-downloaded, but has to be loaded onto your node before it can be used. Below are some of the commands that can help.

- `module avail`  :  list all available modules for loading
- `module load <path/to/module>`  : load a module; give the path listed by the "module avail" command
- `module list`  : list of currently loaded modules (modules ready to be used in future commands)

Only modules that are currently loaded can be used. These are a fast way to get large, commonly used software packages. 
Many of the common coding languages (R, MATLAB, Python, Mathematica) are available here, although you may choose to download them 
instead from source. However, these are only the barebones of what you might need for a computational project. We cannot install packages to the
system python/R/MATLAB of course (we wont have the permission), but we can download them elsewhere. The way this is done can differ language to language.


The logon node you are currently on is pretty slow and does not have a lot of computational power. Its primarily meant for navigating
hyak. Instead, we will request interactive access (see the section below to see what this means) to a build node for downloading the needed
software. To get one of these, run:

```srun -p build --nodes 1 --pty /bin/bash```

Once the build node is acquired, there are two main options for installing Python packages.


#### Creating a conda environment (recommended for Python)

This is only useful if you are using Python exclusively. 
Conda environments are a lot like Python virtual environments (virtualenv). It is a feature of Anaconda python, so to use
it we must specifically use that python on Hyak. To do so, use:

```module load anaconda<python version>```

To see which versions of anaconda are available, use the module avail command (To filter out non-anaconda stuff use "module avail | grep anaconda").
Once we have anaconda, we can create a conda environment using:

conda create -n environment_name

This creates a folder `/usr/lusers/user_name/.conda/envs/environment_name` where all your software will be downloaded to. To download software (i.e. Python libraries) we can use the following
command:

```
conda install package_name  #(e.g. conda install tensorflow)
```

However, just like virtualenv you will have to turn on the environment before using any of this software. To do so, use the command:

```
source activate environment_name
```

Once the environment is activated, we can use this software like we downloaded it normally with pip (i.e in Python just `import package`).
You will probably want to use this on a compute node in a batch script rather than in this interactive setting, so be sure to activate the
conda environment in your batch script (an example of a batch script is below)!

If you are downloading a ton of software packages, consider putting the conda environment in your gscratch folder using:

```
conda create --prefix /gscratch/group_name/user_name/environment_name
source activate /gscratch/group_name/user_name/environment_name
```


#### Downloading to a separate file and Adjusting PYTHONPATH

The next option is to download the required software/modules to a folder in your scratch directory:

```
/gscratch/group_name/user_id   --> /gscratch/stf/zmcnulty (for me)
```

If you are not using Python, you can skip to the lower part of this section where I give more general instructions. Using pip, we can download python modules to locations other than the general site-packages folder. First, use the `module load` command
to get access to a pre-installed version of python on Hyak (intel-python3_2017 is recommended).

```
module load intel-python3_2017   
```

Then, we can use pip to download to a specific directory with the `--target` option. Here, we download them to a folder called "python_libraries":

```
pip install --ignore-installed --target=/gscratch/group_name/user_name/python_libraries/ package_name  
```
As an example, for my user on the STF account I use:
```
pip install --ignore-installed --target=/gscratch/stf/zmcnulty/python_libraries/ package_name  
```

To allow Python to find this folder and the modules in it, we must added the folder path to the PYTHONPATH environment variable:

```
export PYTHONPATH="${PYTHONPATH}:/gscratch/stf/zmcnulty/python_libraries"  
```

Once you download a library, you will not have to do it again. It stays in the scratch folder. However, every time you use a node and
load python, you will have to add the above path to PYTHONPATH so the new instance of python can find the library binaries. I recommend
adding the `export PYTHONPATH ...` line to your .bashrc file (in your home directory) so its executed every time you log on to Hyak.


#### Other Software

In general, you can download any software, not just python packages, to your gscratch file and compile them from there. Use curl or wget to download
files off remote servers (from website). The Hyak Wiki has a [tutorial](https://wiki.cac.washington.edu/display/hyakusers/Hyak+Compiling+HDF5+Serial) that
uses this procedure. To find the address you need to use with wget, you can sometimes go to the download page on the website, right click the download button, and
select "Copy Link Address". This will be the link you need (it often includes /ftp/ in the link). For example, I can download Python 3.7.3
from the Python website to my working directory using:

```
wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz
```

Unzipping this file (`tar -xzf Python-3.7.3.tgz`) you should have a folder. In this folder will be a README file typically with further instructions on how to compile the code.


#### Making a new Module

If you are downloading large and commonly used software, consider making a module so others can use it with the `module load <module_name>` command. To do
so, download and compile the required code to its own folder within the directory `/sw/contrib/`. Again, you will primarily use wget and curl to do the downloading
and follow specific instructions from the software for compiling/configuring the software. Once this is complete, you will have to build a corresponding modulefile
in the directory `/sw/modules-1.775/modulefiles/contrib/`. This tells the operating system where to find the required software by modifying relevant PATH variables.
See some of the other modulefiles in the directory and the [documentation](https://modules.readthedocs.io/en/latest/modulefile.html) for more advice on how to write these files.
Once this is complete, you (and all other users) can use the software anytime they want just with `module load /contrib/path/to/modulefile`.:wq




## Scheduling Jobs

Both Hyak Mox and Ikt use the [Slurm Scheduler](https://slurm.schedmd.com/overview.html) to schedule jobs across the many nodes in Hyak
in a non-conflicting way. Think back to your database management class and the idea of exclusive and inclusive locks.  

Slurm has some very useful [tutorials](https://slurm.schedmd.com/tutorials.html) which walk through how to use it although these cover far more than is
necessay to use Hyak. The few commands we discuss below will suffice for basic usage of Hyak. Since Hyak
runs off Slurm, this is essentially how jobs will be run on Hyak. Here are a few common commands and their options [here](https://slurm.schedmd.com/pdfs/summary.pdf). Again, this is far more details than the basic Hyak user needs to know. 
Before we discuss the commands, lets discuss two different types of Jobs.

#### Interactive Mode

Interactive mode allows you to run commands one at a time through the terminal. It will essentially be exactly like simply using a terminal, but all the operations you powerful nodes. This is helpful for trying to see if your code can run or has any bugs,
but should only be used for short and quick runs really as it requires you to be constantly connected to Hyak. If for some reason you lose connection (wifi drops, you close your laptop, etc) your commands will stop running: all the same issues of running stuff on your laptop.
I use it sometimes to check that my code runs and does not have any bugs. However, if you know your code works and just need to run through the simulation/data analysis, I would recommend using batch mode instead.
Interactive mode is also what you want to use when you are trying to download software as discussed in the section above. To do this, you will run interactive mode on a build node which we will discuss how to use below.

#### Batch Mode

Batch mode is what you will use for most heavy-duty computations. It allows you to specify all the commands you want to run in a file and then
submit the job to run. From there, you do not have to do anything: you can log off hyak, shut off your computer, and everything will still run. Batch mode specifies the operations using something called a
batch/slurm file. An example of this is given below.


#### Running The Jobs : Basic Commands

* `sbatch` : run a batch job; A batch job is a computer program or set of programs processed in batch mode. This means that a sequence of commands to be executed by the operating system is listed in a file (often called a batch file, command file, or shell script) and submitted for execution as a single unit. 
    * The file containing all the commands to run and specifying information about the number of nodes to use, memory allocation, etc is called the slurm script.
    * usage: `sbatch <batch file>`   (e.g. sbatch myscript.slurm)

* `srun` : run a job in interactive mode. Rather than specifying all commands in a slurm script for a batch job, you type the commands one by one in the console
    * usage: `srun -N num_nodes -A group_name -p partition_name --time=2:00:00 --mem=20G --pty /bin/bash`
    * Here, the group name is the group whose nodes you are running on (i.e. STF for the student tech fund) and the partition name is? (i.e. build, group_name, stf-int, ...)
    * Mostly, the "build" partition is used because it can connect to outside hyak. Thus, its useful for using git, transferring files to/from Hyak, and installing software packages (like a Python Library).
    * interactive mode is NOT advisable for large jobs.
	
Running the commands `squeue -u zmcnulty` or `squeue -p group_name` to see your/your groups respectively list of jobs. It will output lines of the form:

```
JOBID    PARTITION     NAME       USER      ST       TIME   NODES   NODELIST(REASON)
997755       stf      test_job   zmcnulty   PD       0:00     1       (Resources)
```

This gives the job ID, the group name, the job name, the user running the job, whether the job is running (R) or waiting (PD), number of nodes the job is running on, and the nodes the job is running on or a reason (REASON) the job is not running.
To cancel a job at any time, just do `scancel job_ID`

If you want information on your groups partition of Hyak (i.e. what nodes your group owns), use the `sinfo -p group_name` command.




## Backfill

The backfill is one of the coolest features of Hyak. Essentially, if someone is not using one of their nodes at any given time, you can run your code off it!
However, this comes with a catch. As soon as that person wants access to their node, you get kicked off. Some info can be found [here](https://wiki.cac.washington.edu/display/hyakusers/Mox_checkpoint) on the Hyak Wiki.
To specify you want your job to run in backfill, simply change the partition / account name in the batch file. Specifically, use the account `group_name-ckpt` and the partition `ckpt`.




## Example Batch Script

More information can be found [here](https://wiki.cac.washington.edu/display/hyakusers/Mox_scheduler) on how to choose these settings and what they all mean.
The Hyak wiki in general is a good source of information.


```python
#!/bin/bash

## Job Name
#SBATCH --job-name=zjm_rnn_analysis

## Allocation Definition

## group name and partition name of the nodes.
## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --account=stf
#SBATCH --partition=stf-gpu

##  Resources

## Number of Nodes to reserve. One will suffice unless you are doing some parallel computing
#SBATCH --nodes=1   

## Amount of time you need the node for (hrs:mins:seconds). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=2:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.
#SBATCH --mem=100G

## Specify the working directory for this job
#SBATCH --workdir=/gscratch/stf/zmcnulty/project

## Specify the directory where STDOUT and STDERR files are stored
#SBATCH --output=/gscratch/stf/zmcnulty/output_files/

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zmcnulty@uw.edu

## export all your environment variables to the batch job session
#SBATCH --export=all


## ======== PUT COMMANDS HERE ==========================================================

## LOAD MODULES
module load anaconda3_5.3


## My Program
conda activate /gscratch/stf/zmcnulty/env  # conda environment with all my needed software
python3 analysis.py -load models/rnn_predictior_BCE_dt_10_l1_0.0.h5 -movie_folder test_movies/uniform/ --positional_activity

```
