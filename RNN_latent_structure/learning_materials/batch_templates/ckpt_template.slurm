#!/bin/bash

## Job Name

#SBATCH --job-name=test_job

## Allocation Definition

## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).

#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt

## Resources

## Nodes

#SBATCH --nodes=1   

## Walltime (hrs:mins:seconds). Do not specify a walltime substantially more than your job needs.

#SBATCH --time=2:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.

## For mox, --mem may be more than 100G depending on the memory of your nodes.

## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=100G

## Specify the working directory for this job
#SBATCH --workdir=/gscratch/stf/zmcnulty/myjobdir

## Specify the directory where STDOUT and STDERR files are stored
#SBATCH --output=/gscratch/stf/zmcnulty/output_files/

##turn on e-mail notification

#SBATCH --mail-type=ALL

#SBATCH --mail-user=zmcnulty@uw.edu

## export all your environment variables to the batch job session

#SBATCH --export=all


## ======== PUT COMMANDS HERE ==========================================================

## LOAD MODULES
