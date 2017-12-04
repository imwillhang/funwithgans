#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name  
#SBATCH --job-name=first_genome_run
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=logs/first_genome_run_%j.out
#################
# a file for errors from the job
#SBATCH --error=logs/first_genome_run_%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=23:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition 
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
#SBATCH -p akundaje --gres gpu:2
#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=4
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=16G
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=willhang@stanford.edu
#now run normal batch commands
# note the "CMD BATCH is an R specific command
# module load python
# You can use srun if your job is parallel
#srun R CMD BATCH  ./rtest.R
# or 
# srun R --no-save < rtest.R
# otherwise: 
#ml cudnn/6.0
python main.py

