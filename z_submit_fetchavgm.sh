#!/bin/bash

job_file="z_job.sh"

identifier_name="imagenet"

dir="rc_out/$identifier_name"
mkdir -p $dir

job_name="imagenet"

out_file=$dir/$job_name.out
error_file=$dir/$job_name.err

sbatch -J $job_name -o $out_file -e $error_file $job_file
