# ECE 284: Assignment 3 and 4

## Contents
* [Deadlines](#deadlines)
* [Overview](#overview)
* [Setting up](#setting-up)
* [Code development and testing](#code-development-and-testing)
* [Submission guidelines](#submission-guidelines)

## Deadlines
- Assignment 3: Monday, Feb 12 2024 (by 11:59pm PT)
- Assignment 4: Wednesday, Feb 21 2024 (by 11:59pm PT)

## Overview

Assignment 3 and 4 have the same codebase. The program (`readMapper`) first reads in an input reference sequence in the FASTA format using the `kseq` library (http://lh3lh3.users.sourceforge.net/kseq.shtml), compresses it using 2-bit encoding and build its corresponding seed table on a GPU using the same algorithm as Assignment 1 and 2. Next, it reads in a set of input read sequences of fixed size (256 characters), also in the FASTA format, and transfers their compressed representation in batches to the GPU device, computes the minimizer seeds, finds the corresponding seed hits and computes the mapping score (in terms of the number of matching characters) for the ungapped extension of sequences around the seed hits. The best mapping score and their corresponding coordinates are stored and printed out for each read. 

In Assignment 3, students will parallelize the read mapping kernel on the GPU, particularly the computation of the minimizer seeds and computing the mapping score for the seed hits.

In Assignment 4, students will implement pipeline parallelism for the different stages in software to maximize the CPU as well as GPU utilization.

## Setting up

Like before, we will be using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)) for these assignments.

To get set up with Assignment 3 and 4, please follow the steps below:

1. Open and accept the following GitHub Classroom invitation link for assignments 3 and 4 through your GitHub account: [https://classroom.github.com/a/7heevtuT](https://classroom.github.com/a/7heevtuT). A new repository for this will be created specifically for your account (e.g. https://github.com/ECE284-WI24/assgn3-4-yatisht) and an email will be sent to you via GitHub with the details. 

2. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. I recommend using PUTTY SSH client (putty.org) or Windows Subsystem for Linux (WSL) for Windows (https://docs.microsoft.com/en-us/windows/wsl/install-manual). MacOS and Linux users can SSH into the server using the following command (replace `yturakhia` with your username)

```
ssh yturakhia@dsmlp-login.ucsd.edu
```

3. Next, clone the assignment repository in your HOME directory using the following example command (replace repository name `assgn3-4-yatisht` with the correct name based on step 1) and decompress the data files:
```
cd ~
git clone https://github.com/ECE284-WI24/assgn3-4-yatisht
cd assgn3-4-yatisht/data
xz --decompress reference.fa.xz
xz --decompress reads.fa.xz
cd ~
```

4. Download a copy of the TBB version 2019_U9 into your HOME directory:

```
wget https://github.com/oneapi-src/oneTBB/archive/2019_U9.tar.gz
tar -xvzf 2019_U9.tar.gz
```

5. Review the source code (in the `src/` directory). In particular, search `TASK` (e.g. in `main.cpp` and `twoBitCompressor.cpp` for tasks related to Assignment ) and `HINT` in these files. Also review the `run-commands.sh` script. This script contains the commands that will be executed via the Docker container on the GPU instance. You may need to modify the commands this script depending on your experiment. Finally, make sure to also review the input test data files in the `data` directory. 
```
cd assgn3-4-yatisht
```

## Code development and testing

Once your environment is set up on the DSMLP server, you can begin code development and testing using either VS code (that many of you must be familiar with) or if you prefer, using the shell terminal itself (with text editors, such as Vim or Emacs). If you prefer the latter, you can skip the step 1 below.

1. Launch a VS code server from the DSMLP login server using the following command:
   ```
   /opt/launch-sh/bin/launch-codeserver -i ucsdets/datascience-notebook:2022.2-stable
   ```
   If successful, the log of the command will include a message such as:
   ```
   You may access your Code-Server (VS Code) at: http://dsmlp-login.ucsd.edu:14672 using password XXXXXX
   ```
   If the launch command is *unsuccessful*, make sure that there are no aleady running pods:
   ```
   # View running pods
   kubectl get pods
   # Delete all pods
   kubectl delete pod --all
   ```
   As conveyed in the message of the successful launch command, you can access the VS code server by going to the URL above (http://dsmlp-login.ucsd.edu:14672 in the above example) and entering the password displayed. Note that you may need to use UCSD's VPN service (https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/) if you are performing this step from outside the campus network. Once you gain access to the VS code server from your browser, you can view the directories and files in your DSMLP filesystem and develop code. You can also open a terminal (https://code.visualstudio.com/docs/editor/integrated-terminal) from the VS code interface and run commands on the login server.

2. We will be using a Docker container, namely `yatisht/ece284-wi24:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. This container already contains the correct Cmake version, CUDA and Boost libraries preinstalled within Ubuntu-18.04 OS. Note that these Docker containers use the same filesystem as the DSMLP login server, and hence the files written to or modified by the conainer is visiable to the login server and vice versa. To submit a job that executes `run-commands.sh` script located inside the `assgn3-4-yatisht` direcotry on a VM instance with 8 CPU cores, 16 GB RAM and 1 GPU device (this is the maxmimum allowed request on the DSMLP platform), the following command can be executed from the VS Code or DSMLP Shell Terminal (replace the username and directory names below appropriately):

```
ssh yturakhia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -v 2080ti -c 8 -g 1 -m 8 -i yatisht/ece284-wi24:latest -f ./assgn3-4-yatisht/run-commands.sh
```
Note that the above command will require you to enter your AD account password again. This command should work and provide a sensible output for the assignment already provided. If you have reached this, you are in good shape to develop and test the code (make sure to modify `run-commands.sh` appropriately before testing). Happy code development! 

## Submission guidelines

* Make sure to keep your code repository (e.g. https://github.com/ECE284-WI24/assgn3-4-yatisht) up-to-date.
* All new files (such as figures and reports) should be uploaded to the `submission-files` directory in the respository
* Once you are ready to submit, create a [new release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for your submission with tag names shown below. Provide a good description for the changes you have made and any information that you would like to be conveyed during the grading. 
  * Assignment 3: `v3.0`
  * Assignment 4: `v4.0`
* Submit the URL corresponding to the release to Canvas for your assignment submission (e.g. https://github.com/ECE284-WI24/assgn3-4-yatisht/releases/tag/v3.0; note that you are only required to submit a URL via Canvas). Only releases will be considered for grading and the date and time of the submitted release will be considered final for the late day policy. Be mindful of the deadlines.
