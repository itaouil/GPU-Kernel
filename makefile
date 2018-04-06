#
# This makefile should run on the School machines or on a Mac.
# Note however that for School machines you will need to start by
# loading the CUDA module:
#
# module load cuda/7.5
#
EXE = cwk3
OS = $(shell uname)

ifeq ($(OS), Linux)
	CC = nvcc
    LIBS = -lOpenCL
    CCFLAGS = 
endif

ifeq ($(OS), Darwin)
	CC = gcc
	CCFLAGS = -Wall
    LIBS = -framework OpenCL
endif

all:
	$(CC) $(LIBS) $(CCFLAGS) -o $(EXE) cwk3.c
