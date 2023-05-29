# File       : Makefile
# Description: Compile the targets.
CXX ?= g++
CXXFLAGS = -std=c++11 -g -O3 -DNDEBUG -Wall -Wextra -Wpedantic -fopenmp -mfma -mavx2
.PHONY: clean

all: parallel

parallel: parallel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f parallel
