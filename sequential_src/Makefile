# File       : Makefile
# Description: Compile the targets.
CXX ?= g++
CXXFLAGS = -std=c++11 -g -DNDEBUG -Wall -Wextra -Wpedantic -Isrc
.PHONY: clean

all: sequential

sequential: sequential.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f sequential
