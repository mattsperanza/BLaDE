#! /bin/bash -l

source modules

mkdir build
cd build
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src

make $1 lady
