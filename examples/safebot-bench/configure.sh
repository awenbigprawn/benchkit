#!/bin/sh
set -e

if [ ! -d "./deps/cpp" ]; then
    git clone git@gitlab.soft.vub.ac.be:safebot/demos/cpp.git ./deps/cpp
    cd ./deps/cpp
    git submodule update --init --recursive
else
    echo "./deps/cpp already exists, skipping clone and submodule update."
fi

../../scripts/install_venv.sh ./venv --pythainer_from_src
