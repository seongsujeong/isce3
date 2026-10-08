#!/usr/bin/env bash
# Configure, build and install isce3 into a conda environment made from
# environment.yml (conda env create -f environment.yml -n <env>).
#
# Usage: tools/build_isce3.sh [conda_env]      (default: isce3_dev)
# Optional environment variables:
#   BUILD_DIR   build directory               (default: <repo>/build)
#   INSTALL_DIR install prefix                (default: $BUILD_DIR/install)
#   WITH_METAL  ON/OFF, Metal GPU ampcor      (default: ON on macOS, else OFF)
#   WITH_CUDA   ON/OFF                        (default: OFF)
#   JOBS        parallel build jobs           (default: all cores)
set -eo pipefail   # no -u: conda activate scripts use unset variables

ENV_NAME=${1:-isce3_dev}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$REPO/build}
INSTALL_DIR=${INSTALL_DIR:-$BUILD_DIR/install}
WITH_CUDA=${WITH_CUDA:-OFF}

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

args=(-DCMAKE_BUILD_TYPE=Release
      -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
      -DISCE3_FETCH_DEPS=OFF
      -DWITH_CUDA="$WITH_CUDA")
if [[ $(uname) == Darwin ]]; then
    # SDK 27 + conda clang hides INFINITY under __STRICT_ANSI__; SIP strips
    # DYLD_* variables, so libisce3 is found through its install name
    args+=(-DISCE3_WITH_METAL="${WITH_METAL:-ON}"
           -DCMAKE_INSTALL_NAME_DIR="$INSTALL_DIR/lib"
           -DCMAKE_CXX_FLAGS=-U__STRICT_ANSI__
           -DCMAKE_OBJCXX_FLAGS=-U__STRICT_ANSI__)
else
    args+=(-DISCE3_WITH_METAL=OFF)
fi

cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja "${args[@]}"
cmake --build "$BUILD_DIR" ${JOBS:+-j "$JOBS"}
cmake --install "$BUILD_DIR"

# make the installed python packages importable in the environment
conda env config vars set -n "$ENV_NAME" PYTHONPATH="$INSTALL_DIR/packages" > /dev/null
echo "isce3 installed in $INSTALL_DIR; run 'conda activate $ENV_NAME' to pick up PYTHONPATH"
