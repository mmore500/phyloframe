#!/bin/bash

set -e

cd "$(dirname "$0")"

python3 -m black .
python3 -m isort .
