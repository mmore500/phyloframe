#!/bin/bash

set -e

cd "$(dirname "$0")/.."

./style.sh
git diff --exit-code
