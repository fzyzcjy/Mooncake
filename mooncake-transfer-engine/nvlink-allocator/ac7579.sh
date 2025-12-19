#!/bin/bash

set -e

SCRIPT_DIR=$(dirname $(readlink -f $0))
CPP_FILE="$SCRIPT_DIR/ac7579.cpp"
OUTPUT="$SCRIPT_DIR/ac7579"

echo "Compiling $CPP_FILE..."
g++ "$CPP_FILE" -o "$OUTPUT" -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64

if [ $? -eq 0 ]; then
    echo "Compilation succeeded: $OUTPUT"
    echo ""
    echo "Running $OUTPUT..."
    echo "========================================"
    "$OUTPUT"
else
    echo "Compilation failed"
    exit 1
fi
