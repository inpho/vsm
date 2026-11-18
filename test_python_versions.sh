#!/bin/bash

# Test script to verify README.md instructions work across multiple Python versions
# Uses uv to manage environments and build the Cython extension

VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14", "3.15")
BASE_DIR="/tmp/vsm_test_$(date +%s)"
SOURCE_DIR="$(pwd)"

echo "Starting Python version testing..."
echo "Source directory: $SOURCE_DIR"
echo "Test directory: $BASE_DIR"
echo ""

# Create test directory
mkdir -p "$BASE_DIR"
cp -r "$SOURCE_DIR" "$BASE_DIR/vsm"

cd "$BASE_DIR/vsm"

# Test each Python version
for VERSION in "${VERSIONS[@]}"; do
    echo "========================================="
    echo "Testing with Python $VERSION"
    echo "========================================="
    
    # Create virtual environment with specific Python version
    echo "Creating virtual environment with Python $VERSION..."

    uv venv --python "$VERSION" --clear
    
    # Sync dependencies and build Cython extension
    echo "Syncing dependencies and building Cython extension..."
    uv sync --group dev
    
    # Check if compilation succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Dependencies synced successfully"
        
        # Run unit tests
        echo "Running unit tests..."
        uv run pytest unit_tests/*
        
        TEST_RESULT=$?
        
        if [ $TEST_RESULT -eq 0 ]; then
            echo "✓ All unit tests passed"
        else
            echo "✗ Some unit tests failed (exit code: $TEST_RESULT)"
        fi
    else
        echo "✗ Dependencies sync failed (exit code: $?)"
    fi
    
    echo ""
done

echo "========================================="
echo "Testing complete"
echo "========================================="
echo "Test directory: $BASE_DIR"
echo "You can inspect the results in: $BASE_DIR/vsm"
