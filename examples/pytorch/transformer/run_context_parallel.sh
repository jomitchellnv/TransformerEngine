#!/bin/bash
torchrun --nproc_per_node=2 --master_port=29501 context_parallel_runner.py


# Run pytest on the context parallel tests
echo "Running Context Parallel Tests with pytest..."
echo "=============================================="

cd "$(dirname "$0")"

# Check if test data exists
if [ ! -f "/tmp/cp1_results.pt" ] || [ ! -f "/tmp/cp2_rank_0_results.pt" ] || [ ! -f "/tmp/cp2_rank_1_results.pt" ] || [ ! -f "/tmp/data.pt" ]; then
    echo "❌ Test data not found. Please run the distributed test first:"
    echo "   bash run_context_parallel.sh"
    exit 1
fi

# Run pytest with verbose output
python -m pytest test_context_parallel.py -v -s --tb=short

echo ""
echo "Test completed!"

# Now delete the test data
rm -f /tmp/cp1_results.pt
rm -f /tmp/cp2_rank_0_results.pt
rm -f /tmp/cp2_rank_1_results.pt
rm -f /tmp/data.pt