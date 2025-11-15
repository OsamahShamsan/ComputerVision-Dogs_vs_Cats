#!/bin/bash
# Quick script to check if training is running
# Checks for training processes and log files in both partial_dataset and full_dataset directories

echo "=== Training Status Check ==="
echo ""

# Check if process is running
if ps aux | grep "python.*train" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    ps aux | grep "python.*train" | grep -v grep | awk '{print "   PID: " $2 " | Started: " $9}'
else
    echo "❌ Training is NOT running"
fi

echo ""

# Check log files in partial_dataset and full_dataset
LOG_FOUND=false

# Check partial dataset logs
if [ -f logs/partial_dataset/training_transfer.log ]; then
    SIZE=$(ls -lh logs/partial_dataset/training_transfer.log | awk '{print $5}')
    LINES=$(wc -l < logs/partial_dataset/training_transfer.log)
    echo "📄 Log file (Partial Dataset): logs/partial_dataset/training_transfer.log"
    echo "   Size: $SIZE"
    echo "   Lines: $LINES"
    echo ""
    echo "Last 3 lines:"
    tail -3 logs/partial_dataset/training_transfer.log
    LOG_FOUND=true
fi

# Check full dataset logs
if [ -f logs/full_dataset/training_transfer.log ]; then
    SIZE=$(ls -lh logs/full_dataset/training_transfer.log | awk '{print $5}')
    LINES=$(wc -l < logs/full_dataset/training_transfer.log)
    echo ""
    echo "📄 Log file (Full Dataset): logs/full_dataset/training_transfer.log"
    echo "   Size: $SIZE"
    echo "   Lines: $LINES"
    echo ""
    echo "Last 3 lines:"
    tail -3 logs/full_dataset/training_transfer.log
    LOG_FOUND=true
fi

if [ "$LOG_FOUND" = false ]; then
    echo "❌ No log files found in logs/partial_dataset/ or logs/full_dataset/"
    echo "   Expected: logs/partial_dataset/training_transfer.log or logs/full_dataset/training_transfer.log"
fi

echo ""
echo "=== Quick Commands ==="
echo "  Watch live (partial): tail -f logs/partial_dataset/training_transfer.log"
echo "  Watch live (full): tail -f logs/full_dataset/training_transfer.log"
echo "  Stop training: pkill -f 'python.*train'"
