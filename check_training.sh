#!/bin/bash
# Quick script to check if training is running

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

# Check log file
if [ -f logs/training_transfer.log ]; then
    SIZE=$(ls -lh logs/training_transfer.log | awk '{print $5}')
    LINES=$(wc -l < logs/training_transfer.log)
    echo "📄 Log file: logs/training_transfer.log"
    echo "   Size: $SIZE"
    echo "   Lines: $LINES"
    echo ""
    echo "Last 3 lines:"
    tail -3 logs/training_transfer.log
else
    echo "❌ Log file not found (logs/training_transfer.log)"
fi

echo ""
echo "=== Quick Commands ==="
echo "  Watch live: tail -f logs/training_transfer.log"
echo "  Stop training: pkill -f 'python.*train'"
