#!/bin/sh
set -eu
df -h /work /tmp / 2>/dev/null
echo "---"
ls -la /work/ 2>/dev/null | head -10
echo "---"
# Is /work persistent across jobs? Check for our previous output
ls /work/out/ 2>/dev/null | head -10
ls /work/mmh3-target/ 2>/dev/null | head -5
