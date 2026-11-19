#!/bin/sh
set -eu
find /usr/local/cuda -name "cupti*" -type f 2>/dev/null | head -20
find /usr/local/cuda -name "libcupti*" 2>/dev/null | head -10
ls /usr/local/cuda/extras/ 2>/dev/null || echo "no extras"
ls /usr/local/cuda/include/cupti*.h 2>/dev/null || echo "no cupti headers"
ls /usr/local/cuda/targets/sbsa-linux/lib/libcupti* 2>/dev/null || echo "no cupti libs"
