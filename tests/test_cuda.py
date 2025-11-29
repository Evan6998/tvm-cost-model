#!/usr/bin/env python3
"""Quick test to verify TVM CUDA support"""

import tvm

print("=" * 60)
print("TVM CUDA Support Test")
print("=" * 60)
print(f"TVM version: {tvm.__version__}")
print(f"CUDA device available: {tvm.cuda().exist}")
print(f"CUDA runtime enabled: {tvm.runtime.enabled('cuda')}")

if tvm.cuda().exist:
    print(f"CUDA device name: {tvm.cuda().device_name}")
    print("✓ TVM CUDA support is working!")
else:
    print("✗ No CUDA devices found")
print("=" * 60)

