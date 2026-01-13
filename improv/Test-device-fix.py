#!/usr/bin/env python3
"""
Quick test script to verify device handling is working correctly.
Tests both CPU and GPU (if available) modes.
"""

import torch
from secure_federated_learning import FLConfig, run_mnist_experiment, SelectionMethod

print("=" * 60)
print("Device Handling Test")
print("=" * 60)

# Check available device
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    test_gpu = True
else:
    print("✓ CUDA not available, testing CPU only")
    test_gpu = False

print("\n" + "=" * 60)
print("Test 1: CPU Mode with Minimal Config")
print("=" * 60)

config_cpu = FLConfig(
    num_clients=5,
    num_rounds=2,
    local_epochs=1,
    use_dp=True,
    use_homomorphic_encryption=False,
    use_secure_aggregation=False,
    selection_method=SelectionMethod.MULTI_KRUM,
    enable_hierarchical_defense=False,
    malicious_ratio=0.2,
)

try:
    server = run_mnist_experiment(config_cpu)
    print("\n✓ CPU test passed!")
except Exception as e:
    print(f"\n✗ CPU test failed: {e}")
    import traceback
    traceback.print_exc()

if test_gpu:
    print("\n" + "=" * 60)
    print("Test 2: GPU Mode with Minimal Config")
    print("=" * 60)
    
    config_gpu = FLConfig(
        num_clients=5,
        num_rounds=2,
        local_epochs=1,
        use_dp=True,
        use_homomorphic_encryption=False,
        use_secure_aggregation=False,
        selection_method=SelectionMethod.MULTI_KRUM,
        enable_hierarchical_defense=False,
        malicious_ratio=0.2,
    )
    
    try:
        server = run_mnist_experiment(config_gpu)
        print("\n✓ GPU test passed!")
    except Exception as e:
        print(f"\n✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("All Tests Complete!")
print("=" * 60)
