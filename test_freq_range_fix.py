#!/usr/bin/env python
"""
Test script to verify that freq_range parameter in PulseParameters is properly handled.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import PulseGenerator, PulseParameters

def test_freq_range_mismatch_error():
    """Test that error is raised when freq_range doesn't match and flexible=False"""
    print("\n=== Test 1: Error on mismatch (flexible=False) ===")
    
    gen = PulseGenerator(n_freq=256, freq_min=1200, freq_max=1600, flexible_freq_range=False)
    
    params = PulseParameters(
        dm=100,
        freq_range=(400, 800),  # Different from generator
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    try:
        gen.add_pulse(params)
        print("FAILED: Expected ValueError but none was raised")
        return False
    except ValueError as e:
        print(f"PASSED: Got expected error: {e}")
        return True

def test_freq_range_flexible_adaptation():
    """Test that generator adapts when flexible=True"""
    print("\n=== Test 2: Adaptation with flexible=True ===")
    
    gen = PulseGenerator(n_freq=256, freq_min=1200, freq_max=1600, flexible_freq_range=True)
    
    print(f"Initial freq range: {gen.freq_min}-{gen.freq_max} MHz")
    print(f"Initial frequencies: {gen.frequencies[0]:.2f} to {gen.frequencies[-1]:.2f} MHz")
    
    params = PulseParameters(
        dm=100,
        freq_range=(400, 800),  # Different from generator
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gen.reset()
        gen.add_noise(mean=0, std=1.0)
        gen.add_pulse(params)
        
        # Check if warning was raised
        if len(w) > 0:
            print(f"Got expected warning: {w[0].message}")
        
    print(f"After add_pulse freq range: {gen.freq_min}-{gen.freq_max} MHz")
    print(f"After add_pulse frequencies: {gen.frequencies[0]:.2f} to {gen.frequencies[-1]:.2f} MHz")
    
    # Verify frequency range was updated
    if abs(gen.freq_min - 400) < 1e-6 and abs(gen.freq_max - 800) < 1e-6:
        print("PASSED: Frequency range correctly adapted to (400, 800) MHz")
        return True
    else:
        print(f"FAILED: Expected freq range (400, 800), got ({gen.freq_min}, {gen.freq_max})")
        return False

def test_freq_range_matching():
    """Test that no error occurs when freq_range matches"""
    print("\n=== Test 3: No error when ranges match ===")
    
    gen = PulseGenerator(n_freq=256, freq_min=1200, freq_max=1600, flexible_freq_range=False)
    
    params = PulseParameters(
        dm=100,
        freq_range=(1200, 1600),  # Matches generator
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    try:
        gen.reset()
        gen.add_noise(mean=0, std=1.0)
        gen.add_pulse(params)
        print("PASSED: No error when frequency ranges match")
        return True
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False

def test_dispersion_at_different_freq_ranges():
    """Test that dispersion is calculated correctly for different frequency ranges"""
    print("\n=== Test 4: Dispersion at different frequency ranges ===")
    
    # Test at 400-800 MHz range
    gen1 = PulseGenerator(n_freq=256, n_time=512, freq_min=400, freq_max=800, flexible_freq_range=True)
    params1 = PulseParameters(
        dm=100,
        freq_range=(400, 800),
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    gen1.reset()
    gen1.add_noise(mean=0, std=1.0)
    gen1.add_pulse(params1)
    spectrum1 = gen1.get_spectrum()
    
    # Test at 1200-1600 MHz range
    gen2 = PulseGenerator(n_freq=256, n_time=512, freq_min=1200, freq_max=1600, flexible_freq_range=True)
    params2 = PulseParameters(
        dm=100,
        freq_range=(1200, 1600),
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    gen2.reset()
    gen2.add_noise(mean=0, std=1.0)
    gen2.add_pulse(params2)
    spectrum2 = gen2.get_spectrum()
    
    # Calculate dispersion delays at lowest frequency for both ranges
    delay1 = gen1.calculate_dispersion_delay(400, 100)  # at 400 MHz
    delay2 = gen2.calculate_dispersion_delay(1200, 100)  # at 1200 MHz
    
    print(f"Dispersion delay at 400 MHz (DM=100): {delay1:.2f} ms")
    print(f"Dispersion delay at 1200 MHz (DM=100): {delay2:.2f} ms")
    
    # Lower frequencies should have larger dispersion delays
    if delay1 > delay2:
        print("PASSED: Lower frequency range has larger dispersion delay as expected")
        return True
    else:
        print(f"FAILED: Expected delay1 > delay2, got {delay1} <= {delay2}")
        return False

def test_dataset_generation():
    """Test that PulseDataset correctly uses generator's frequency range"""
    print("\n=== Test 5: Dataset generation uses generator's freq range ===")
    
    from data_generation import PulseDataset
    
    gen = PulseGenerator(n_freq=256, freq_min=400, freq_max=800)
    dataset = PulseDataset(gen)
    
    # Generate random parameters
    params = dataset.generate_random_params(has_pulse=True)
    
    print(f"Generator freq range: {gen.freq_min}-{gen.freq_max} MHz")
    print(f"Generated params freq range: {params.freq_range[0]}-{params.freq_range[1]} MHz")
    
    # Verify generated params match generator's range
    if params.freq_range == (gen.freq_min, gen.freq_max):
        print("PASSED: Generated parameters use generator's frequency range")
        return True
    else:
        print(f"FAILED: Mismatch in frequency ranges")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing frequency range parameter handling")
    print("=" * 60)
    
    results = []
    results.append(("Mismatch Error Test", test_freq_range_mismatch_error()))
    results.append(("Flexible Adaptation Test", test_freq_range_flexible_adaptation()))
    results.append(("Matching Range Test", test_freq_range_matching()))
    results.append(("Dispersion Test", test_dispersion_at_different_freq_ranges()))
    results.append(("Dataset Generation Test", test_dataset_generation()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
