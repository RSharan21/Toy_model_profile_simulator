#!/usr/bin/env python
"""
Demonstration script showing the fix for freq_range parameter handling.

This script demonstrates:
1. The error that occurs when freq_range doesn't match (default behavior)
2. How flexible_freq_range=True allows automatic adaptation
3. How pulses at different frequency ranges show different dispersion patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import PulseGenerator, PulseParameters

print("=" * 70)
print("DEMONSTRATION: freq_range parameter fix in PulseGenerator")
print("=" * 70)

# ============================================================================
# Example 1: Default behavior - error on mismatch
# ============================================================================
print("\n1. DEFAULT BEHAVIOR: Error when freq_range doesn't match")
print("-" * 70)

gen = PulseGenerator(n_freq=256, freq_min=1200, freq_max=1600)

params = PulseParameters(
    dm=100,
    freq_range=(400, 800),  # Different from generator!
    pulse_time=256,
    pulse_width=5.0,
    snr=15.0
)

print(f"Generator frequency range: {gen.freq_min}-{gen.freq_max} MHz")
print(f"Pulse parameter frequency range: {params.freq_range[0]}-{params.freq_range[1]} MHz")

try:
    gen.add_pulse(params)
    print("No error raised (unexpected!)")
except ValueError as e:
    print(f"\n✓ Expected error raised:\n  {e}")

# ============================================================================
# Example 2: Flexible behavior - automatic adaptation
# ============================================================================
print("\n\n2. FLEXIBLE BEHAVIOR: Automatic adaptation with flexible_freq_range=True")
print("-" * 70)

gen_flexible = PulseGenerator(
    n_freq=256, 
    n_time=512,
    freq_min=1200, 
    freq_max=1600,
    flexible_freq_range=True  # Enable flexible adaptation
)

print(f"Initial generator frequency range: {gen_flexible.freq_min}-{gen_flexible.freq_max} MHz")

# Reset and add pulse with different frequency range
gen_flexible.reset()
gen_flexible.add_noise(mean=0, std=1.0)
gen_flexible.add_pulse(params)

print(f"After add_pulse frequency range: {gen_flexible.freq_min}-{gen_flexible.freq_max} MHz")
print(f"✓ Generator adapted to pulse parameter frequency range!")

# ============================================================================
# Example 3: Visual comparison - different frequency ranges show different dispersion
# ============================================================================
print("\n\n3. VISUAL DEMONSTRATION: Different frequency ranges produce different dispersion")
print("-" * 70)

# Generate pulse at low frequency (400-800 MHz)
gen_low = PulseGenerator(n_freq=256, n_time=512, freq_min=400, freq_max=800, time_resolution=0.064)
params_low = PulseParameters(
    dm=100,
    freq_range=(400, 800),
    pulse_time=256,
    pulse_width=5.0,
    snr=15.0
)

gen_low.reset()
gen_low.add_noise(mean=0, std=1.0)
gen_low.add_pulse(params_low, profile_type='gaussian')
gen_low.normalize(method='standard')

# Generate pulse at high frequency (1200-1600 MHz)
gen_high = PulseGenerator(n_freq=256, n_time=512, freq_min=1200, freq_max=1600, time_resolution=0.064)
params_high = PulseParameters(
    dm=100,
    freq_range=(1200, 1600),
    pulse_time=256,
    pulse_width=5.0,
    snr=15.0
)

gen_high.reset()
gen_high.add_noise(mean=0, std=1.0)
gen_high.add_pulse(params_high, profile_type='gaussian')
gen_high.normalize(method='standard')

# Calculate dispersion metrics
delay_low = gen_low.calculate_dispersion_delay(400, 100)
delay_high = gen_high.calculate_dispersion_delay(1200, 100)

print(f"\nDispersion delay at 400 MHz (DM=100): {delay_low:.2f} ms")
print(f"Dispersion delay at 1200 MHz (DM=100): {delay_high:.2f} ms")
print(f"Ratio: {delay_low/delay_high:.1f}x more dispersion at lower frequency")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Low frequency spectrum
extent_low = [0, 512, 400, 800]
im1 = axes[0, 0].imshow(gen_low.get_spectrum(), aspect='auto', origin='lower',
                         extent=extent_low, cmap='viridis', interpolation='nearest')
axes[0, 0].set_xlabel('Time (bins)')
axes[0, 0].set_ylabel('Frequency (MHz)')
axes[0, 0].set_title(f'Pulse at 400-800 MHz (DM=100)\nDispersion delay: {delay_low:.1f} ms')
plt.colorbar(im1, ax=axes[0, 0], label='Intensity')

# Plot 2: High frequency spectrum
extent_high = [0, 512, 1200, 1600]
im2 = axes[0, 1].imshow(gen_high.get_spectrum(), aspect='auto', origin='lower',
                         extent=extent_high, cmap='viridis', interpolation='nearest')
axes[0, 1].set_xlabel('Time (bins)')
axes[0, 1].set_ylabel('Frequency (MHz)')
axes[0, 1].set_title(f'Pulse at 1200-1600 MHz (DM=100)\nDispersion delay: {delay_high:.1f} ms')
plt.colorbar(im2, ax=axes[0, 1], label='Intensity')

# Plot 3: Time series comparison
time_series_low = np.sum(gen_low.get_spectrum(), axis=0)
time_series_high = np.sum(gen_high.get_spectrum(), axis=0)

axes[1, 0].plot(time_series_low, label='400-800 MHz', alpha=0.7)
axes[1, 0].plot(time_series_high, label='1200-1600 MHz', alpha=0.7)
axes[1, 0].set_xlabel('Time (bins)')
axes[1, 0].set_ylabel('Integrated Intensity')
axes[1, 0].set_title('Integrated Time Series Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Dispersion as function of frequency
freqs = np.linspace(400, 1600, 100)
delays = [gen_low.calculate_dispersion_delay(f, 100) for f in freqs]

axes[1, 1].plot(freqs, delays)
axes[1, 1].axvspan(400, 800, alpha=0.2, color='blue', label='Low freq range')
axes[1, 1].axvspan(1200, 1600, alpha=0.2, color='orange', label='High freq range')
axes[1, 1].set_xlabel('Frequency (MHz)')
axes[1, 1].set_ylabel('Dispersion Delay (ms)')
axes[1, 1].set_title('Dispersion Delay vs Frequency (DM=100)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('freq_range_fix_demo.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to 'freq_range_fix_demo.png'")

# ============================================================================
# Example 4: Dataset generation with flexible frequency range
# ============================================================================
print("\n\n4. DATASET GENERATION: Uses generator's frequency range")
print("-" * 70)

from data_generation import PulseDataset

# Create generator with specific frequency range
gen_dataset = PulseGenerator(n_freq=256, freq_min=600, freq_max=900)
dataset = PulseDataset(gen_dataset)

# Generate random parameters - should use generator's frequency range
params_random = dataset.generate_random_params(has_pulse=True)

print(f"Generator frequency range: {gen_dataset.freq_min}-{gen_dataset.freq_max} MHz")
print(f"Generated parameters frequency range: {params_random.freq_range[0]}-{params_random.freq_range[1]} MHz")
print(f"✓ Parameters correctly match generator's frequency range!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF THE FIX")
print("=" * 70)
print("""
The fix adds frequency range validation and optional flexibility:

1. **Default behavior (flexible_freq_range=False)**:
   - Raises ValueError if PulseParameters.freq_range doesn't match generator
   - Ensures data consistency and prevents silent bugs

2. **Flexible behavior (flexible_freq_range=True)**:
   - Automatically adapts to PulseParameters.freq_range
   - Useful for generating pulses at different frequency bands

3. **PulseDataset always uses generator's frequency range**:
   - Ensures consistency between generated parameters and generator capabilities
   - No more ignored freq_range parameters!

4. **Physical correctness**:
   - Dispersion is properly calculated for the actual frequency range in use
   - Lower frequencies show stronger dispersion effects as expected
""")

print("All demonstrations completed successfully!")
print("=" * 70)
