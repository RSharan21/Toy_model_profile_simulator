import numpy as np
import time
from numba import jit, prange

# Setup
n_freq = 256
n_time = 10000
base_profile = np.random.randn(n_time).astype(np.float32)
delay_bins = np.random.randint(-100, 500, n_freq).astype(np.int32)
spectral_amps = np.random.randn(n_freq).astype(np.float32)
snr = 15.0

# ============================================================================
# Method 1: Original nested loop (Numba JIT)
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def method1_nested_loop(base_profile, delay_bins, spectral_amps, snr):
    n_freq = len(delay_bins)
    n_time = len(base_profile)
    pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
    
    for f_idx in prange(n_freq):
        delay = delay_bins[f_idx]
        amp = spectral_amps[f_idx] * snr
        
        for t in range(n_time):
            t_shifted = t - delay
            if 0 <= t_shifted < n_time:
                pulse_2d[f_idx, t] = base_profile[t_shifted] * amp
    
    return pulse_2d

# ============================================================================
# Method 2: np.roll in loop (Pure NumPy)
# ============================================================================

def method2_roll_loop(base_profile, delay_bins, spectral_amps, snr):
    n_freq = len(delay_bins)
    n_time = len(base_profile)
    pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
    
    for f_idx in range(n_freq):
        delay = delay_bins[f_idx]
        amp = spectral_amps[f_idx] * snr
        
        if delay >= 0:
            shifted_profile = np.roll(base_profile, delay)
            shifted_profile[: delay] = 0
        else:
            shifted_profile = base_profile.copy()
        
        pulse_2d[f_idx, :] = shifted_profile * amp
    
    return pulse_2d

# ============================================================================
# Method 3: np.roll in Numba loop
# ============================================================================

@jit(nopython=True, parallel=True)
def method3_roll_numba(base_profile, delay_bins, spectral_amps, snr):
    n_freq = len(delay_bins)
    n_time = len(base_profile)
    pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
    
    for f_idx in prange(n_freq):
        delay = delay_bins[f_idx]
        amp = spectral_amps[f_idx] * snr
        
        if delay >= 0:
            shifted_profile = np.roll(base_profile, delay)
            shifted_profile[: delay] = 0
            pulse_2d[f_idx, :] = shifted_profile * amp
        else:
            pulse_2d[f_idx, :] = base_profile * amp
    
    return pulse_2d

# ============================================================================
# Method 4: Fully vectorized with advanced indexing
# ============================================================================

def method4_vectorized(base_profile, delay_bins, spectral_amps, snr):
    n_freq = len(delay_bins)
    n_time = len(base_profile)
    
    # Create index array
    time_indices = np.arange(n_time)
    
    # Compute shifted indices for all frequencies at once
    # Shape: (n_freq, n_time)
    shifted_indices = (time_indices[np.newaxis, :] - delay_bins[: , np.newaxis])
    
    # Create mask for valid indices
    valid_mask = (shifted_indices >= 0) & (shifted_indices < n_time)
    
    # Initialize output
    pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
    
    # Apply shifts where valid
    pulse_2d[valid_mask] = base_profile[shifted_indices[valid_mask]]
    
    # Apply amplitude scaling (vectorized)
    pulse_2d *= (spectral_amps[: , np.newaxis] * snr)
    
    return pulse_2d

# ============================================================================
# Method 5: Optimized vectorized (best)
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def method5_optimized(base_profile, delay_bins, spectral_amps, snr):
    """
    Combines benefits of JIT compilation with smart indexing
    """
    n_freq = len(delay_bins)
    n_time = len(base_profile)
    pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
    
    for f_idx in prange(n_freq):
        delay = delay_bins[f_idx]
        amp = spectral_amps[f_idx] * snr
        
        if delay >= 0:
            # Copy valid region
            copy_len = n_time - delay
            pulse_2d[f_idx, delay:] = base_profile[: copy_len] * amp
        else:
            # Negative delay (shouldn't happen for dispersion, but handle it)
            copy_len = n_time + delay
            pulse_2d[f_idx, :copy_len] = base_profile[-delay:] * amp
    
    return pulse_2d

# ============================================================================
# Benchmark
# ============================================================================

# Warm up JIT
_ = method1_nested_loop(base_profile, delay_bins, spectral_amps, snr)
_ = method3_roll_numba(base_profile, delay_bins, spectral_amps, snr)
_ = method5_optimized(base_profile, delay_bins, spectral_amps, snr)

n_runs = 1000

print("="*70)
print("BENCHMARK RESULTS (averaged over {} runs)".format(n_runs))
print("="*70)

# Method 1
t0 = time.time()
for _ in range(n_runs):
    result1 = method1_nested_loop(base_profile, delay_bins, spectral_amps, snr)
t1 = time.time()
time1 = (t1 - t0) / n_runs
print(f"Method 1 (Nested loop + Numba JIT):     {time1*1000:.3f} ms")

# Method 2
t0 = time.time()
for _ in range(n_runs):
    result2 = method2_roll_loop(base_profile, delay_bins, spectral_amps, snr)
t1 = time.time()
time2 = (t1 - t0) / n_runs
print(f"Method 2 (np.roll + Python loop):       {time2*1000:.3f} ms  (Speedup: {time2/time1:.2f}x)")

# Method 3
t0 = time.time()
for _ in range(n_runs):
    result3 = method3_roll_numba(base_profile, delay_bins, spectral_amps, snr)
t1 = time.time()
time3 = (t1 - t0) / n_runs
print(f"Method 3 (np.roll + Numba JIT):          {time3*1000:.3f} ms  (Speedup: {time3/time1:.2f}x)")

# Method 4
t0 = time.time()
for _ in range(n_runs):
    result4 = method4_vectorized(base_profile, delay_bins, spectral_amps, snr)
t1 = time.time()
time4 = (t1 - t0) / n_runs
print(f"Method 4 (Fully vectorized):             {time4*1000:.3f} ms  (Speedup: {time4/time1:.2f}x)")

# Method 5
t0 = time.time()
for _ in range(n_runs):
    result5 = method5_optimized(base_profile, delay_bins, spectral_amps, snr)
t1 = time.time()
time5 = (t1 - t0) / n_runs
print(f"Method 5 (Optimized + Numba JIT):        {time5*1000:.3f} ms  (Speedup:  {time1/time5:.2f}x FASTER)")

print("\n" + "="*70)
print("Verify all methods give same result:")
print(f"  Method 1 vs 2: {np.allclose(result1, result2)}")
print(f"  Method 1 vs 3: {np.allclose(result1, result3)}")
print(f"  Method 1 vs 4: {np.allclose(result1, result4)}")
print(f"  Method 1 vs 5: {np.allclose(result1, result5)}")
print("="*70)
