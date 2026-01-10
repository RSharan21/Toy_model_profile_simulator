
import numpy as np
from numba import jit, prange, set_num_threads
from dm_time_plot_utils import plot_dm_time_simple

# Limit to 10 threads instead of all cores
set_num_threads(10)

from typing import Tuple, Optional

@jit(nopython=True, parallel=True, fastmath=True)
def dm_time_transform(dynamic_spectrum, frequencies, time_resolution, 
					 dm_trials, ref_freq=None):
	"""
	Apply DM-time transform to dynamic spectrum
	
	For each trial DM: 
	  1. Calculate dispersion delays for each frequency
	  2. Shift each frequency channel to dedisperse
	  3. Sum across frequency axis to get time series
	
	Parameters:
	-----------
	dynamic_spectrum : np.ndarray (float32)
		2D array (n_freq, n_time) - the input dynamic spectrum
	frequencies : np.ndarray (float32)
		1D array (n_freq,) - frequency of each channel in MHz
	time_resolution : float32
		Time resolution in milliseconds
	dm_trials : np.ndarray (float32)
		1D array (n_dm_trials,) - trial DM values in pc/cm³
	ref_freq : float32, optional
		Reference frequency in MHz (if None, uses highest frequency)
	
	Returns:
	--------
	dm_time_array : np.ndarray (float32)
		2D array (n_dm_trials, n_time) - dedispersed time series for each DM
	"""
	n_freq, n_time = dynamic_spectrum.shape
	n_dm_trials = len(dm_trials)
	
	# Use highest frequency as reference if not specified
	if ref_freq is None or ref_freq <= 0:
		ref_freq = frequencies.max()
	
	# Output array
	dm_time_array = np.zeros((n_dm_trials, n_time), dtype=np.float32)
	
	# Dispersion constant:  k_DM = 4.148808 ms
	k_dm = np.float32(4.148808)
	
	# Convert reference frequency to GHz
	ref_freq_ghz = ref_freq / 1000.0
	
	# Process each trial DM in parallel
	for dm_idx in prange(n_dm_trials):
		dm = dm_trials[dm_idx]
		
		# Temporary time series for this DM
		time_series = np.zeros(n_time, dtype=np.float32)
		
		# Process each frequency channel
		for f_idx in range(n_freq):
			freq = frequencies[f_idx]
			freq_ghz = freq / 1000.0
			
			# Calculate dispersion delay relative to reference frequency
			# delay = k_DM * DM * (1/f^2 - 1/f_ref^2)
			delay_ms = k_dm * dm * (1.0 / (freq_ghz * freq_ghz) - 
									1.0 / (ref_freq_ghz * ref_freq_ghz))
			
			# Convert delay to bins
			delay_bins = int(delay_ms / time_resolution + 0.5)
			
			# Shift and add this frequency channel
			if delay_bins > 0:
				# Positive delay: this frequency arrives later, shift left to dedisperse
				for t in range(n_time - delay_bins):
					time_series[t] += dynamic_spectrum[f_idx, t + delay_bins]
			elif delay_bins < 0:
				# Negative delay: this frequency arrives earlier, shift right
				for t in range(-delay_bins, n_time):
					time_series[t] += dynamic_spectrum[f_idx, t + delay_bins]
			else:
				# No delay
				for t in range(n_time):
					time_series[t] += dynamic_spectrum[f_idx, t]
		
		# Store time series for this DM
		dm_time_array[dm_idx, :] = time_series
	
	return dm_time_array


@jit(nopython=True, parallel=True, fastmath=True)
def dm_time_transform_optimized(dynamic_spectrum, frequencies, time_resolution, 
								dm_trials, ref_freq=None):
	"""
	Optimized DM-time transform with pre-computed delays
	
	This version pre-computes all delays once, then applies them.
	Faster for large number of DM trials.
	
	Parameters:
	-----------
	dynamic_spectrum : np.ndarray (float32)
		2D array (n_freq, n_time)
	frequencies : np.ndarray (float32)
		1D array (n_freq,) in MHz
	time_resolution : float32
		Time resolution in ms
	dm_trials : np.ndarray (float32)
		1D array (n_dm_trials,) in pc/cm³
	ref_freq : float32, optional
		Reference frequency in MHz
	
	Returns:
	--------
	dm_time_array : np.ndarray (float32)
		2D array (n_dm_trials, n_time)
	"""
	n_freq, n_time = dynamic_spectrum.shape
	n_dm_trials = len(dm_trials)
	
	if ref_freq is None or ref_freq <= 0:
		ref_freq = frequencies.max()
	
	# Pre-compute delay matrix:  (n_freq, n_dm_trials)
	k_dm = np.float32(4.148808)
	ref_freq_ghz = ref_freq / 1000.0
	
	delay_matrix = np.zeros((n_freq, n_dm_trials), dtype=np.int32)
	
	for f_idx in range(n_freq):
		freq_ghz = frequencies[f_idx] / 1000.0
		freq_factor = 1.0 / (freq_ghz * freq_ghz) - 1.0 / (ref_freq_ghz * ref_freq_ghz)
		
		for dm_idx in range(n_dm_trials):
			delay_ms = k_dm * dm_trials[dm_idx] * freq_factor
			delay_matrix[f_idx, dm_idx] = int(delay_ms / time_resolution + 0.5)
	
	# Apply delays and sum
	dm_time_array = np.zeros((n_dm_trials, n_time), dtype=np.float32)
	
	for dm_idx in prange(n_dm_trials):
		for f_idx in range(n_freq):
			delay = delay_matrix[f_idx, dm_idx]
			
			if delay > 0 and delay < n_time:
				#print('Case 1 ')
				for t in range(n_time - delay):
					dm_time_array[dm_idx, t] += dynamic_spectrum[f_idx, t + delay]
			elif delay <= 0 and -delay < n_time:
				#print('Case 2 ')
				for t in range(-delay, n_time):
					dm_time_array[dm_idx, t] += dynamic_spectrum[f_idx, t + delay]
			else:
				# No significant delay
				#print('Case 3 ')
				for t in range(n_time):
					dm_time_array[dm_idx, t] += dynamic_spectrum[f_idx, t]
	
	return dm_time_array


@jit(nopython=True, parallel=False, fastmath=True)
def dm_time_transform_rolled(dynamic_spectrum, frequencies, time_resolution, 
							 dm_trials, ref_freq=None):
	"""
	DM-time transform using array operations (slightly different approach)
	
	This version is more memory efficient and can be faster for certain sizes.
	
	Parameters:
	-----------
	dynamic_spectrum :  np.ndarray (float32)
		2D array (n_freq, n_time)
	frequencies : np.ndarray (float32)
		1D array (n_freq,) in MHz
	time_resolution : float32
		Time resolution in ms
	dm_trials : np.ndarray (float32)
		1D array (n_dm_trials,) in pc/cm³
	ref_freq : float32, optional
		Reference frequency in MHz
	
	Returns:
	--------
	dm_time_array : np.ndarray (float32)
		2D array (n_dm_trials, n_time)
	"""
	n_freq, n_time = dynamic_spectrum.shape
	n_dm_trials = len(dm_trials)
	
	if ref_freq is None or ref_freq <= 0:
		ref_freq = frequencies.max()
	
	dm_time_array = np.zeros((n_dm_trials, n_time), dtype=np.float32)
	
	k_dm = np.float32(4.148808)
	ref_freq_ghz = ref_freq / 1000.0
	
	# Temporary buffer for shifted data
	shifted = np.zeros(n_time, dtype=np.float32)
	
	for dm_idx in range(n_dm_trials):
		dm = dm_trials[dm_idx]
		
		for f_idx in range(n_freq):
			freq_ghz = frequencies[f_idx] / 1000.0
			
			# Calculate delay
			delay_ms = k_dm * dm * (1.0 / (freq_ghz * freq_ghz) - 
									1.0 / (ref_freq_ghz * ref_freq_ghz))
			delay_bins = int(delay_ms / time_resolution + 0.5)
			
			# Clear shifted buffer
			for t in range(n_time):
				shifted[t] = 0.0
			
			# Apply shift
			if delay_bins > 0:
				for t in range(n_time - delay_bins):
					shifted[t] = dynamic_spectrum[f_idx, t + delay_bins]
			elif delay_bins < 0:
				for t in range(-delay_bins, n_time):
					shifted[t] = dynamic_spectrum[f_idx, t + delay_bins]
			else:
				for t in range(n_time):
					shifted[t] = dynamic_spectrum[f_idx, t]
			
			# Add to time series
			for t in range(n_time):
				dm_time_array[dm_idx, t] += shifted[t]
	
	return dm_time_array

class DMTimeTransform:
	"""
	DM-Time Transform for dynamic spectra
	"""
	plot_dm_time_simple = plot_dm_time_simple
	def __init__(self, frequencies:  np.ndarray, time_resolution: float, 
				 ref_freq: Optional[float] = None):
		"""
		Initialize DM-Time transform
		
		Parameters:
		-----------
		frequencies : np.ndarray
			Frequency of each channel in MHz
		time_resolution : float
			Time resolution in milliseconds
		ref_freq : float, optional
			Reference frequency in MHz (defaults to max frequency)
		"""
		self.frequencies = frequencies.astype(np.float32)
		self.time_resolution = np.float32(time_resolution)
		self.ref_freq = np.float32(ref_freq if ref_freq else frequencies.max())
		self.n_freq = len(frequencies)
	
	def transform(self, dynamic_spectrum: np.ndarray, 
				 dm_min: float = 0, dm_max: float = 2000, 
				 dm_step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Apply DM-time transform to dynamic spectrum
		
		Parameters:
		-----------
		dynamic_spectrum : np.ndarray
			2D array (n_freq, n_time)
		dm_min : float
			Minimum trial DM in pc/cm³
		dm_max : float
			Maximum trial DM in pc/cm³
		dm_step : float
			DM step size in pc/cm³
		
		Returns:
		--------
		dm_time_array : np.ndarray
			2D array (n_dm_trials, n_time) - the DM-time plane
		dm_trials : np.ndarray
			1D array (n_dm_trials,) - the DM values used
		"""
		# Generate DM trials
		dm_trials = np.arange(dm_min, dm_max + dm_step, dm_step, dtype=np.float32)
		
		# Ensure input is float32
		spectrum = dynamic_spectrum.astype(np.float32).copy()
		
		# Apply transform
		dm_time_array = dm_time_transform_optimized(
			spectrum, self.frequencies, self.time_resolution, 
			dm_trials, self.ref_freq
		)
		
		return dm_time_array, dm_trials
	
	def find_best_dm(self, dynamic_spectrum: np.ndarray,
					dm_min: float = 0, dm_max: float = 2000,
					dm_step: float = 1.0) -> Tuple[float, np.ndarray]:
		"""
		Find best DM by maximizing S/N in DM-time plane
		
		Parameters:
		-----------
		dynamic_spectrum : np.ndarray
			2D array (n_freq, n_time)
		dm_min, dm_max, dm_step : float
			DM search range
		
		Returns: 
		--------
		best_dm : float
			DM with highest S/N
		dm_time_array : np.ndarray
			The full DM-time plane
		"""
		
		if not hasattr(self, 'spectrum'):
			self.spectrum = dynamic_spectrum.copy()
			
		if not hasattr(self, 'n_time'):
			self.n_time = dynamic_spectrum.shape[-1]
		self.dm_time_array, self.dm_trials = self.transform(
			dynamic_spectrum, dm_min, dm_max, dm_step
		)
		
		# Find DM with maximum peak S/N
		snr_per_dm = np.max(self.dm_time_array, axis=1)
		best_dm_idx = np.argmax(snr_per_dm)
		self.best_dm = self.dm_trials[best_dm_idx]
		
		return self.best_dm, self.dm_time_array, self.dm_trials
		
		
		

	
	
	
