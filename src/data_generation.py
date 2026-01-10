import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import matplotlib.pyplot as plt
from numba import jit, prange, set_num_threads
from spectrum_plot_utils import *


# Limit to 10 threads instead of all cores
set_num_threads(10)

@dataclass
class PulseParameters:
	"""Store physical parameters for pulse generation"""
	dm: float  # Dispersion measure (pc/cm^3)
	freq_range: Tuple[float, float]  # (freq_min, freq_max) in MHz
	pulse_time: float  # Central pulse arrival time (in time bins)
	pulse_width: float  # Intrinsic pulse width (in time bins)
	snr: float  # Signal-to-noise ratio
	
	# Profile shape parameters
	n_components: int = 1  # Number of pulse components
	component_separations:  Optional[List[float]] = None  # Time between components
	component_amplitudes: Optional[List[float]] = None  # Relative amplitudes
	
	# Scattering parameters
	tau_0: float = 0.0  # Scattering timescale at reference frequency (ms)
	tau_ref_freq: float = 1000.0  # Reference frequency for scattering (MHz)
	scattering_index: float = -4.0  # Frequency scaling (alpha)
	
	# Spectral parameters
	spectral_index: float = 0.0  # Spectral index (S ∝ ν^α)
	spectral_running:  float = 0.0  # Curvature in spectrum
	
	def __post_init__(self):
		"""Initialize derived parameters"""
		if self.component_separations is None:
			self.component_separations = []
		if self.component_amplitudes is None:
			self.component_amplitudes = [1.0] * self.n_components

# ============================================================================
	# OPTIMIZED JIT-COMPILED FUNCTIONS
	# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def _apply_dispersion_optimized(base_profile, delay_bins, spectral_amps, snr):
	"""
	Fast vectorized dispersion application with Numba JIT
	
	Parameters: 
	-----------
	base_profile : np.ndarray (float32)
		Base pulse profile (1D array, length n_time)
	delay_bins : np.ndarray (int32)
		Delay in bins for each frequency (1D array, length n_freq)
	spectral_amps : np.ndarray (float32)
		Spectral amplitude for each frequency (1D array, length n_freq)
	snr : float32
		Signal-to-noise ratio
		
	Returns:
	--------
	pulse_2d : np.ndarray (float32)
		2D array (n_freq, n_time) with dispersed pulse
	"""
	n_freq = len(delay_bins)
	n_time = len(base_profile)
	pulse_2d = np.zeros((n_freq, n_time), dtype=np.float32)
	
	for f_idx in prange(n_freq):  # Parallel loop across frequencies
		delay = delay_bins[f_idx]
		amp = spectral_amps[f_idx] * snr / np.sqrt(n_freq)
		
		if delay > 0:
			# Positive delay: pulse arrives later at this frequency
			copy_len = n_time - delay
			if copy_len > 0:
				pulse_2d[f_idx, delay:] = base_profile[: copy_len] * amp
		elif delay < 0:
			# Negative delay: pulse arrives earlier (rare, but handle it)
			copy_len = n_time + delay
			if copy_len > 0:
				pulse_2d[f_idx, :copy_len] = base_profile[-delay:] * amp
		else:
			# No delay
			pulse_2d[f_idx, :] = base_profile * amp
	
	return pulse_2d


class PulseGenerator:
	"""
	Generate synthetic frequency-time radio pulse data with realistic effects
	"""
    # ========================================================================
    # PLOT METHODS
    # ========================================================================
    # These are imported from spectrum_plot_utils. py
    # 
    # How this works:
    # 1. Functions are imported at module level (top of file)
    # 2. Assigned as class attributes here
    # 3. When called on an instance (e.g., gen.plot_v1()), Python automatically
    #    passes the instance as 'self' to the function
    # 4. The function can then access self.spectrum, self.frequencies, etc.
    #
    # This keeps plotting code separate while maintaining clean method syntax
    # ========================================================================
	plot_v1 = plot_v1  # Simple plot with colorbar on right
	plot_v2 = plot_v2  # Plot with side panels, colorbar on right
	plot_v3 = plot_v3  # Plot with side panels, colorbar on left
	
	def __init__(self, n_freq: int = 256, n_time: int = 512, 
				 freq_min:  float = 1200, freq_max: float = 1600,
				 time_resolution: float = 0.064):  # ms
		"""
		Parameters:
		-----------
		n_freq : int
			Number of frequency channels
		n_time : int
			Number of time bins
		freq_min, freq_max : float
			Frequency range in MHz
		time_resolution : float
			Time resolution in milliseconds
		"""
		self.n_freq = n_freq
		self.n_time = n_time
		self.freq_min = freq_min
		self.freq_max = freq_max
		self.time_resolution = time_resolution
		
		# Create frequency array (LOW to HIGH)
		self.frequencies = np.linspace(freq_min, freq_max, n_freq)
		self.freq_channels = np.arange(n_freq)
		
		# Time array
		self.times = np.arange(n_time) * time_resolution
		
		# Initialize empty spectrum
		self.spectrum = None



	def reset(self):
		"""Reset the spectrum to zeros"""
		self.spectrum = np.zeros((self.n_freq, self.n_time))
		
	def add_noise(self, mean:  float = 0.0, std: float = 1.0):
		"""
		Add Gaussian noise to the spectrum
		
		Parameters:
		-----------
		mean : float
			Mean of noise distribution
		std : float
			Standard deviation of noise
		"""
		if self.spectrum is None:
			self.reset()
		
		noise = np.random.normal(mean, std, (self.n_freq, self.n_time))
		self.spectrum += noise
		
		return self
	
	def add_rfi(self, n_rfi: int = 0, rfi_strength: float = 10.0):
		"""
		Add random RFI (Radio Frequency Interference)
		
		Parameters:
		-----------
		n_rfi : int
			Number of RFI events
		rfi_strength : float
			Amplitude of RFI
		"""
		if self.spectrum is None:
			self.reset()
			
		for _ in range(n_rfi):
			if np.random.rand() < 0.5:
				# Narrowband RFI (single channel)
				freq_idx = np.random.randint(0, self.n_freq)
				time_start = np.random.randint(0, self.n_time - 10)
				time_end = time_start + np.random.randint(5, 20)
				self.spectrum[freq_idx, time_start:time_end] += rfi_strength
			else: 
				# Broadband RFI (multiple channels)
				freq_start = np.random.randint(0, self.n_freq - 10)
				freq_end = freq_start + np.random.randint(5, 30)
				time_idx = np.random.randint(0, self.n_time)
				self.spectrum[freq_start:freq_end, time_idx] += rfi_strength
				
		return self
	
	def _gaussian_profile(self, times: np.ndarray, center: float, width: float) -> np.ndarray:
		"""Generate a Gaussian pulse profile"""
		return np.exp(-0.5 * ((times - center) / width) ** 2)
	
	def _vonmises_profile(self, times: np.ndarray, center: float, width: float) -> np.ndarray:
		"""Generate a von Mises pulse profile (more pulsar-like)"""
		kappa = 1.0 / (width ** 2)
		phase = 2 * np.pi * (times - center) / self.n_time
		return np.exp(kappa * np.cos(phase))
	
	def _exponential_profile(self, times:  np.ndarray, center: float, width: float) -> np.ndarray:
		"""Generate an exponential pulse profile"""
		profile = np.zeros_like(times)
		mask = times >= center
		profile[mask] = np.exp(-(times[mask] - center) / width)
		return profile
	
	def calculate_dispersion_delay(self, freq: float, dm: float, 
								   ref_freq: Optional[float] = None) -> float:
		"""
		Calculate dispersion delay relative to reference frequency
		
		Parameters:
		-----------
		freq :  float
			Frequency in MHz
		dm : float
			Dispersion measure in pc/cm^3
		ref_freq : float, optional
			Reference frequency (defaults to highest frequency)
			
		Returns:
		--------
		delay : float
			Delay in milliseconds
		"""
		if ref_freq is None:
			ref_freq = self.freq_max
			
		# Dispersion constant:  4.148808 ms GHz^2 pc^-1 cm^3
		k_dm = 4.148808			# refence : https://arxiv.org/pdf/2007.02886
		delay = k_dm * dm * (1.0 / (freq/1e3)**2 - 1.0 / (ref_freq/1e3)**2)
		
		return delay
	
	def calculate_scattering_timescale(self, freq: float, tau_0: float,
									   ref_freq: float, alpha: float) -> float:
		"""
		Calculate frequency-dependent scattering timescale
		
		τ(ν) = τ_0 * (ν / ν_ref)^α
		
		Parameters:
		-----------
		freq : float
			Frequency in MHz
		tau_0 : float
			Scattering timescale at reference frequency (ms)
		ref_freq : float
			Reference frequency (MHz)
		alpha : float
			Frequency scaling index (typically -4 to -4.4)
			
		Returns: 
		--------
		tau : float
			Scattering timescale in ms
		"""
		return tau_0 * (freq / ref_freq) ** alpha
	
	def apply_scattering(self, profile: np.ndarray, tau:  float) -> np.ndarray:
		"""
		Apply exponential scattering to a pulse profile
		
		Parameters: 
		-----------
		profile : np.ndarray
			Input pulse profile
		tau : float
			Scattering timescale in ms
			
		Returns:
		--------
		scattered_profile : np.ndarray
			Scattered pulse profile
		"""
		if tau <= 0:
			return profile
			
		# Convert tau to time bins
		tau_bins = tau / self.time_resolution
		
		# Create scattering kernel (exponential)
		kernel_length = min(int(5 * tau_bins), self.n_time // 2)
		if kernel_length < 2:
			return profile
			
		kernel_times = np.arange(kernel_length) * self.time_resolution
		kernel = np.exp(-kernel_times / tau)
		kernel /= kernel.sum()  # Normalize
		
		# Convolve profile with scattering kernel
		scattered = np.convolve(profile, kernel, mode='same')
		
		return scattered
	
	def apply_scattering_batch(self, pulse_2d: np.ndarray, taus_ms: np.ndarray) -> np.ndarray:
		"""
		Apply scattering to all frequency channels at once using batch FFT
		
		Parameters:
		-----------
		pulse_2d : np.ndarray
			2D pulse array (n_freq, n_time)
		taus_ms : np.ndarray
			Scattering timescales for each frequency (n_freq,)
		
		Returns:
		--------
		scattered_2d : np.ndarray
			Scattered pulse (n_freq, n_time)
		"""
		n_freq, n_time = pulse_2d.shape
		scattered_2d = np.zeros_like(pulse_2d)
		
		# Find unique tau values (often many frequencies have similar tau)
		unique_taus = np.unique(np.round(taus_ms, 2))
		
		for tau in unique_taus:
			if tau <= 0:
				continue
			
			# Find all frequencies with this tau
			mask = np.abs(taus_ms - tau) < 0.01
			
			if not np.any(mask):
				continue
			
			# Create kernel once for this tau
			tau_bins = tau / self.time_resolution
			kernel_length = min(int(5 * tau_bins), n_time // 2)
			
			if kernel_length < 2:
				scattered_2d[mask] = pulse_2d[mask]
				continue
			
			kernel = np.zeros(n_time)
			kernel_times = np.arange(kernel_length) * self.time_resolution
			kernel[: kernel_length] = np.exp(-kernel_times / tau)
			kernel /= kernel.sum()
			
			# FFT kernel once
			kernel_fft = np.fft.rfft(kernel)
			
			# Apply to all matching frequencies
			for f_idx in np.where(mask)[0]:
				profile_fft = np.fft.rfft(pulse_2d[f_idx])
				scattered_fft = profile_fft * kernel_fft
				scattered_2d[f_idx] = np.fft.irfft(scattered_fft, n=n_time)
		
		# Copy non-scattered channels
		no_scatter_mask = taus_ms <= 0
		scattered_2d[no_scatter_mask] = pulse_2d[no_scatter_mask]
		
		return scattered_2d
	
	def calculate_spectral_amplitude(self, freq: float, spectral_index: float,
									 spectral_running: float, 
									 ref_freq: Optional[float] = None) -> float:
		"""
		Calculate frequency-dependent amplitude
		
		S(ν) = S_0 * (ν / ν_ref)^α * exp(β * log(ν / ν_ref)^2)
		
		Parameters: 
		-----------
		freq : float
			Frequency in MHz
		spectral_index : float
			Spectral index α
		spectral_running : float
			Spectral running/curvature β
		ref_freq : float, optional
			Reference frequency (defaults to center frequency)
			
		Returns: 
		--------
		amplitude :  float
			Relative amplitude at this frequency
		"""
		if ref_freq is None:
			ref_freq = (self.freq_min + self.freq_max) / 2
			
		freq_ratio = freq / ref_freq
		log_freq_ratio = np.log(freq_ratio)
		
		amplitude = (freq_ratio ** spectral_index * 
					np.exp(spectral_running * log_freq_ratio ** 2))
		
		return amplitude
	
	def calculate_required_n_time(self, dm: float, pulse_time: float, 
							  pulse_width: float, tau_0: float = 0,
							  scattering_index: float = -4.0,
							  tau_ref_freq: float = 1000.0,
							  padding: int = 100) -> int:
		"""
		Calculate required n_time for a given DM and pulse parameters
		
		Parameters:
		-----------
		dm : float
			Dispersion measure in pc/cm³
		pulse_time : float
			Pulse arrival time in bins
		pulse_width : float
			Pulse width in bins
		tau_0 :  float
			Scattering timescale at reference frequency
		scattering_index : float
			Scattering frequency index
		tau_ref_freq : float
			Reference frequency for scattering
		padding : int
			Extra bins for safety margin
			
		Returns:
		--------
		required_n_time : int
			Required number of time bins
		"""
		# Calculate dispersive delay
		max_delay_ms = self.calculate_dispersion_delay(self.freq_min, dm, ref_freq=self.freq_max)
		max_delay_bins = max_delay_ms / self.time_resolution
		
		# Calculate scattering tail
		max_tau_ms = self.calculate_scattering_timescale(
			self.freq_min, tau_0, tau_ref_freq, scattering_index
		)
		scatter_tail_bins = 5 * max_tau_ms / self.time_resolution if max_tau_ms > 0 else 0
		
		# Calculate required length
		pulse_end_bin = pulse_time + max_delay_bins + 3 * pulse_width + scatter_tail_bins
		required_n_time = int(np.ceil(pulse_end_bin)) + padding
		
		return required_n_time

	def _estimate_noise_params(self) -> dict:
		"""
		Estimate noise parameters from current spectrum
		
		Returns:
		--------
		noise_params : dict
			Dictionary with 'mean' and 'std' keys
		"""
		if self.spectrum is None or np.all(self.spectrum == 0):
			# Default noise parameters if spectrum is empty
			return {'mean': 0.0, 'std': 1.0}
		
		# Use median and MAD for robust estimation
		noise_median = np.median(self.spectrum)
		mad = np.median(np.abs(self.spectrum - noise_median))
		noise_std_robust = 1.4826 * mad
		
		# Avoid zero std
		if noise_std_robust < 1e-10:
			noise_std_robust = 1.0
		
		return {'mean': noise_median, 'std': noise_std_robust}

	def _resize_spectrum(self, new_n_freq: int = None, new_n_time: int = None, 
						 noise_params: dict = None) -> None:
		"""
		Resize spectrum array and fill new regions with noise
		
		Parameters:
		-----------
		new_n_freq : int, optional
			New number of frequency channels (if None, keep current)
		new_n_time : int, optional
			New number of time bins (if None, keep current)
		noise_params : dict, optional
			Noise parameters for new regions. If None, estimated from existing spectrum
		"""
		if new_n_freq is None:
			new_n_freq = self.n_freq
		if new_n_time is None: 
			new_n_time = self.n_time
		
		# Check if resize is needed
		if new_n_freq == self.n_freq and new_n_time == self.n_time:
			return  # No resize needed
		
		# Store old spectrum
		old_spectrum = self.spectrum.copy() if self.spectrum is not None else None
		old_n_freq = self.n_freq
		old_n_time = self.n_time
		
		# Estimate noise if not provided
		if noise_params is None:
			noise_params = self._estimate_noise_params()
		
		print(f"INFO:  Resizing spectrum array")
		print(f"  Old shape: ({old_n_freq}, {old_n_time})")
		print(f"  New shape:  ({new_n_freq}, {new_n_time})")
		print(f"  Noise params: mean={noise_params['mean']:.4f}, std={noise_params['std']:.4f}")
		
		# Update dimensions
		self.n_freq = new_n_freq
		self.n_time = new_n_time
		
		# Create new spectrum filled with noise
		self.spectrum = np.random.normal(
			noise_params['mean'],
			noise_params['std'],
			(self.n_freq, self.n_time)
		)
		
		# Copy old data into overlapping region
		if old_spectrum is not None: 
			copy_n_freq = min(old_n_freq, new_n_freq)
			copy_n_time = min(old_n_time, new_n_time)
			self.spectrum[: copy_n_freq, :copy_n_time] = old_spectrum[:copy_n_freq, :copy_n_time]
		
		# Update time array
		self.times = np.arange(self.n_time) * self.time_resolution


	def add_pulse(self, params: PulseParameters, 
				  profile_type: str = 'gaussian',
				  auto_expand_time:  bool = True,
				  auto_shrink_time: bool = True,  
				  noise_params: Optional[dict] = None) -> 'PulseGenerator':
		"""
		Add a dispersed, scattered pulse with spectral structure
		
		Parameters:
		-----------
		params : PulseParameters
			Physical parameters for the pulse
		profile_type : str
			Type of pulse profile ('gaussian', 'vonmises', 'exponential')
		auto_expand_time : bool
			If True, automatically expand time array if pulse doesn't fit
		noise_params : dict, optional
			Noise parameters to apply to new regions.
			If None, estimated from existing spectrum.
			Example: {'mean': 0.0, 'std': 1.0}
			
		Returns:
		--------
		self : PulseGenerator
			For method chaining
		"""
		if self.spectrum is None:
			self.reset()
		
		# ========================================================================
		# UPDATE FREQUENCY RANGE FROM PARAMS
		# ========================================================================
		
		param_freq_min, param_freq_max = params.freq_range
		
		# Check if frequency range has changed
		freq_range_changed = (abs(param_freq_min - self.freq_min) > 0.1 or 
							  abs(param_freq_max - self.freq_max) > 0.1)
		
		if freq_range_changed: 
			print(f"\n{'='*70}")
			print(f"INFO:  Updating frequency range from PulseParameters")
			print(f"  Old range: {self.freq_min:.1f} - {self.freq_max:.1f} MHz")
			print(f"  New range: {param_freq_min:.1f} - {param_freq_max:.1f} MHz")
			print(f"{'='*70}\n")
			
			# Estimate noise before changing anything
			if noise_params is None:
				noise_params = self._estimate_noise_params()
			
			# Update frequency parameters
			self.freq_min = param_freq_min
			self.freq_max = param_freq_max
			self.frequencies = np.linspace(self.freq_min, self.freq_max, self.n_freq)
			
			# Note: If n_freq changes in the future, we'd resize here
			# For now, frequency array is just remapped to new range
			# But we should re-add noise to be consistent
			
			# Re-generate spectrum with noise (since frequency mapping changed)
			self._resize_spectrum(new_n_freq=self.n_freq, new_n_time=self.n_time, 
								 noise_params=noise_params)
		
		# ========================================================================
		# CHECK AND AUTO-EXPAND TIME ARRAY IF NEEDED
		# ========================================================================
		
		# Calculate maximum dispersive delay
		max_delay_ms = self.calculate_dispersion_delay(self.freq_min, params.dm, 
														ref_freq=self.freq_max)
		max_delay_bins = max_delay_ms / self.time_resolution
		
		# Calculate scattering tail at lowest frequency
		max_tau_ms = self.calculate_scattering_timescale(
			self.freq_min, params.tau_0, params.tau_ref_freq, params.scattering_index
		)
		scatter_tail_bins = 5 * max_tau_ms / self.time_resolution if max_tau_ms > 0 else 0
		
		# Total extent of pulse in time
		pulse_start_bin = params.pulse_time - 3 * params.pulse_width
		pulse_end_bin = params.pulse_time + max_delay_bins + 3 * params.pulse_width + scatter_tail_bins
		
		# Check if we need more time bins
		required_n_time = int(np.ceil(pulse_end_bin)) + 100  # Add 100 bins padding
		
		time_array_needs_expansion = required_n_time > self.n_time
		
		# ========================================================================
		# HANDLE TIME ARRAY EXPANSION
		# ========================================================================
		
		if required_n_time > self.n_time:
			if auto_expand_time:
				print(f"\n{'='*70}")
				print(f"INFO: Automatically expanding time array")
				print(f"  Current n_time: {self.n_time} bins ({self.n_time * self.time_resolution:.2f} ms)")
				print(f"  Required n_time: {required_n_time} bins ({required_n_time * self.time_resolution:.2f} ms)")
				print(f"  Pulse parameters:")
				print(f"	- DM: {params.dm:.1f} pc/cm³")
				print(f"	- Dispersive delay: {max_delay_ms:.2f} ms ({max_delay_bins:.1f} bins)")
				print(f"	- Pulse time: {params.pulse_time:.1f} bins")
				print(f"	- Scattering tail: {scatter_tail_bins:.1f} bins")
				print(f"{'='*70}\n")
				
				# Estimate noise if not provided
				if noise_params is None:
					noise_params = self._estimate_noise_params()
				
				# Resize spectrum (this handles noise properly)
				self._resize_spectrum(new_n_freq=self.n_freq, new_n_time=required_n_time,
									 noise_params=noise_params)
			else:
				raise ValueError(
					f"Pulse with DM={params.dm:.1f} pc/cm³ extends beyond time array."
					f"Need {required_n_time} bins, have {self.n_time} bins."
					f"Set auto_expand_time=True to expand automatically."
				)
		
		# ========================================================================
		# HANDLE TIME ARRAY SHRINKING (NEW)
		# ========================================================================
		
		elif required_n_time < self.n_time and auto_shrink_time: 
			# Only shrink if the difference is significant (> 20%)
			shrink_threshold = 0.2
			size_ratio = required_n_time / self.n_time
			
			if size_ratio < (1 - shrink_threshold):
				print(f"\n{'='*70}")
				print(f"INFO: Automatically shrinking time array")
				print(f"  Current n_time: {self.n_time} bins ({self.n_time * self.time_resolution:.2f} ms)")
				print(f"  Required n_time: {required_n_time} bins ({required_n_time * self.time_resolution:.2f} ms)")
				print(f"  Savings: {(1 - size_ratio)*100:.1f}% reduction in size")
				print(f"  WARNING: This will regenerate noise in the array")
				print(f"  Pulse parameters:")
				print(f"	- DM: {params.dm:.1f} pc/cm³")
				print(f"	- Dispersive delay: {max_delay_ms:.2f} ms ({max_delay_bins:.1f} bins)")
				print(f"	- Pulse time: {params.pulse_time:.1f} bins")
				print(f"	- Scattering tail: {scatter_tail_bins:.1f} bins")
				print(f"{'='*70}\n")
				
				# Estimate noise if not provided
				if noise_params is None:
					noise_params = self._estimate_noise_params()
				
				# Resize spectrum (shrink)
				self._resize_spectrum(new_n_freq=self.n_freq, new_n_time=required_n_time,
									 noise_params=noise_params)
		
		# ========================================================================
		# WARNINGS
		# ========================================================================
		
		# Check pulse start (warning only)
		if pulse_start_bin < 0:
			print(f"\nWARNING: Pulse starts before time array begins!")
			print(f"  Pulse start: {pulse_start_bin:.1f} bins (< 0)")
			print(f"  Pulse time: {params.pulse_time:.1f} bins")
			print(f"  Recommendation:  Increase pulse_time to at least {3 * params.pulse_width + 50:.0f} bins\n")
		
		# Info message for tight margins
		margin_start = max(0, pulse_start_bin)
		margin_end = self.n_time - pulse_end_bin
		
		if 0 < margin_start < 50 or 0 < margin_end < 50:
			print(f"INFO: Pulse margins are tight")
			print(f"  Start margin: {margin_start:.0f} bins, End margin: {margin_end:.0f} bins")
			print(f"  Dispersive delay: {max_delay_ms:.2f} ms ({max_delay_bins:.0f} bins)\n")
		
		# ========================================================================
		# GENERATE AND ADD PULSE
		# ========================================================================
		
		# Select profile function
		profile_functions = {
			'gaussian': self._gaussian_profile,
			'vonmises': self._vonmises_profile,
			'exponential': self._exponential_profile
		}
		profile_func = profile_functions.get(profile_type, self._gaussian_profile)
		
		# Generate multi-component profile
		self.base_profile_1d = np.zeros(self.n_time)
		
		for i in range(params.n_components):
			# Component parameters
			if i == 0:
				component_time = params.pulse_time
			else:
				component_time = params.pulse_time + params.component_separations[i-1]
			
			amplitude = params.component_amplitudes[i] if i < len(params.component_amplitudes) else 1.0
			
			# Generate component profile
			component = profile_func(np.arange(self.n_time), 
									component_time, 
									params.pulse_width)
			self.base_profile_1d += amplitude * component
		
		# Normalize base profile
		if self.base_profile_1d.max() > 0:
			self.base_profile_1d /= self.base_profile_1d.max()
		
		# ========================================================================
		# CALCULATE DISPERSION PARAMETERS (VECTORIZED)
		# ========================================================================
		
		# Calculate delays for all frequencies at once
		delays_ms = self.calculate_dispersion_delay(
			self.frequencies, params.dm, ref_freq=self.freq_max
		)
		delay_bins = np.round(delays_ms / self.time_resolution).astype(np.int32)
		

		# Calculate spectral amplitudes for all frequencies
		spectral_amps = self.calculate_spectral_amplitude(
			self.frequencies, params.spectral_index, params.spectral_running
		).astype(np.float32)
		
		# ========================================================================
		# APPLY DISPERSION (OPTIMIZED JIT-COMPILED)
		# ========================================================================
		
		pulse_2d = _apply_dispersion_optimized(
			self.base_profile_1d,
			delay_bins,
			spectral_amps,
			np.float32(params.snr)
		)

		# ========================================================================
		# APPLY SCATTERING (IF NEEDED)
		# ========================================================================
		
		if params.tau_0 > 0:
			taus_ms = self.calculate_scattering_timescale(
				self.frequencies, params.tau_0, params.tau_ref_freq, 
				params.scattering_index
			)
			pulse_2d = self.apply_scattering_batch(pulse_2d, taus_ms)
		
		'''
		if params.tau_0 > 0:
			# Calculate scattering timescales for all frequencies
			taus_ms = self.calculate_scattering_timescale(
				self.frequencies, params.tau_0, params.tau_ref_freq, 
				params.scattering_index
			)
			
			# Apply scattering to each frequency channel
			for f_idx in range(self.n_freq):
				if taus_ms[f_idx] > 0:
					pulse_2d[f_idx, :] = self.apply_scattering(
						pulse_2d[f_idx, :], taus_ms[f_idx]
					)
		'''
		# ========================================================================
		# ADD TO SPECTRUM
		# ========================================================================
		#self.spectral_amps_1d = np.nanmean(pulse_2d, axis=1)
		self.pulse_2d = pulse_2d
		self.spectrum += pulse_2d
		
		return self

	
	def normalize(self, method: str = 'standard'):
		"""
		Normalize the spectrum
		
		Parameters:
		-----------
		method : str
			'standard':  zero mean, unit variance
			'minmax': scale to [0, 1]
			'robust': use median and MAD
		"""
		if self.spectrum is None:
			return self
			
		if method == 'standard':
			mean = np.mean(self.spectrum)
			std = np.std(self.spectrum)
			self.spectrum = (self.spectrum - mean) / (std + 1e-8)
			
		elif method == 'minmax':
			min_val = np.min(self.spectrum)
			max_val = np.max(self.spectrum)
			self.spectrum = (self.spectrum - min_val) / (max_val - min_val + 1e-8)
			
		elif method == 'robust':
			median = np.median(self.spectrum)
			mad = np.median(np.abs(self.spectrum - median))
			self.spectrum = (self.spectrum - median) / (1.4826 * mad + 1e-8)
			
		return self
	
	def get_spectrum(self) -> np.ndarray:
		"""Return the current spectrum"""
		return self.spectrum if self.spectrum is not None else self.reset().spectrum
	

# ============================================================================
# EXAMPLE USAGE AND DATASET GENERATION
# ============================================================================

class PulseDataset:
	"""Generate a dataset of pulse spectra with labels"""
	
	def __init__(self, generator: PulseGenerator):
		self.generator = generator
		self.data = []
		self.labels = []
		self.params = []
		
		# Parameter ranges for random generation
		self.param_ranges = {
			'dm': (10, 500),
			'pulse_time': (150, 350),
			'pulse_width': (2, 15),
			'snr': (20, 100),
			'n_components': [1, 2, 3],
			'n_components_prob': [0.7, 0.2, 0.1],
			'tau_0': (0, 5),
			'scattering_index': (-4.4, -3.5),
			'spectral_index': (-3, 0),
			'spectral_running': (-0.5, 0.5)
		}
	
	def set_param_ranges(self, **kwargs):
		"""
		Update parameter ranges for random generation
		
		Example:
		--------
		dataset.set_param_ranges(
			dm=(50, 200),
			snr=(10, 30),
			tau_0=(0, 2)
		)
		"""
		self.param_ranges.update(kwargs)
	
	def generate_random_params(self, has_pulse: bool = True) -> Optional[PulseParameters]:
		"""Generate random pulse parameters within specified ranges"""
		if not has_pulse:
			return None
		
		# Get ranges
		ranges = self.param_ranges
		
		# Random DM
		dm = np.random.uniform(*ranges['dm'])
		
		# Random pulse position
		pulse_time = np.random.uniform(*ranges['pulse_time'])
		
		# Random pulse width
		pulse_width = np.random.uniform(*ranges['pulse_width'])
		
		# Random SNR
		snr = np.random.uniform(*ranges['snr'])
		
		# Random number of components
		n_components = np.random.choice(
			ranges['n_components'], 
			p=ranges['n_components_prob']
		)
		
		# Component parameters
		if n_components > 1:
			separations = [np.random.uniform(10, 30) for _ in range(n_components - 1)]
			amplitudes = [1.0] + [np.random.uniform(0.3, 0.8) for _ in range(n_components - 1)]
		else:
			separations = []
			amplitudes = [1.0]
		
		# Scattering parameters
		tau_0 = np.random.uniform(*ranges['tau_0'])
		scattering_index = np.random.uniform(*ranges['scattering_index'])
		
		# Spectral parameters
		spectral_index = np.random.uniform(*ranges['spectral_index'])
		spectral_running = np.random.uniform(*ranges['spectral_running'])
		
		return PulseParameters(
			dm=dm,
			freq_range=(self.generator.freq_min, self.generator.freq_max),
			pulse_time=pulse_time,
			pulse_width=pulse_width,
			snr=snr,
			n_components=n_components,
			component_separations=separations,
			component_amplitudes=amplitudes,
			tau_0=tau_0,
			tau_ref_freq=1000.0,
			scattering_index=scattering_index,
			spectral_index=spectral_index,
			spectral_running=spectral_running
		)
	
	def generate_sample(self, has_pulse: bool = True, 
					   params: Optional[PulseParameters] = None,
					   profile_type: Optional[str] = None,
					   noise_mean: float = 0.0,
					   noise_std: float = 1.0) -> Tuple[np.ndarray, int, Optional[PulseParameters]]:
		"""
		Generate a single sample
		
		Parameters:
		-----------
		has_pulse : bool
			Whether to include a pulse
		params : PulseParameters, optional
			If provided, use these parameters instead of generating random ones
		profile_type : str, optional
			Type of profile ('gaussian', 'exponential', 'vonmises')
		noise_mean : float
			Mean of noise distribution
		noise_std : float
			Standard deviation of noise distribution
		
		Returns:
		--------
		spectrum : np.ndarray
			The frequency-time array
		label : int
			1 if pulse present, 0 otherwise
		params : PulseParameters or None
			Parameters used (if pulse present)
		"""
		self.generator.reset()
		self.generator.add_noise(mean=noise_mean, std=noise_std)
		
		# Prepare noise params dict for potential resizing
		noise_params = {'mean': noise_mean, 'std': noise_std}
		
		# Optionally add RFI
		if np.random.rand() < 0.1:
			self.generator.add_rfi(n_rfi=np.random.randint(1, 4), rfi_strength=5.0)
		
		# Use provided params or generate random ones
		if has_pulse:
			if params is None:
				params = self.generate_random_params(has_pulse=True)
			
			if profile_type is None:
				profile_type = np.random.choice(['gaussian', 'exponential', 'vonmises'])
			
			# Pass noise params so resize can maintain consistency
			self.generator.add_pulse(params, profile_type=profile_type, 
									noise_params=noise_params)
		
		self.generator.normalize(method='standard')
		
		return self.generator.get_spectrum(), int(has_pulse), params
		
	def generate_dataset(self, n_samples: int = None,
						pulse_fraction: float = 0.5,
						param_list: Optional[List[PulseParameters]] = None) -> Tuple[np.ndarray, np.ndarray, List]: 
		"""
		Generate a full dataset
		
		Parameters: 
		-----------
		n_samples :  int, optional
			Number of samples to generate (ignored if param_list is provided)
		pulse_fraction :  float
			Fraction of samples with pulses (ignored if param_list is provided)
		param_list :  List[PulseParameters], optional
			If provided, generate samples using these specific parameters
			
		Returns:
		--------
		data : np.ndarray
			Array of shape (n_samples, n_freq, n_time)
		labels : np.ndarray
			Array of shape (n_samples,)
		params : List
			List of PulseParameters (None for non-pulse samples)
		"""
		data = []
		labels = []
		params_list = []
		
		# Mode 1: Use provided parameter list
		if param_list is not None:
			print(f"Generating {len(param_list)} samples from parameter list...")
			
			for i, params in enumerate(param_list):
				spectrum, label, _ = self.generate_sample(
					has_pulse=True, 
					params=params
				)
				
				data.append(spectrum)
				labels.append(1)
				params_list.append(params)
				
				if (i + 1) % 100 == 0:
					print(f"Generated {i + 1}/{len(param_list)} samples")
			
			# Add noise samples
			n_noise = len(param_list)
			print(f"Generating {n_noise} noise samples...")
			for i in range(n_noise):
				spectrum, label, params = self.generate_sample(has_pulse=False)
				data.append(spectrum)
				labels.append(0)
				params_list.append(None)
				
				if (i + 1) % 100 == 0:
					print(f"Generated {i + 1}/{n_noise} noise samples")
		
		# Mode 2: Random generation
		else:
			if n_samples is None:
				raise ValueError("Either n_samples or param_list must be provided")
			
			n_with_pulse = int(n_samples * pulse_fraction)
			
			print(f"Generating {n_samples} random samples...")
			for i in range(n_samples):
				has_pulse = i < n_with_pulse
				spectrum, label, params = self.generate_sample(has_pulse=has_pulse)
				
				data.append(spectrum)
				labels.append(label)
				params_list.append(params)
				
				if (i + 1) % 100 == 0:
					print(f"Generated {i + 1}/{n_samples} samples")
		
		#return np.array(data), np.array(labels), params_list
		return data, np.array(labels), params_list


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__": 
	# Create generator
	gen = PulseGenerator(n_freq=256, n_time=512, 
						freq_min=1200, freq_max=1600,
						time_resolution=0.064)
	
	print(f"Frequency array (first 5): {gen.frequencies[:5]}")
	print(f"Frequency array (last 5): {gen.frequencies[-5:]}")
	print(f"Frequency order: LOW to HIGH")
	
	# Example 1: Simple pulse
	print("\nExample 1: Simple Gaussian pulse with dispersion")
	params1 = PulseParameters(
		dm=50.0,
		freq_range=(1200, 1600),
		pulse_time=256,
		pulse_width=5.0,
		snr=30.0
	)
	
	gen.reset()
	gen.add_noise(mean=0, std=1.0)
	gen.add_pulse(params1, profile_type='gaussian')
	gen.plot(title="Simple Pulse (DM=50) - Low to High Freq")
	
	# Example 2: Pulse with scattering
	print("\nExample 2: Pulse with strong scattering")
	params2 = PulseParameters(
		dm=100.0,
		freq_range=(1200, 1600),
		pulse_time=256,
		pulse_width=3.0,
		snr=50.0,
		tau_0=3.0,
		tau_ref_freq=1000.0,
		scattering_index=-4.0
	)
	
	gen.reset()
	gen.add_noise(mean=0, std=1.0)
	gen.add_pulse(params2, profile_type='gaussian')
	gen.plot(title="Pulse with Scattering (τ₀=3ms) - Low to High Freq")
	
	# Example 3: Multi-component pulse with spectral structure
	print("\nExample 3: Two-component pulse with spectral index")
	params3 = PulseParameters(
		dm=150.0,
		freq_range=(1200, 1600),
		pulse_time=200,
		pulse_width=4.0,
		snr=50.0,
		n_components=2,
		component_separations=[205.0],
		component_amplitudes=[1.0, 0.6],
		tau_0=1.0,
		tau_ref_freq=1000.0,
		scattering_index=-4.0,
		spectral_index=-1.5,
		spectral_running=0.2
	)
	
	gen.reset()
	gen.add_noise(mean=0, std=1.0)
	gen.add_pulse(params3, profile_type='gaussian')
	gen.plot(title="Multi-component Pulse (α=-1.5) - Low to High Freq")
	
	# Example 4: Generate a small dataset
	print("\nExample 4: Generating dataset...")
	dataset = PulseDataset(gen)
	X, y, params_list = dataset.generate_dataset(n_samples=100, pulse_fraction=0.5)
	
	#print(f"\nDataset shape: {X.shape}")
	print(f"Labels shape: {y.shape}")
	print(f"Number with pulses: {y.sum()}")
	print(f"Number without pulses:  {len(y) - y.sum()}")
	
	# Show some examples from dataset
	fig, axes = plt.subplots(2, 3, figsize=(15, 8))
	for i, ax in enumerate(axes.flat):
		idx = np.random.randint(0, len(X))
		# CHANGED: extent now shows freq_min to freq_max
		extent = [0, 512, 1200, 1600]
		ax.imshow(X[idx], aspect='auto', origin='lower', 
				 extent=extent, cmap='viridis', interpolation='nearest')
		ax.set_title(f"Sample {idx}:  {'Pulse' if y[idx] == 1 else 'Noise'}")
		ax.set_xlabel('Time (bins)')
		ax.set_ylabel('Frequency (MHz)')
	plt.tight_layout()
	plt.savefig('dataset_examples.png', dpi=150)
	plt.show()
