"""
spectrum_plot_utils.py - Visualization utilities for PulseGenerator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

	
	
def plot_v1(self, title: str = "Pulse Spectrum", save_path: Optional[str] = None):
	"""
	Visualize the following plots : 
	
	1. Frequency-time spectrum
	2. Frequency azveraged time series
	
	Parameters:
	-----------
	title : str
		Plot title
	save_path :  str, optional
		Path to save the figure
	"""
	if self.spectrum is None:
		print("No spectrum to plot")
		return
		
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	# Frequency-time plot
	# CHANGED: extent now goes from freq_min to freq_max (low to high)
	extent = [self.times[0], self.times[-1], self.freq_min, self.freq_max]
	im = ax1.imshow(self.spectrum, aspect='auto', origin='lower',
				   extent=extent, cmap='viridis', interpolation='nearest')
	ax1.set_xlabel('Time (msec)')
	ax1.set_ylabel('Frequency (MHz)')
	ax1.set_title(title)
	plt.colorbar(im, ax=ax1, label='Intensity')
	
	# Time series (dedispersed)
	time_series = np.sum(self.spectrum, axis=0)
	ax2.plot(self.times, time_series)
	ax2.set_xlabel('Time (msec)')
	ax2.set_ylabel('Intensity')
	ax2.set_title('Integrated Time Series')
	ax2.grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.show()

def plot_v2(self, title:  str = "Pulse Observation", 
		 figsize: tuple = (14, 10),
		 cmap: str = 'viridis',
		 vmin: float = None,
		 vmax: float = None) -> None:
	"""
	Plot of the following : 
	
	1. Dynamic spectrum (Flux as a function of frequency and time) 
	2. Frequency-averaged of the Dynamic spectrum (i.e. frequency averaged profile)
	3. Time-averaged of Dynamic spectrum (i.e spectral behaviour)
	
	Layout:
	- Top panel:  Frequency-averaged profile (time series)
	- Center panel: Dynamic spectrum (frequency vs time)
	- Right panel: Time-averaged spectrum (frequency plot)
	
	Parameters: 
	-----------
	title : str
		Plot title
	figsize :  tuple
		Figure size (width, height)
	cmap : str
		Colormap for dynamic spectrum
	vmin, vmax : float, optional
		Color scale limits
	"""
	import matplotlib.pyplot as plt
	from matplotlib.gridspec import GridSpec
	
	if self.spectrum is None:
		print("No spectrum to plot. Generate data first.")
		return
	
	# Create figure with GridSpec
	fig = plt.figure(figsize=figsize)
	gs = GridSpec(4, 5, figure=fig, 
			  height_ratios=[1, 3, 3, 3],  # Top panel smaller
			  width_ratios=[0.1, 3, 3, 3, 1],   # Right panel smaller
			  hspace=0.05, wspace=0.05)
	
	# ========================================================================
	# CENTER PANEL:  Dynamic spectrum
	# ========================================================================
	ax_center = fig.add_subplot(gs[1:, 1:4])  # Rows 1-3, columns 1-3
	
	# Determine color scale
	if vmin is None: 
		vmin = np.percentile(self.spectrum, 1)
	if vmax is None:
		vmax = np.percentile(self.spectrum, 99)
	
	# Plot dynamic spectrum
	extent = [self.times[0], self.times[-1], 
				  self.frequencies[0], self.frequencies[-1]]
	
	im = ax_center.imshow(self.spectrum, 
						 aspect='auto', 
						 origin='lower',
						 extent=extent,
						 cmap=cmap,
						 vmin=vmin,
						 vmax=vmax,
						 interpolation='nearest')
	
	ax_center.set_xlabel('Time (ms)', fontsize=11)
	ax_center.set_ylabel('Frequency (MHz)', fontsize=11)
	
	# Add colorbar
	#cbar = plt.colorbar(im, ax=ax_center, pad=0.01, aspect=30)
	#cbar.set_label('Intensity', fontsize=10)
	
	# ========================================================================
	# COLORBAR - LEFT SIDE (dedicated axis)
	# ========================================================================
	cax = fig.add_subplot(gs[1:, 0])  # Rows 1-3, column 0
	cbar = fig.colorbar(im, cax=cax)
	cbar.set_label('Intensity', fontsize=10, rotation=90, labelpad=15)
	
	# ========================================================================
	# TOP PANEL:  Frequency-averaged profile (time series)
	# ========================================================================
	ax_top = fig.add_subplot(gs[0, 1:4])  # Row 0, columns 1-3
	
	# Mean along frequency axis (axis=0)
	freq_avg_profile = np.mean(self.spectrum, axis=0)
	
	ax_top.plot(self.times, freq_avg_profile, 'k-', linewidth=1.5)
	ax_top.set_xlim(self.times[0], self.times[-1])
	ax_top.set_ylabel('Intensity', fontsize=10)
	ax_top.set_title(title, fontsize=12, fontweight='bold')
	ax_top.grid(True, alpha=0.3)
	ax_top.tick_params(labelbottom=False)  # Hide x-axis labels
	
	# Add statistics
	snr_estimate = (freq_avg_profile.max() - freq_avg_profile.mean()) / freq_avg_profile.std()
	ax_top.text(0.02, 0.95, f'S/N ≈ {snr_estimate:.1f}', 
			transform=ax_top.transAxes, 
			verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
			fontsize=9)
	
	
	# ========================================================================
	# RIGHT PANEL:  Time-averaged spectrum (frequency plot)
	# ========================================================================
	ax_right = fig.add_subplot(gs[1:, 4])  # Rows 1-3, column 3
	
	# Mean along time axis (axis=1)
	time_avg_spectrum = np.mean(self.spectrum, axis=1)
	model_time_avg_spectrum = np.mean(self.pulse_2d, axis=1)
	
	ax_right.plot(time_avg_spectrum, self.frequencies, 'k-', linewidth=1.5)
	ax_right.plot(model_time_avg_spectrum, self.frequencies, 'r-', linewidth=1.5)
	ax_right.set_ylim(self.frequencies[0], self.frequencies[-1])
	ax_right.set_xlabel('Intensity', fontsize=10)
	ax_right.grid(True, alpha=0.3, axis='x')
	ax_right.tick_params(labelleft=False)  # Hide y-axis labels
	
	# Add bandwidth info
	bandwidth = self.frequencies[-1] - self.frequencies[0]
	ax_right.text(0.95, 0.02, f'BW:  {bandwidth:.0f} MHz', 
			  transform=ax_right.transAxes,
			  verticalalignment='bottom',
			  horizontalalignment='right',
			  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
			  fontsize=8,
			  rotation=90)
	
	# ========================================================================
	# TOP-RIGHT CORNER: Add metadata
	# ========================================================================
	ax_corner = fig.add_subplot(gs[0, 4])  # Row 0, column 4
	ax_corner.axis('off')
	
	# Add text with spectrum info
	info_text = (
		f'Shape: {self.spectrum.shape[0]}×{self.spectrum.shape[1]}\n'
		f'Freq:  {self.freq_min:.0f}-{self.freq_max:.0f} MHz\n'
		f'Δt: {self.time_resolution:.3f} ms\n'
		f'Duration: {self.times[-1]:.1f} ms'
	)
	
	ax_corner.text(0.5, 0.5, info_text,
			   transform=ax_corner.transAxes,
			   verticalalignment='center',
			   horizontalalignment='center',
			   fontsize=8,
			   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
	
	plt.show()


def plot_v3(self, title: str = "Pulse Observation", 
		 figsize: tuple = (14, 10),
		 cmap: str = 'viridis',
		 vmin: float = None,
		 vmax: float = None,
		 show_pulse_peak: bool = True,
		 normalize_profiles: bool = False) -> None:
	"""
	Plot dynamic spectrum with frequency-averaged and time-averaged profiles
	
	Layout:
	- Top panel: Frequency-averaged profile (time series)
	- Center panel:  Dynamic spectrum (frequency vs time)
	- Right panel: Time-averaged spectrum (frequency plot)
	
	Parameters:
	-----------
	title : str
		Plot title
	figsize : tuple
		Figure size (width, height)
	cmap : str
		Colormap for dynamic spectrum
	vmin, vmax : float, optional
		Color scale limits
	show_pulse_peak : bool
		If True, mark the pulse peak location
	normalize_profiles : bool
		If True, normalize the averaged profiles to [0, 1]
	"""
	import matplotlib.pyplot as plt
	from matplotlib.gridspec import GridSpec
	
	if self.spectrum is None:
		print("No spectrum to plot. Generate data first.")
		return
	
	# Create figure with GridSpec
	fig = plt.figure(figsize=figsize)
	gs = GridSpec(4, 5, figure=fig, 
			  height_ratios=[1, 3, 3, 3],  # Top panel smaller
			  width_ratios=[0.1, 3, 3, 3, 1],   # Right panel smaller
			  hspace=0.05, wspace=0.05)
	
	# ========================================================================
	# TOP PANEL: Frequency-averaged profile (time series)
	# ========================================================================
	ax_top = fig.add_subplot(gs[0, 1:4])  # Row 0, columns 1-3
	
	# Mean along frequency axis (axis=0)
	freq_avg_profile = np.mean(self.spectrum, axis=0)
	
	if normalize_profiles:
		profile_min = freq_avg_profile.min()
		profile_max = freq_avg_profile.max()
		if profile_max > profile_min:
			freq_avg_profile = (freq_avg_profile - profile_min) / (profile_max - profile_min)
	
	ax_top.plot(self.times, freq_avg_profile, 'k-', linewidth=1.5, label='Time series')
	ax_top.fill_between(self.times, freq_avg_profile, alpha=0.3, color='blue')
	
	# Mark pulse peak
	if show_pulse_peak: 
		peak_idx = np.argmax(freq_avg_profile)
		peak_time = self.times[peak_idx]
		peak_value = freq_avg_profile[peak_idx]
		ax_top.plot(peak_time, peak_value, 'r*', markersize=15, 
			   label=f'Peak at {peak_time:.2f} ms')
		ax_top.axvline(peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
	
	ax_top.set_xlim(self.times[0], self.times[-1])
	ax_top.set_ylabel('Intensity', fontsize=10, fontweight='bold')
	ax_top.set_title(title, fontsize=12, fontweight='bold', pad=10)
	ax_top.grid(True, alpha=0.3, linestyle='--')
	ax_top.tick_params(labelbottom=False, labelsize=9)
	ax_top.legend(loc='upper right', fontsize=8)
	
	# Add statistics
	snr_estimate = (freq_avg_profile.max() - np.median(freq_avg_profile)) / np.std(freq_avg_profile)
	rms = np.sqrt(np.mean(freq_avg_profile**2))
	
	stats_text = f'S/N ≈ {snr_estimate:.1f}\nRMS = {rms:.2f}'
	ax_top.text(0.02, 0.98, stats_text, 
			transform=ax_top.transAxes, 
			verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
			fontsize=9,
			family='monospace')
	
	# ========================================================================
	# CENTER PANEL: Dynamic spectrum
	# ========================================================================
	ax_center = fig.add_subplot(gs[1:, 1:4])  # Rows 1-3, columns 1-3
	
	# Determine color scale
	if vmin is None:
		vmin = np.percentile(self.spectrum, 1)
	if vmax is None: 
		vmax = np.percentile(self.spectrum, 99)
	
	# Plot dynamic spectrum
	extent = [self.times[0], self.times[-1], 
			  self.frequencies[0], self.frequencies[-1]]
	
	im = ax_center.imshow(self.spectrum, 
					 aspect='auto', 
					 origin='lower',
					 extent=extent,
					 cmap=cmap,
					 vmin=vmin,
					 vmax=vmax,
					 #interpolation='nearest'
					 )
	
	# Mark pulse peak across frequencies
	if show_pulse_peak:
		ax_center.axvline(peak_time, color='red', linestyle='--', 
					 alpha=0.7, linewidth=1.5, label='Peak time')
	
	ax_center.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
	ax_center.set_ylabel('Frequency (MHz)', fontsize=11, fontweight='bold')
	ax_center.tick_params(labelsize=9)
	
	# Add colorbar
	#cbar = plt.colorbar(im, ax=ax_center, pad=0.01, aspect=30)
	#cbar.set_label('Intensity', fontsize=10, fontweight='bold')
	#cbar.ax.tick_params(labelsize=8)
	
	# ========================================================================
	# COLORBAR - LEFT SIDE (dedicated axis)
	# ========================================================================
	cax = fig.add_subplot(gs[1:, 0])  # Rows 1-3, column 0
	cbar = fig.colorbar(im, cax=cax)
	cbar.set_label('Intensity', fontsize=10, rotation=90, labelpad=15)
	
	# ========================================================================
	# RIGHT PANEL:  Time-averaged spectrum (frequency plot)
	# ========================================================================
	ax_right = fig.add_subplot(gs[1:, 4])  # Rows 1-3, column 3
	
	# Mean along time axis (axis=1)
	time_avg_spectrum = np.mean(self.spectrum, axis=1)
	model_time_avg_spectrum = np.mean(self.pulse_2d, axis=1)
	
	if normalize_profiles:
		spec_min = time_avg_spectrum.min()
		spec_max = time_avg_spectrum.max()
		if spec_max > spec_min:
			time_avg_spectrum = (time_avg_spectrum - spec_min) / (spec_max - spec_min)
			model_time_avg_spectrum = (model_time_avg_spectrum - spec_min) / (spec_max - spec_min)
	
	ax_right.plot(time_avg_spectrum, self.frequencies, 'k-', linewidth=1.5, label='Spectrum')
	ax_right.plot(model_time_avg_spectrum, self.frequencies, 'r-', linewidth=1.5, label='Spectrum')
	
	ax_right.fill_betweenx(self.frequencies, time_avg_spectrum, alpha=0.3, color='green')
	
	ax_right.set_ylim(self.frequencies[0], self.frequencies[-1])
	ax_right.set_xlabel('Intensity', fontsize=10, fontweight='bold')
	ax_right.grid(True, alpha=0.3, axis='x', linestyle='--')
	ax_right.tick_params(labelleft=False, labelsize=9)
	
	# Add bandwidth and center frequency info
	bandwidth = self.frequencies[-1] - self.frequencies[0]
	center_freq = (self.frequencies[0] + self.frequencies[-1]) / 2
	
	info_text = f'BW:  {bandwidth:.0f} MHz\nf_c: {center_freq:.0f} MHz'
	ax_right.text(0.95, 0.02, info_text, 
			  transform=ax_right.transAxes,
			  verticalalignment='bottom',
			  horizontalalignment='right',
			  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
			  fontsize=8,
			  rotation=90,
			  family='monospace')
	
	# ========================================================================
	# TOP-RIGHT CORNER:  Metadata
	# ========================================================================
	ax_corner = fig.add_subplot(gs[0, 4])  # Row 0, column 3
	ax_corner.axis('off')
	
	# Add text with spectrum info
	n_freq, n_time = self.spectrum.shape
	total_time = self.times[-1] - self.times[0]
	freq_resolution = (self.freq_max - self.freq_min) / n_freq if n_freq > 1 else 0
	
	info_text = (
		f'Shape\n{n_freq} × {n_time}\n'
		f'\nFreq Range\n{self.freq_min:.1f}-{self.freq_max:.1f}\n'
		f'\nΔf:  {freq_resolution:.2f} MHz\n'
		f'Δt: {self.time_resolution:.3f} ms\n'
		f'\nDuration\n{total_time:.1f} ms'
	)
	
	ax_corner.text(0.5, 0.5, info_text,
			   transform=ax_corner.transAxes,
			   verticalalignment='center',
			   horizontalalignment='center',
			   fontsize=7,
			   family='monospace',
			   bbox=dict(boxstyle='round', facecolor='lightgray', 
					alpha=0.7, edgecolor='black', linewidth=1))
	
	# Add subtle grid lines to align with main plot
	ax_top.set_xlim(ax_center.get_xlim())
	ax_right.set_ylim(ax_center.get_ylim())
	
	#plt.tight_layout()
	plt.show()

def plot_spectrum_fft(self, vmin: float = None,
		 vmax: float = None):
	'''
	Plot the fourier transform the dynamic spectrum
	'''
	#plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(self.spectrum))), aspect='auto', origin='lower')
	#plt.show()
	self.spectrum_fft = np.fft.fftshift(np.fft.fft2(self.spectrum))
	fig, axes = plt.subplots(2, 1, figsize=(12, 8))

	# Determine color scale
	if vmin is None: 
		vmin = np.percentile(self.spectrum, 1)
	if vmax is None:
		vmax = np.percentile(self.spectrum, 99)

	# Original dynamic spectrum
	im1 = axes[0].imshow(self.spectrum, aspect='auto', origin='lower',
					 extent=[0, self.n_time * self.time_resolution, 
						self.frequencies[0], self.frequencies[-1]],
						vmin=vmin,
					 vmax=vmax,
					 cmap='viridis')
	axes[0].set_xlabel('Time (ms)')
	axes[0].set_ylabel('Frequency (MHz)')
	axes[0].set_title('Dynamic Spectrum')
	plt.colorbar(im1, ax=axes[0])

	# Determine color scale
	'''
	if vmin is None: 
		vmin = np.percentile(np.abs(self.spectrum_fft), 1)
	if vmax is None:
		vmax = np.percentile(np.abs(self.spectrum_fft), 99)
	'''
	# DM-time plane
	im2 = axes[1].imshow(np.log10(np.abs(self.spectrum_fft)), aspect='auto', origin='lower',
					 #extent=[0, self.n_time * self.time_resolution,
					#	self.dm_trials[0], self.dm_trials[-1]],
					#	vmin=vmin,
					# vmax=vmax,
					 cmap='viridis')
	axes[1].set_xlabel('Fourier conjugate variable of Time (ms)')
	axes[1].set_ylabel('Fourier conjugate variable of Frequenct(MHz)')
	#axes[1].set_title(f'DM-Time Plane (Best DM: {self.best_dm:.1f})')
	#axes[1].axhline(params.dm, color='red', linestyle='--', label=f'True DM: {params.dm}')
	#axes[1].axhline(self.best_dm, color='cyan', linestyle='--', label=f'Found DM: {self.best_dm:.1f}')
	#axes[1].legend()
	plt.colorbar(im2, ax=axes[1])

	plt.tight_layout()
	plt.show()

# Convenience dictionary for easy access
PLOT_VERSIONS = {
    'v1': plot_v1,
    'v2': plot_v2,
    'v3': plot_v3,
    'simple': plot_v1,
    'colorbar_right': plot_v2,
    'colorbar_left': plot_v3,
}

def plot_spectrum(spectrum, frequencies, times, time_resolution, 
                 freq_min=None, freq_max=None, version='v3', **kwargs):
    """
    Unified interface to plot spectrum with different versions
    
    Parameters: 
    -----------
    spectrum : np. ndarray
        2D array (n_freq, n_time)
    frequencies : np.ndarray
        Frequency array in MHz
    times : np.ndarray
        Time array in ms
    time_resolution : float
        Time resolution in ms
    freq_min, freq_max : float, optional
        Frequency range
    version : str
        Plot version:  'v1', 'v2', 'v3' or 'simple', 'colorbar_right', 'colorbar_left'
    **kwargs :  dict
        Additional arguments passed to plot function
    """
    if freq_min is None:
        freq_min = frequencies[0]
    if freq_max is None: 
        freq_max = frequencies[-1]
    
    plot_func = PLOT_VERSIONS. get(version, plot_v3)
    
    # Call appropriate plot function
    if version == 'v1' or version == 'simple': 
        plot_func(spectrum, frequencies, times, time_resolution, **kwargs)
    else: 
        plot_func(spectrum, frequencies, times, time_resolution, 
                 freq_min, freq_max, **kwargs)



