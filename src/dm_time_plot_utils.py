import matplotlib.pyplot as plt
import numpy as np

def plot_dm_time_decimated(dm_time_array, dm_trials, times, 
                          max_dm_bins=500, max_time_bins=2000,
                          title="DM-Time Transform", 
                          figsize=(14, 8), cmap='viridis'):
    """
    Plot DM-time array with automatic decimation for large arrays
    
    Parameters: 
    -----------
    dm_time_array : np.ndarray
        2D array (n_dm_trials, n_time)
    dm_trials : np.ndarray
        DM values
    times : np.ndarray
        Time values in ms
    max_dm_bins :  int
        Maximum DM bins to plot
    max_time_bins : int
        Maximum time bins to plot
    """
    n_dm, n_time = dm_time_array.shape
    
    # Calculate decimation factors
    dm_decimate = max(1, n_dm // max_dm_bins)
    time_decimate = max(1, n_time // max_time_bins)
    
    print(f"Original shape: {dm_time_array.shape}")
    print(f"Decimation:  DM every {dm_decimate}, Time every {time_decimate}")
    
    # Decimate using slicing (creates a view, not a copy - very fast!)
    dm_plot = dm_time_array[:: dm_decimate, ::time_decimate]
    dm_trials_plot = dm_trials[:: dm_decimate]
    times_plot = times[::time_decimate]
    
    print(f"Plotting shape: {dm_plot.shape}")
    print(f"Memory reduction: {dm_time_array.nbytes / (1024**2):.1f} MB → {dm_plot.nbytes / (1024**2):.1f} MB")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [times_plot[0], times_plot[-1], 
              dm_trials_plot[0], dm_trials_plot[-1]]
    
    im = ax.imshow(dm_plot, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('DM (pc/cm³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S/N', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return dm_plot
    
def plot_dm_time_adaptive(dm_time_array, dm_trials, times,
                         max_dm_bins=500, max_time_bins=2000,
                         peak_dm_window=100, peak_time_window=500,
                         title="DM-Time Transform (Adaptive)",
                         figsize=(14, 8), cmap='viridis'):
    """
    Plot DM-time with adaptive resolution - high res around peak, low res elsewhere
    
    Parameters:
    -----------
    dm_time_array : np.ndarray
        2D array (n_dm_trials, n_time)
    dm_trials : np.ndarray
        DM values
    times : np.ndarray
        Time values
    max_dm_bins : int
        Max DM bins for full array
    max_time_bins :  int
        Max time bins for full array
    peak_dm_window : int
        Number of DM bins around peak to keep at full resolution
    peak_time_window : int
        Number of time bins around peak to keep at full resolution
    """
    n_dm, n_time = dm_time_array.shape
    
    # Find peak location
    peak_idx = np.unravel_index(np.argmax(dm_time_array), dm_time_array.shape)
    peak_dm_idx, peak_time_idx = peak_idx
    peak_dm = dm_trials[peak_dm_idx]
    peak_time = times[peak_time_idx]
    
    print(f"Peak found at DM={peak_dm:.1f} pc/cm³, Time={peak_time:.2f} ms")
    
    # Define high-resolution region around peak
    dm_start = max(0, peak_dm_idx - peak_dm_window // 2)
    dm_end = min(n_dm, peak_dm_idx + peak_dm_window // 2)
    time_start = max(0, peak_time_idx - peak_time_window // 2)
    time_end = min(n_time, peak_time_idx + peak_time_window // 2)
    
    # Extract high-res region
    dm_plot_highres = dm_time_array[dm_start:dm_end, time_start:time_end]
    
    # Decimate if still too large
    dm_decimate = max(1, dm_plot_highres.shape[0] // max_dm_bins)
    time_decimate = max(1, dm_plot_highres.shape[1] // max_time_bins)
    
    dm_plot = dm_plot_highres[::dm_decimate, ::time_decimate]
    dm_trials_plot = dm_trials[dm_start:dm_end: dm_decimate]
    times_plot = times[time_start:time_end: time_decimate]
    
    print(f"Plotting region: DM [{dm_trials_plot[0]:.1f}, {dm_trials_plot[-1]:.1f}], "
          f"Time [{times_plot[0]:.2f}, {times_plot[-1]:.2f}] ms")
    print(f"Shape: {dm_plot.shape}")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [times_plot[0], times_plot[-1], 
              dm_trials_plot[0], dm_trials_plot[-1]]
    
    im = ax.imshow(dm_plot, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, interpolation='bilinear')
    
    # Mark peak
    ax.plot(peak_time, peak_dm, 'r*', markersize=20, 
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'Peak:  DM={peak_dm:.1f}')
    ax.legend(loc='upper right')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('DM (pc/cm³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S/N', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return dm_plot


from scipy.ndimage import zoom

def plot_dm_time_binned(dm_time_array, dm_trials, times,
                       target_shape=(500, 2000),
                       title="DM-Time Transform (Binned)",
                       figsize=(14, 8), cmap='viridis'):
    """
    Plot DM-time array with binning (averaging) for downsampling
    
    This preserves S/N better than simple decimation
    
    Parameters:
    -----------
    dm_time_array : np.ndarray
        2D array (n_dm_trials, n_time)
    dm_trials : np.ndarray
        DM values
    times : np.ndarray
        Time values
    target_shape : tuple
        Desired (n_dm, n_time) for plotting
    """
    n_dm, n_time = dm_time_array.shape
    target_dm, target_time = target_shape
    
    print(f"Original shape: {dm_time_array.shape}")
    
    # Calculate zoom factors
    zoom_factors = (target_dm / n_dm, target_time / n_time)
    
    # Only downsample if necessary
    if zoom_factors[0] >= 1 and zoom_factors[1] >= 1:
        dm_plot = dm_time_array
        dm_trials_plot = dm_trials
        times_plot = times
        print("No downsampling needed")
    else:
        # Downsample using area averaging (order=1 is bilinear)
        print(f"Downsampling with zoom factors: {zoom_factors}")
        dm_plot = zoom(dm_time_array, zoom_factors, order=1)
        
        # Resample axes
        dm_trials_plot = np.linspace(dm_trials[0], dm_trials[-1], dm_plot.shape[0])
        times_plot = np.linspace(times[0], times[-1], dm_plot.shape[1])
    
    print(f"Plotting shape: {dm_plot.shape}")
    print(f"Memory:  {dm_plot.nbytes / (1024**2):.1f} MB")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [times_plot[0], times_plot[-1], 
              dm_trials_plot[0], dm_trials_plot[-1]]
    
    im = ax.imshow(dm_plot, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, interpolation='bilinear')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('DM (pc/cm³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S/N', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return dm_plot

def max_pool_2d(array, pool_size):
    """
    Apply max pooling to 2D array
    
    Parameters:
    -----------
    array : np.ndarray
        Input 2D array
    pool_size : tuple
        (pool_dm, pool_time) - size of pooling window
    
    Returns:
    --------
    pooled : np.ndarray
        Max-pooled array
    """
    from skimage.measure import block_reduce
    
    # Use max function for pooling
    pooled = block_reduce(array, block_size=pool_size, func=np.max)
    
    return pooled


def plot_dm_time_maxpool(dm_time_array, dm_trials, times,
                        max_dm_bins=500, max_time_bins=2000,
                        title="DM-Time Transform (Max Pooling)",
                        figsize=(14, 8), cmap='viridis'):
    """
    Plot DM-time array using max pooling (preserves peaks)
    
    Parameters:
    -----------
    dm_time_array : np.ndarray
        2D array (n_dm_trials, n_time)
    dm_trials : np.ndarray
        DM values
    times : np.ndarray
        Time values
    max_dm_bins : int
        Target number of DM bins
    max_time_bins : int
        Target number of time bins
    """
    n_dm, n_time = dm_time_array.shape
    
    # Calculate pool size
    pool_dm = max(1, n_dm // max_dm_bins)
    pool_time = max(1, n_time // max_time_bins)
    
    print(f"Original shape: {dm_time_array.shape}")
    print(f"Pool size: ({pool_dm}, {pool_time})")
    
    # Apply max pooling
    try:
        from skimage.measure import block_reduce
        dm_plot = block_reduce(dm_time_array, 
                              block_size=(pool_dm, pool_time), 
                              func=np.max)
    except ImportError:
        # Fallback:  manual max pooling
        dm_plot = dm_time_array[::pool_dm, ::pool_time]
        print("scikit-image not available, using decimation instead")
    
    # Resample axes
    dm_trials_plot = np.linspace(dm_trials[0], dm_trials[-1], dm_plot.shape[0])
    times_plot = np.linspace(times[0], times[-1], dm_plot.shape[1])
    
    print(f"Plotting shape: {dm_plot.shape}")
    print(f"Memory: {dm_plot.nbytes / (1024**2):.1f} MB")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [times_plot[0], times_plot[-1], 
              dm_trials_plot[0], dm_trials_plot[-1]]
    
    im = ax.imshow(dm_plot, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('DM (pc/cm³)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S/N', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return dm_plot

def plot_dm_time_smart(dm_time_array, dm_trials, times,
                      max_size_mb=10,
                      method='auto',
                      title="DM-Time Transform",
                      figsize=(14, 8), cmap='viridis',
                      save_path=None):
    """
    Smart DM-time plotting with automatic method selection
    
    Parameters: 
    -----------
    dm_time_array : np.ndarray
        2D array (n_dm_trials, n_time)
    dm_trials : np.ndarray
        DM values
    times : np.ndarray
        Time values in ms
    max_size_mb : float
        Maximum array size to plot (in MB)
    method : str
        'auto', 'decimate', 'adaptive', 'binned', 'maxpool'
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    save_path : str, optional
        Path to save figure
    """

    
    # Calculate current size
    size_mb = dm_time_array.nbytes / (1024**2)
    n_dm, n_time = dm_time_array.shape
    
    print(f"DM-Time Array Info:")
    print(f"  Shape: {dm_time_array.shape}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  DM range: {dm_trials[0]:.1f} - {dm_trials[-1]:.1f} pc/cm³")
    print(f"  Time range: {times[0]:.2f} - {times[-1]:.2f} ms")
    
    # Determine if downsampling is needed
    if size_mb <= max_size_mb:
        print(f"  ✓ Size OK, plotting full resolution\n")
        dm_plot = dm_time_array
        dm_trials_plot = dm_trials
        times_plot = times
    else:
        # Calculate target shape
        reduction_factor = np.sqrt(max_size_mb / size_mb)
        target_dm = int(n_dm * reduction_factor)
        target_time = int(n_time * reduction_factor)
        
        print(f"  ⚠ Size too large, downsampling needed")
        print(f"  Target shape: ({target_dm}, {target_time})")
        
        # Auto-select method
        if method == 'auto':
            # Use adaptive if there's a clear peak
            if dm_time_array.max() > 5 * dm_time_array.mean():
                method = 'adaptive'
            else:
                method = 'decimate'
        
        print(f"  Method: {method}\n")
        
        # Apply selected method
        if method == 'decimate':
            dm_dec = max(1, n_dm // target_dm)
            time_dec = max(1, n_time // target_time)
            dm_plot = dm_time_array[::dm_dec, ::time_dec]
            dm_trials_plot = dm_trials[:: dm_dec]
            times_plot = times[::time_dec]
            
        elif method == 'adaptive': 
            # Find peak
            peak_idx = np.unravel_index(np.argmax(dm_time_array), dm_time_array.shape)
            peak_dm_idx, peak_time_idx = peak_idx
            
            # Extract region around peak
            dm_window = min(target_dm * 2, n_dm)
            time_window = min(target_time * 2, n_time)
            
            dm_start = max(0, peak_dm_idx - dm_window // 2)
            dm_end = min(n_dm, dm_start + dm_window)
            time_start = max(0, peak_time_idx - time_window // 2)
            time_end = min(n_time, time_start + time_window)
            
            # Decimate region
            region = dm_time_array[dm_start:dm_end, time_start:time_end]
            dm_dec = max(1, region.shape[0] // target_dm)
            time_dec = max(1, region.shape[1] // target_time)
            
            dm_plot = region[:: dm_dec, ::time_dec]
            dm_trials_plot = dm_trials[dm_start:dm_end:dm_dec]
            times_plot = times[time_start:time_end:time_dec]
            
        elif method == 'binned':
            from scipy.ndimage import zoom
            zoom_factors = (target_dm / n_dm, target_time / n_time)
            dm_plot = zoom(dm_time_array, zoom_factors, order=1)
            dm_trials_plot = np.linspace(dm_trials[0], dm_trials[-1], target_dm)
            times_plot = np.linspace(times[0], times[-1], target_time)
            
        elif method == 'maxpool':
            pool_dm = max(1, n_dm // target_dm)
            pool_time = max(1, n_time // target_time)
            
            try:
                from skimage.measure import block_reduce
                dm_plot = block_reduce(dm_time_array, 
                                      block_size=(pool_dm, pool_time),
                                      func=np.max)
            except ImportError: 
                dm_plot = dm_time_array[::pool_dm, ::pool_time]
            
            dm_trials_plot = np.linspace(dm_trials[0], dm_trials[-1], dm_plot.shape[0])
            times_plot = np.linspace(times[0], times[-1], dm_plot.shape[1])
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    print(f"Final plotting shape: {dm_plot.shape}")
    print(f"Final size: {dm_plot.nbytes / (1024**2):.1f} MB\n")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [times_plot[0], times_plot[-1], 
              dm_trials_plot[0], dm_trials_plot[-1]]
    
    im = ax.imshow(dm_plot, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, interpolation='bilinear')
    
    # Find and mark peak
    peak_idx = np.unravel_index(np.argmax(dm_plot), dm_plot.shape)
    peak_dm = dm_trials_plot[peak_idx[0]]
    peak_time = times_plot[peak_idx[1]]
    peak_snr = dm_plot[peak_idx]
    
    ax.plot(peak_time, peak_dm, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)
    ax.axhline(peak_dm, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('DM (pc/cm³)', fontsize=12)
    ax.set_title(f"{title}\nPeak: DM={peak_dm:.1f} pc/cm³, t={peak_time:.2f} ms, S/N={peak_snr:.1f}", 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S/N', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return dm_plot, dm_trials_plot, times_plot

def plot_dm_time_simple(self, params,
		 vmin: float = None,
		 vmax: float = None):
	
	
	# Plot DM-time plane
	import matplotlib.pyplot as plt

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
						 cmap='viridis',
						 #interpolation='nearest'
						 )
	axes[0].set_xlabel('Time (ms)')
	axes[0].set_ylabel('Frequency (MHz)')
	axes[0].set_title('Dynamic Spectrum')
	plt.colorbar(im1, ax=axes[0])

	if self.dm_time_array.nbytes/(8 * 1024**2) > 100:
		dm_time_array = self.dm_time_array.copy()[::20, ::20]
	else:
		dm_time_array = self.dm_time_array.copy()
	# Determine color scale
	if vmin is None: 
		vmin = np.percentile(dm_time_array, 1)
	if vmax is None:
		vmax = np.percentile(dm_time_array, 99)
		
	# DM-time plane
	im2 = axes[1].imshow(dm_time_array, aspect='auto', origin='lower',
						 extent=[0, self.n_time * self.time_resolution,
								self.dm_trials[0], self.dm_trials[-1]],
								vmin=vmin,
						 vmax=vmax,
						 cmap='viridis')
	axes[1].set_xlabel('Time (ms)')
	axes[1].set_ylabel('DM (pc/cm³)')
	axes[1].set_title(f'DM-Time Plane (Best DM: {self.best_dm:.1f})')
	axes[1].axhline(params.dm, color='red', linestyle='--', label=f'True DM: {params.dm}')
	axes[1].axhline(self.best_dm, color='cyan', linestyle='--', label=f'Found DM: {self.best_dm:.1f}')
	axes[1].legend()
	plt.colorbar(im2, ax=axes[1])

	plt.tight_layout()
	plt.show()

