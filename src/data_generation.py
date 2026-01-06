import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import matplotlib.pyplot as plt

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


class PulseGenerator:
    """
    Generate synthetic frequency-time radio pulse data with realistic effects
    """
    
    def __init__(self, n_freq: int = 256, n_time: int = 512, 
                 freq_min:  float = 1200, freq_max: float = 1600,
                 time_resolution: float = 0.064,  # ms
                 flexible_freq_range: bool = False):
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
        flexible_freq_range : bool
            If True, allows the generator to adapt to different frequency ranges
            specified in PulseParameters. If False, raises a warning when there's
            a mismatch between generator and pulse parameter frequency ranges.
        """
        self.n_freq = n_freq
        self.n_time = n_time
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_resolution = time_resolution
        self.flexible_freq_range = flexible_freq_range
        
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
    
    def _reinitialize_frequency_range(self, freq_min: float, freq_max: float):
        """
        Reinitialize the frequency array with new range.
        
        Parameters:
        -----------
        freq_min, freq_max : float
            New frequency range in MHz
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.frequencies = np.linspace(freq_min, freq_max, self.n_freq)
        # Reset spectrum if it exists since frequency range changed
        if self.spectrum is not None:
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
    
    def add_pulse(self, params: PulseParameters, 
                  profile_type: str = 'gaussian') -> 'PulseGenerator':
        """
        Add a dispersed, scattered pulse with spectral structure
        
        Parameters:
        -----------
        params : PulseParameters
            Physical parameters for the pulse
        profile_type : str
            Type of pulse profile ('gaussian', 'vonmises', 'exponential')
            
        Returns:
        --------
        self : PulseGenerator
            For method chaining
        """
        if self.spectrum is None:
            self.reset()
        
        # Check if pulse parameters frequency range matches generator's range
        param_freq_min, param_freq_max = params.freq_range
        freq_mismatch = (abs(param_freq_min - self.freq_min) > 1e-6 or 
                        abs(param_freq_max - self.freq_max) > 1e-6)
        
        if freq_mismatch:
            if self.flexible_freq_range:
                # Reinitialize with new frequency range
                import warnings
                warnings.warn(
                    f"PulseGenerator frequency range ({self.freq_min}-{self.freq_max} MHz) "
                    f"differs from PulseParameters frequency range ({param_freq_min}-{param_freq_max} MHz). "
                    f"Reinitializing generator with new range.",
                    UserWarning
                )
                self._reinitialize_frequency_range(param_freq_min, param_freq_max)
            else:
                # Raise an error if not flexible
                raise ValueError(
                    f"PulseGenerator frequency range ({self.freq_min}-{self.freq_max} MHz) "
                    f"does not match PulseParameters frequency range ({param_freq_min}-{param_freq_max} MHz). "
                    f"Either create a new PulseGenerator with matching frequency range, "
                    f"or set flexible_freq_range=True to allow automatic adaptation."
                )
        
        # Select profile function
        profile_functions = {
            'gaussian': self._gaussian_profile,
            'vonmises': self._vonmises_profile,
            'exponential': self._exponential_profile
        }
        profile_func = profile_functions.get(profile_type, self._gaussian_profile)
        
        # Generate multi-component profile
        base_profile = np.zeros(self.n_time)
        
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
            base_profile += amplitude * component
        
        # Normalize base profile
        if base_profile.max() > 0:
            base_profile /= base_profile.max()
        
        # Add pulse to each frequency channel
        for f_idx, freq in enumerate(self.frequencies):
            # Calculate dispersion delay
            delay = self.calculate_dispersion_delay(freq, params.dm)
            delay_bins = int(delay / self.time_resolution)
            shifted_profile = np.roll(base_profile, delay_bins)

            # Shift profile for dispersion
            if delay_bins >= 0:
                shifted_profile = np.roll(base_profile, delay_bins)
                shifted_profile[:delay_bins] = 0  # Zero out wrapped values
            else:
                shifted_profile = base_profile.copy()
            
            # Apply scattering
            tau = self.calculate_scattering_timescale(
                freq, params.tau_0, params.tau_ref_freq, params.scattering_index
            )
            scattered_profile = self.apply_scattering(shifted_profile, tau)
            
            # Apply spectral amplitude modulation
            spectral_amp = self.calculate_spectral_amplitude(
                freq, params.spectral_index, params.spectral_running
            )
            
            # Add to spectrum with SNR scaling
            self.spectrum[f_idx, : ] += params.snr * spectral_amp * scattered_profile
        
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
    
    def plot(self, title: str = "Pulse Spectrum", save_path: Optional[str] = None):
        """
        Visualize the frequency-time spectrum
        
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
        extent = [0, self.n_time, self.freq_min, self.freq_max]
        im = ax1.imshow(self.spectrum, aspect='auto', origin='lower',
                       extent=extent, cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Time (bins)')
        ax1.set_ylabel('Frequency (MHz)')
        ax1.set_title(title)
        plt.colorbar(im, ax=ax1, label='Intensity')
        
        # Time series (dedispersed)
        time_series = np.sum(self.spectrum, axis=0)
        ax2.plot(time_series)
        ax2.set_xlabel('Time (bins)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Integrated Time Series')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


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
            'snr': (5, 20),
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
                       profile_type: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[PulseParameters]]:
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
        self.generator.add_noise(mean=0, std=1.0)
        
        # Optionally add RFI
        if np.random.rand() < 0.1:
            self.generator.add_rfi(n_rfi=np.random.randint(1, 4), rfi_strength=5.0)
        
        # Use provided params or generate random ones
        if has_pulse:
            if params is None:
                params = self.generate_random_params(has_pulse=True)
            
            if profile_type is None:
                profile_type = np.random.choice(['gaussian', 'exponential', 'vonmises'])
            
            self.generator.add_pulse(params, profile_type=profile_type)
        
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
        
        return np.array(data), np.array(labels), params_list


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__": 
    # Create generator
    gen = PulseGenerator(n_freq=256, n_time=2048, 
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
        snr=15.0
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
        snr=20.0,
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
        snr=18.0,
        n_components=2,
        component_separations=[25.0],
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
    
    # Example 4: Flexible frequency range adaptation
    print("\nExample 4: Flexible frequency range (NEW FEATURE)")
    print("Creating generator with flexible_freq_range=True...")
    gen_flex = PulseGenerator(n_freq=256, n_time=2048,
                             freq_min=1200, freq_max=1600,
                             time_resolution=0.064,
                             flexible_freq_range=True)
    
    # Now we can add pulses at different frequency ranges!
    params_low_freq = PulseParameters(
        dm=100.0,
        freq_range=(400, 800),  # Different from generator's initial range
        pulse_time=256,
        pulse_width=5.0,
        snr=15.0
    )
    
    gen_flex.reset()
    gen_flex.add_noise(mean=0, std=1.0)
    gen_flex.add_pulse(params_low_freq, profile_type='gaussian')
    print(f"Generator adapted to frequency range: {gen_flex.freq_min}-{gen_flex.freq_max} MHz")
    gen_flex.plot(title="Pulse at 400-800 MHz (with flexible_freq_range=True)")
    
    # Example 5: Generate a small dataset
    print("\nExample 5: Generating dataset...")
    dataset = PulseDataset(gen)
    X, y, params_list = dataset.generate_dataset(n_samples=100, pulse_fraction=0.5)
    
    print(f"\nDataset shape: {X.shape}")
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
