from data_generation import *
from dm_time_tranform_utils import *
from dm_time_plot_utils import *


# Generate a test dynamic spectrum with a pulse
gen = PulseGenerator()
#gen = PulseGenerator(n_freq=256, n_time=512, freq_min=1200, freq_max=1600)
gen.reset()
#gen.add_noise(mean=0, std=1.0)



params = PulseParameters(
	dm=122.23,					  # Dispersion measure
	freq_range=(1200, 1600),	  # Frequency range (not really used internally)
	pulse_time=1000,			   # Central time bin
	pulse_width=50.0,			  # Width in time bins
	snr=100.0,					 # Signal-to-noise ratio
	#n_components=1,			   # Number of components
	
	n_components=3,			   # Number of components
	component_separations=[1400.0, 500],
	component_amplitudes=[1.0, 0.5, 0.6],
	
	tau_0=0.0,					# Scattering at reference freq
	tau_ref_freq=1000.0,		  # Reference frequency
	scattering_index=-4.0,		# Scattering spectral index
	spectral_index=-10,		  # Flux spectral index
	spectral_running=-100.0		  # Spectral curvature
)

gen.add_noise()#.add_pulse(params).plot()

gen.add_pulse(params)

#gen.plot_v1()
#gen.plot()
#gen.plot_v2()
gen.plot_v3()

# Get dynamic spectrum
dynamic_spectrum = gen.get_spectrum()

# Apply DM-time transform
dm_transform = DMTimeTransform(
	frequencies=gen.frequencies,
	time_resolution=gen.time_resolution,
	ref_freq=gen.freq_max
)
'''
# Transform with trial DMs
dm_time_array, dm_trials = dm_transform.transform(
	dynamic_spectrum,
	dm_min=0,
	dm_max=1000,
	dm_step=0.01
)

print(f"DM-time array shape: {dm_time_array.shape}")
print(f"DM trials: {len(dm_trials)} values from {dm_trials[0]} to {dm_trials[-1]}")
'''
# Find best DM
best_dm, dm_time_plane, dm_trials = dm_transform.find_best_dm(
	dynamic_spectrum,
	dm_min=0,
	dm_max=500,
	dm_step=0.05
)
print(f"Best DM found: {best_dm:.1f} pc/cm³ (true DM: {params.dm})")

plot_dm_time_smart(dm_time_plane, dm_trials, gen.times, max_size_mb=50, method='auto')

#dm_transform.plot_dm_time_simple(params)

