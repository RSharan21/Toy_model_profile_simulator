import sys 
sys.path.append("/location_to_the_code/Scattering_code")

from Toy_models import Pulsar_model

import time
import importlib

def reload_class(module_name_str, class_name_str):
	'''
	Reloads module module_name and updates class_name
	Note : module_name_str, and class_name_str has to be string
	#	Usage :
			   class_A = reload_class(module_for_class_A, class_A)
	'''
	#	Reload the module without importing it explicitly
	if module_name_str in sys.modules:
		importlib.reload(sys.modules[module_name_str])

	return getattr(sys.modules[module_name_str], class_name_str)  # Fetch class B dynamically



#	for reloading Pulsar_model modul from the file Pulsar_models.py, run the following commands
Pulsar_model = reload_class('Toy_models', 'Pulsar_model')

#	Initialize the psr_model :
psr_model = Pulsar_model()

#	Setting the the spectral nature for the model
psr_model.spec_params = [-20, 3.5, -2]

#	Creating the model
psr_model.create_model()

#	ploting the frequency vs phase 2d plot
psr_model.plot_dynamic_spectra_slider_ultra() #plot_dynamic_spectra_slider_mega()


