import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

def gaussian(x, params):
	"""
	y=gaussian(x, params)
	params=[Amp, mean, std]
	return a Gaussian distribution with mean and standard deviation evaluated at x
	"""
	Amp_, mean_, std_ = params
	return Amp_ * np.exp(-(x-mean_)**2/2/std_**2) #/np.sqrt(2*np.pi*std_**2)

class Pulsar_model:
	'''
	Contains the parameters of pulsar: P0, P1 etc.
	
	P0 : period of pulsar 				(unit: seconds)
	P1 : period first derivative 			(unit: second/second)
	P2 : period second derivative			(unit: second**-1)
	DM : despersive measure
	Flux_f0 : Flux density at f0 (inherited from telescope_details)	(unit: Jansky)
	'''
	obs_BW = 400
	f0 = 1060
	N_chan = 512

	P0 = 0.0054
	P1 = 0
	P2 = 0
	DM = 10.5
	nbin = 512													# for now it is hard coded. Will be changed later	# P0 // fft_sampling_time

	Flux_f0 = 0.01

	
	def __init__(self):
		# self.nbin = np.max(64, int(self.P0 // self.fft_sampling_time))	# max so as to have atleast
		self.F0 = 1/self.P0
		self.F1 = -(self.P1/(self.P0**2))/2
		self.F2 = ((2*self.P1**2) - (self.P2*self.P0))/(6*self.P0**3)
		self.folded_phase_index = np.arange(self.nbin)
		self.frequency_MHz = np.linspace(self.f0, self.f0 + self.obs_BW, self.N_chan)




		'''
		params_1 = [-1, np.log10(Flux_f0)]				# st line (in log space of flux-freq)) : params_1 = [m, c]
		params_2 = [-2, 0.5, np.log10(Flux_f0)]				# parbola : params_2 = [a, b, c]	log10(S_nu) = a*log(nu)**2 + b*log(nu) + c
		'''

	def phase_mapping(self, mjd_i):
		'''
		takes the mjd_array and does a phase mapping of a pulsar parameters (P0,P1,P2...) used for folding.
		'''

		phase_ = (self.F0 * mjd_i) + (self.F1 * mjd_i**2) + (self.F2 * mjd_i**3)
		return phase_%1
	
	
	def profile_model(self, component_type=gaussian):
		'''
		Creates a 1d profile template to be used for adding spectra and scattering informations.

		Note : User can supply 3n parameters for creating n component gaussian profile with profile_model_param of the template [Amp1, mean1, std1, Amp2, mean2, std2, Amp3, mean3, std3, ...]
		'''

		if not hasattr(self, 'profile_model_param'):
			self.profile_model_param = [1, 0.5 * self.nbin, 0.05 * self.nbin]
		self.n_component = len(self.profile_model_param)//3
		self.component_func = [component_type] * self.n_component

		for j in range(self.n_component):
			if j == 0:
				self.profile_template_1d = self.component_func[j](self.folded_phase_index, [self.profile_model_param[3*j], self.profile_model_param[3*j + 1], self.profile_model_param[3*j + 2]])
			else:
				self.profile_template_1d += self.component_func[j](self.folded_phase_index, [self.profile_model_param[3*j], self.profile_model_param[3*j + 1], self.profile_model_param[3*j + 2]])

			
	
	
	
	def spectral_model(self):
		'''
		This is the simplest spectral behaviour, in the sense that all the ON bins has the same spectral nature. The pamrams can be changed as per needed
		
		params_1 = [-1, np.log10(Flux_f0)]				# st line (in log space of flux-freq)) : params_1 = [m, c]
		params_2 = [-2, 0.5, np.log10(Flux_f0)]				# parbola : params_2 = [a, b, c]	log10(S_nu) = a*log(nu)**2 + b*log(nu) + c
		'''
		if not hasattr(self, 'spec_params'):
			self.spec_params = [-1, np.log10(self.Flux_f0)]

		self.radio_frequency_spectra = 10**np.polyval(self.spec_params, np.log10(self.frequency_MHz) - np.log10(self.frequency_MHz[0]))


	def calc_tau_arr(self):
		'''
		Add the parameter tau (frequency dependent) for modelling the scattering time scale.

		scat_params = [tau_0, alpha]		#	tau(f) = tau(f0) * f**-alpha , where tau_0 = tau(f0) (in number of pulse phase bin), f is and f0 is in MHz. 
		This means that self.tau_arr is also in number of phase bin of the pulse phase bin axis.
		Note that at the current form of the function tau_0 represents the tau at 1GHz (i.e. if f0 = 1e3 tau(1GHz) = tau_0).
		For this to be in general f0 the formulae should be changed to self.tau_arr = self.scat_params[0] * ((self.frequency_MHz/f0) ** (-1 * self.scat_params[1])),
		where self.frequency_MHz and f0 both has to be in MHz (or in common units, here the convention is in MHz).
		'''
		if not hasattr(self, 'scat_params'):
			print(f'Picking up default values of scattering parameters : {[ 6.5, 4]}')
			self.scat_params = [ 6.5, 4]

		self.tau_arr = self.scat_params[0] * ((self.frequency_MHz/1e3) ** (-1 * self.scat_params[1]))		#	self.tau_arr is in unit of bins

		self.find_scat_param_range()		

	def add_tau_to_model(self):
		'''
		Once calc_tau_arr is called, tau_arra is calculated. This method concolves tau_arr with the existing model_data.
		'''
		self.calc_tau_arr()


		if hasattr(self, 'model_data'):
			if self.model_data.shape[0] == self.tau_arr.shape[0]:
				for freq_ind in range(self.model_data.shape[0]):
					#self.model_data[freq_ind] =  1/(self.tau_arr[freq_ind]* (1 - np.exp(-self.nbin/self.tau_arr[freq_ind]))) * np.convolve(self.model_data[freq_ind], np.exp(-np.arange(self.nbin) / self.tau_arr[freq_ind]))[:self.nbin]
					self.model_data[freq_ind] =  1/self.tau_arr[freq_ind] * np.convolve(self.model_data[freq_ind], np.exp(-np.arange(self.nbin) / self.tau_arr[freq_ind]))[:self.nbin]
			else:
				print(f'{self.model_data.shape[0]} != {self.tau_arr.shape[0]}')
		
	def find_scat_param_range(self):
		'''
		Find the range of scattering parameters which depends on self.nbin (as tau_arr, and self.scat_params[0] is in terms of fractional part of self.nbins) and self.frequency_MHz ranges.
		The idea is if the ranges aren't properly set, the signal get washed out or supressed easily.
		'''
		factor = 1
		if np.max(self.tau_arr) > factor * self.nbin:		
			'''
			This condition means that the largest self.tau_arr shouldn't very large.
			In other words if tau, in terms of fraction part of nbin, is > 2 the exponential will become a linear function.
			User can verify this by plotting np.exp(-np.arange(self.nbin)/self.nbin/fractional_tol))  with fractional_tol = {0.5, 0.75, 1, 2, 3, 4, 5}
			'''
			#print('Out of parameter space for good scattering parameters')
			self.out_good_scattering_range_alert = True
			self.label_color = 'red'
		else:
			self.out_good_scattering_range_alert = False
			self.label_color = 'green'		
		#	if self.nbin * np.max(self.tau_arr) > 
		self.tau_0_max_range =	self.nbin# np.max(factor * self.nbin/((self.frequency_MHz/1e3) ** (-1 * self.scat_params[1])))
		#print(fr'For frequency range $\tau_0$ should have max of {self.tau_0_max_range}')

	def add_noise(self):
		'''
		Add Noise to the model_data
		'''
		pass
		

	def create_model(self):
		'''
		This is the main function which creates the model of pulsar with spectra and scattering (frequency dependent or independent).
		The following are the steps being followed for creating a model.

			step 1 :	Create imp attributes for further computations, like phase bin index array : folded_phase_index, F0, F1, F2.
			step 2 :	Create a Gaussian (or any) 1d profile. This simplest 1d will be scaled as the spectra for a 2d profile (along frequency)
			step 3 :	Create a spectral nature of the pulsar, i.e. create the spectra for one of the ON pulse bin and apply that same spectra to other bins to a 2d model_data (frequency vs phase).
		'''
		pass

		#print('creating new profile template')
		self.profile_model()
		#print('creating new spectra model')
		self.spectral_model()

		self.model_data = self.radio_frequency_spectra.reshape(self.N_chan, 1) * self.profile_template_1d.reshape(1, self.nbin)
		if hasattr(self, 'scat_params'):
			#print('Calculate and add new tau')
			self.add_tau_to_model()
		#print('************************************************************************************************************************')

	def plot_dynamic_spectra(self):
		'''
		Plots profile (ax0), frequency vs phase plot (ax1) and spectral nature (ax2)
		'''

		fig, axs = plt.subplot_mosaic([['Profile', '.'],
							   ['Dynamic_spectra', 'spectra']],
							  figsize=(6, 6),
							  width_ratios=(4, 1), height_ratios=(1, 4))
	
		
		axs['Profile'].plot(self.profile_template_1d)
		axs['Profile'].set_xlim([0, self.nbin])

		axs['Dynamic_spectra'].imshow(self.model_data, origin='lower', aspect='auto')

		axs['spectra'].plot(self.radio_frequency_spectra, np.arange(self.N_chan))
		axs['spectra'].set_ylim([0, self.N_chan])

		plt.show()

	def plot_dynamic_spectra_slider(self):
		'''
		Plots profile (ax0), frequency vs phase plot (ax1) and spectral nature (ax2)
		'''

		fig, axs = plt.subplot_mosaic([['Profile', '.'],
							   ['Dynamic_spectra', 'spectra']],
							  figsize=(6, 6),
							  width_ratios=(4, 1), height_ratios=(1, 4))
	
		profile_ax, dyn_spec_ax, spectra_ax = [], [], []

		profiles_1d,  = axs['Profile'].plot(self.profile_template_1d)
		axs['Profile'].set_xlim([0, self.nbin])
		profile_ax.append(profiles_1d)

		dyn_spec_2d = axs['Dynamic_spectra'].imshow(self.model_data, origin='lower', aspect='auto')
		dyn_spec_ax.append(dyn_spec_2d)

		spectra_1d, = axs['spectra'].plot(self.radio_frequency_spectra, np.arange(self.N_chan))
		axs['spectra'].set_ylim([0, self.N_chan])
		spectra_ax.append(spectra_1d)


		ax_sliders_amp = plt.axes([0.2, 0.06, 0.6, 0.03])  # Position: [left, bottom, width, height]
		slider_amp = Slider(ax_sliders_amp, 'Amp', 0, 1, valinit=1)

		ax_sliders_mean = plt.axes([0.2, 0.04, 0.6, 0.03])  # Position: [left, bottom, width, height]
		slider_mean = Slider(ax_sliders_mean, 'Mean', 0, self.nbin, valinit=1)

		ax_sliders_std = plt.axes([0.2, 0.02, 0.6, 0.03])  # Position: [left, bottom, width, height]
		slider_std = Slider(ax_sliders_std, 'Std', 0.001, self.nbin, valinit=1)

		def update(val):
			self.profile_model_param = [slider_amp.val, slider_mean.val, slider_std.val]
			self.create_model()

			
			profiles_1d.set_ydata(self.profile_template_1d)
			dyn_spec_2d.set_data(self.model_data)
			spectra_1d.set_xdata(self.radio_frequency_spectra)

			fig.canvas.draw_idle()
		slider_amp.on_changed(update)
		slider_mean.on_changed(update)
		slider_std.on_changed(update)
		plt.show()

	def plot_dynamic_spectra_slider_mega(self):
		fig, axs = plt.subplot_mosaic([['Profile', '.'],
									   ['Dynamic_spectra', 'spectra'],
									   ['Convolved_profile', '.']],
									  figsize=(8, 8),
									  width_ratios=(4, 1), height_ratios=(1, 4, 1))
		
		fig.subplots_adjust(left=0.35)

		profile_plot, = axs['Profile'].plot(self.profile_template_1d)
		axs['Profile'].set_xlim([0, self.nbin])
		dyn_spec_plot = axs['Dynamic_spectra'].imshow(self.model_data, origin='lower', aspect='auto')
		spectra_plot, = axs['spectra'].plot(self.radio_frequency_spectra, np.arange(self.N_chan))
		axs['spectra'].set_ylim([0, self.N_chan])
		
		conv_f = self.model_data[0]
		P_conv_plot, = axs['Convolved_profile'].plot(conv_f)
		axs['Convolved_profile'].set_xlim([0, self.nbin])
		axs['Convolved_profile'].set_ylim([np.min(conv_f), np.max(conv_f)])
		

		slider_axes = []
		sliders = []
		
		def create_sliders():
			for ax in slider_axes:
				ax.remove()
			slider_axes.clear()
			sliders.clear()
			
			y_start = 0.9  # Start position for sliders
			spacing = 0.09  # Space between sliders
			
			for i in range(self.n_component):
				slider_axes.append(plt.axes([0.05, y_start - i * spacing, 0.2, 0.01]))  # Amp slider
				slider_axes.append(plt.axes([0.05, y_start - (i * spacing + 0.03), 0.2, 0.01]))  # Mean slider
				slider_axes.append(plt.axes([0.05, y_start - (i * spacing + 0.06), 0.2, 0.01]))  # Std slider
				
				sliders.append((
					Slider(slider_axes[-3], f'Amp {i+1}', 0, 1, valinit=self.profile_model_param[3*i]),
					Slider(slider_axes[-2], f'Mean {i+1}', 0, self.nbin, valinit=self.profile_model_param[3*i+1]),
					Slider(slider_axes[-1], f'Std {i+1}', 0.001, self.nbin, valinit=self.profile_model_param[3*i+2])
				))
			
			for s_amp, s_mean, s_std in sliders:
				s_amp.on_changed(update)
				s_mean.on_changed(update)
				s_std.on_changed(update)
		
		def update(val):
			for i, (s_amp, s_mean, s_std) in enumerate(sliders):
				self.profile_model_param[3*i] = s_amp.val
				self.profile_model_param[3*i+1] = s_mean.val
				self.profile_model_param[3*i+2] = s_std.val
			self.create_model()
			#self.add_tau_to_model()
			profile_plot.set_ydata(self.profile_template_1d)
			dyn_spec_plot.set_data(self.model_data)
			spectra_plot.set_xdata(self.radio_frequency_spectra)

			conv_f = self.model_data[0]
			P_conv_plot.set_ydata(conv_f)
			axs['Convolved_profile'].set_ylim([np.min(conv_f), np.max(conv_f)])

			fig.canvas.draw_idle()
		
		ax_add = plt.axes([0.6, 0.02, 0.15, 0.05])
		ax_remove = plt.axes([0.8, 0.02, 0.15, 0.05])
		
		def add_component(event):
			if self.n_component < 5:
				self.n_component += 1
				self.profile_model_param.extend([1, 0.5 * self.nbin, 0.05 * self.nbin])
				create_sliders()
				fig.canvas.draw_idle()
		
		def remove_component(event):
			if self.n_component > 1:
				self.n_component -= 1
				self.profile_model_param = self.profile_model_param[:-3]
				create_sliders()
				fig.canvas.draw_idle()
		
		button_add = Button(ax_add, '+ Component')
		button_remove = Button(ax_remove, '- Component')
		button_add.on_clicked(add_component)
		button_remove.on_clicked(remove_component)
		
		create_sliders()
		plt.show()

	def plot_dynamic_spectra_tau_alpha_slider(self):
		fig, axs = plt.subplot_mosaic([['Profile', '.'],
									   ['Dynamic_spectra', 'spectra'],
									   ['Convolved_profile', '.'],
									   ['Scattering_f','.']],
									  figsize=(8, 8),
									  width_ratios=(4, 1), height_ratios=(1, 4, 1, 1))
		
		fig.subplots_adjust(hspace=0, wspace=0, left=0.35)
		n_y_ticks = 10
		dy_ticks = self.N_chan // n_y_ticks
		y0, y1 = np.arange(self.N_chan)[::dy_ticks][:-1], self.frequency_MHz[::dy_ticks].astype(int)[:-1]
		if not hasattr(self, 'scat_params'):
			print('scat_params not found !')

			#self.calc_tau_arr()
			self.add_tau_to_model()

		profile_plot, = axs['Profile'].plot(self.profile_template_1d)

		axs['Profile'].set_xlim([0, self.nbin])
		dyn_spec_plot = axs['Dynamic_spectra'].imshow(self.model_data, origin='lower', aspect='auto')
		axs['Dynamic_spectra'].set_ylabel('Frequency (MHz)')
		max_pixel_plot, = axs['Dynamic_spectra'].plot(np.argmax(self.model_data, axis=1), np.arange(self.N_chan), 'r.')

		spectra_plot, = axs['spectra'].plot(self.radio_frequency_spectra, np.arange(self.N_chan), label='Model Spectra')
		max_pixel_spectra, = axs['spectra'].plot(np.max(self.model_data, axis=1), np.arange(self.N_chan), label='Peak Pixel Spectra')
		axs['spectra'].set_ylim([0, self.N_chan])
		axs['spectra'].set_xlim([0, np.max(np.array([np.max(self.model_data, axis=1), self.radio_frequency_spectra]))])
		spectra_legend = axs['spectra'].legend(prop={'size': 10}, ncol=1)
		plt.setp(spectra_legend.get_texts(), rotation=-90)


		#conv_f = 1/self.tau_arr[0] * np.exp(-np.arange(self.nbin) / self.tau_arr[0])
		#conv_f = np.convolve(self.profile_template_1d, conv_f)[:self.N_chan]
		conv_f1 = self.model_data.mean(0)
		conv_f2 = self.model_data[0]
		P_conv_plot0, = axs['Convolved_profile'].plot(conv_f1, label= 'Mean Convolved model')
		P_conv_plot1, = axs['Convolved_profile'].plot(conv_f2, label=f'Convolved model at {self.frequency_MHz[0]} MHz')
		P_conv_plot_max_f0 = axs['Convolved_profile'].scatter(np.argmax(conv_f2), np.max(conv_f2))
		axs['Convolved_profile'].legend()
		axs['Convolved_profile'].set_xlim([0, self.nbin])
		axs['Convolved_profile'].set_ylim([np.min(conv_f1), np.max(conv_f1)])

		add_label = axs['Convolved_profile'].text(0,0.8, s=fr'$\tau_{{\nu_l}}$ : {self.tau_arr[0]}', 
                                 transform=axs['Convolved_profile'].transAxes, fontsize=15, color=self.label_color)

		
		Scattering_f = 1/self.tau_arr[0] * np.exp(-np.arange(self.nbin) / self.tau_arr[0])
		Scattering_f_plot, = axs['Scattering_f'].plot(Scattering_f)
		axs['Scattering_f'].set_xlim([0, self.nbin])
		axs['Scattering_f'].set_ylim([np.min(Scattering_f), np.max(Scattering_f)])
		#		Merging the X-axis (phase bins axis)
		#axs['Dynamic_spectra'].sharex(axs['Profile'])
		#axs['Convolved_profile'].sharex(axs['Profile'])
		#axs['Scattering_f'].sharex(axs['Profile'])
		#axs['spectra'].sharey(axs['Dynamic_spectra'])


		axs['Profile'].set_title('Profile')
		axs['spectra'].set_title('Spectra')
		axs['Profile'].set_xticks([])
		axs['Profile'].set_ylabel('Normalised I')
		axs['Dynamic_spectra'].set_yticks(y0, y1)
		axs['Dynamic_spectra'].set_xticks([])
		axs['Convolved_profile'].set_xticks([])
		#axs['Scattering_f'].set_xticks([])
		axs['spectra'].set_yticks([])
		axs['spectra'].set_xlabel('Flux (Jy)')

		slider_axes = []
		sliders = []
		
		def create_sliders_profile():
			for ax in slider_axes:
				ax.remove()
			slider_axes.clear()
			sliders.clear()
			
			y_start = 0.9  # Start position for sliders
			spacing = 0.09  # Space between sliders
			
			for i in range(self.n_component):
				slider_axes.append(plt.axes([0.05, y_start - i * spacing, 0.2, 0.01]))  # Amp slider
				slider_axes.append(plt.axes([0.05, y_start - (i * spacing + 0.03), 0.2, 0.01]))  # Mean slider
				slider_axes.append(plt.axes([0.05, y_start - (i * spacing + 0.06), 0.2, 0.01]))  # Std slider
				
				sliders.append((
					Slider(slider_axes[-3], f'Amp {i+1}', 0, 1, valinit=self.profile_model_param[3*i]),
					Slider(slider_axes[-2], f'Mean {i+1}', 0, self.nbin, valinit=self.profile_model_param[3*i+1]),
					Slider(slider_axes[-1], f'Std {i+1}', 0.001, self.nbin, valinit=self.profile_model_param[3*i+2])
				))

			for s_amp, s_mean, s_std in sliders:
				s_amp.on_changed(update)
				s_mean.on_changed(update)
				s_std.on_changed(update)
		
			
		# Add sliders for scat_params (Tau and Alpha)
		ax_tau = plt.axes([0.05, 0.14, 0.2, 0.01])
		ax_alpha = plt.axes([0.05, 0.11, 0.2, 0.01])

		tau_slider = Slider(ax_tau, r'$\tau_{1GHz}$', 1e-5, self.tau_0_max_range, valinit=self.scat_params[0])			#	in units of number of phase bins (nbin)
		alpha_slider = Slider(ax_alpha, r'$\alpha$', -5, 5, valinit=self.scat_params[1])

		def update(val):
			for i, (s_amp, s_mean, s_std) in enumerate(sliders):
				self.profile_model_param[3*i] = s_amp.val
				self.profile_model_param[3*i+1] = s_mean.val
				self.profile_model_param[3*i+2] = s_std.val

			self.scat_params[0] = tau_slider.val				#
			self.scat_params[1] = alpha_slider.val				#
			self.create_model()
			#self.add_tau_to_model()
			profile_plot.set_ydata(self.profile_template_1d)
			axs['Profile'].set_ylim([np.min(self.profile_template_1d), np.max(self.profile_template_1d)])

			dyn_spec_plot.set_data(self.model_data)
			dyn_spec_plot.set_clim(vmin=np.min(self.model_data), vmax=np.max(self.model_data))
			max_pixel_plot.set_xdata(np.argmax(self.model_data, axis=1))
			spectra_plot.set_xdata(self.radio_frequency_spectra)
			max_pixel_spectra.set_xdata(np.max(self.model_data, axis=1))
			axs['spectra'].set_xlim([0, np.max(np.array([np.max(self.model_data, axis=1), self.radio_frequency_spectra]))])
			

			conv_f1 = self.model_data.mean(0)
			conv_f2 = self.model_data[0]
			P_conv_plot0.set_ydata(conv_f1)
			P_conv_plot1.set_ydata(conv_f2)
			P_conv_plot_max_f0.set_offsets(np.c_[np.argmax(conv_f2), np.max(conv_f2)])
			axs['Convolved_profile'].set_ylim([np.min(np.array([conv_f1, conv_f2])), np.max(np.array([conv_f1, conv_f2]))])
			add_label.set_text(fr'$\tau_{{\nu_l}}$ : {self.tau_arr[0]}')
			add_label.set_color(self.label_color)

			Scattering_f = 1/self.tau_arr[0] * np.exp(-np.arange(self.nbin) / self.tau_arr[0])
			Scattering_f_plot.set_ydata(Scattering_f)
			axs['Scattering_f'].set_ylim([np.min(Scattering_f), np.max(Scattering_f)])

			fig.canvas.draw_idle()
		

		tau_slider.on_changed(update)
		alpha_slider.on_changed(update)
		
		ax_add = plt.axes([0.6, 0.02, 0.15, 0.05])
		ax_remove = plt.axes([0.8, 0.02, 0.15, 0.05])
		
		def add_component(event):
			if self.n_component < 5:
				self.n_component += 1
				self.profile_model_param.extend([1, 0.5 * self.nbin, 0.05 * self.nbin])
				create_sliders_profile()
				fig.canvas.draw_idle()
		
		def remove_component(event):
			if self.n_component > 1:
				self.n_component -= 1
				self.profile_model_param = self.profile_model_param[:-3]
				create_sliders_profile()
				fig.canvas.draw_idle()
		
		button_add = Button(ax_add, '+ Component')
		button_remove = Button(ax_remove, '- Component')
		button_add.on_clicked(add_component)
		button_remove.on_clicked(remove_component)
		
		create_sliders_profile()
		plt.show()

	"""
	def plot_dynamic_spectra_slider_ultra(self):
		fig, axs = plt.subplot_mosaic([['Profile', '.'],
									   ['Dynamic_spectra', 'spectra']],
									  figsize=(10, 8),
									  width_ratios=(4, 1), height_ratios=(1, 4))
		fig.subplots_adjust(left=0.3)  # Adjust left margin to make space for sliders
		
		profile_plot, = axs['Profile'].plot(self.profile_template_1d)
		axs['Profile'].set_xlim([0, self.nbin])
		dyn_spec_plot = axs['Dynamic_spectra'].imshow(self.model_data, origin='lower', aspect='auto')
		spectra_plot, = axs['spectra'].plot(self.radio_frequency_spectra, np.arange(self.N_chan))
		axs['spectra'].set_ylim([0, self.N_chan])
		
		slider_axes = []
		sliders = []
		
		def create_sliders():
			for ax in slider_axes:
				ax.remove()
			slider_axes.clear()
			sliders.clear()
			
			x_start = 0.05  # Move sliders to the left side
			spacing_x = 0.05  # Space between vertical slider sets
			
			for i in range(self.n_component):
				slider_axes.append(plt.axes([x_start, 0.6 - i * spacing_x, 0.03, 0.3]))  # Amp slider
				slider_axes.append(plt.axes([x_start + 0.05, 0.6 - i * spacing_x, 0.03, 0.3]))  # Mean slider
				slider_axes.append(plt.axes([x_start + 0.1, 0.6 - i * spacing_x, 0.03, 0.3]))  # Std slider
				
				sliders.append((
					Slider(slider_axes[-3], f'Amp {i+1}', 0, 1, valinit=self.profile_model_param[3*i], orientation='vertical'),
					Slider(slider_axes[-2], f'Mean {i+1}', 0, self.nbin, valinit=self.profile_model_param[3*i+1], orientation='vertical'),
					Slider(slider_axes[-1], f'Std {i+1}', 0.001, self.nbin, valinit=self.profile_model_param[3*i+2], orientation='vertical')
				))
			
			ax_scat = plt.axes([0.2, 0.15, 0.03, 0.3])
			ax_spec = plt.axes([0.25, 0.15, 0.03, 0.3])
			
			if not hasattr(self, 'scat_params'):
				self.calc_tau_arr()
			scat_slider = Slider(ax_scat, 'Scat', 0, 2, valinit=self.scat_params[0], orientation='vertical')
			spec_slider = Slider(ax_spec, 'Spec', -2, 2, valinit=self.spec_params[0], orientation='vertical')
			
		def update_scat_spec(val):
			self.scat_params[0] = scat_slider.val
			self.spec_params[0] = spec_slider.val
			self.create_model()
			dyn_spec_plot.set_data(self.model_data)
			spectra_plot.set_xdata(self.radio_frequency_spectra)
			fig.canvas.draw_idle()
			
			scat_slider.on_changed(update_scat_spec)
			spec_slider.on_changed(update_scat_spec)
		
		ax_add_spec = plt.axes([0.05, 0.02, 0.1, 0.05])
		ax_remove_spec = plt.axes([0.17, 0.02, 0.1, 0.05])
		
		def add_spec_param(event):
			if len(self.spec_params) < 5:
				self.spec_params.append(0)
				create_sliders()
				fig.canvas.draw_idle()
		
		def remove_spec_param(event):
			if len(self.spec_params) > 1:
				self.spec_params.pop()
				create_sliders()
				fig.canvas.draw_idle()
		
		button_add_spec = Button(ax_add_spec, '+ Spec')
		button_remove_spec = Button(ax_remove_spec, '- Spec')
		button_add_spec.on_clicked(add_spec_param)
		button_remove_spec.on_clicked(remove_spec_param)
		
		create_sliders()
		plt.show()

	"""














