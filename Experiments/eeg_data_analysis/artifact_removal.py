# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 07:33:30 2017

@author: MartianMartin
"""
import mne

# create ecg ephocs
average_ecg = create_ecg_epochs(raw, ch_name='0').average()
print('We found %i ECG events' % average_ecg.nave)
average_ecg.plot_joint()

# create eog ephocs
average_eog = create_eog_epochs(raw).average()
print('We found %i EOG events' % average_eog.nave)
average_eog.plot_joint()

# set eeg reference? 
raw.set_eeg_reference()

# mark channels as bad
raw.info['bads'] = ['MEG 2443']

# skip interpolation

eog_events = mne.preprocessing.find_eog_events(raw, ch_name='0')
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=raw.info['meas_date'])
                                  

raw.plot(events=eog_events, remove_dc=True)  # To see the annotated segments.

# rejection criteria for peak to peak amplitude
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

# construct ephocs
# Arbitrary epochs
reject = dict(grad=4000)
events = np.empty((n_epochs, 3), dtype=int)
first_event_sample = 100
event_id = dict(sin50hz=1)
for k in range(n_epochs):
    events[k, :] = first_event_sample + k * n_times, 0, event_id['sin50hz']

epochs = EpochsArray(data=data, info=info, events=events, event_id=event_id,
                     reject=reject)




#TODO we can construct the events ourself...
events = mne.find_events(raw, stim_channel='STI 014')
event_id = {"auditory/left": 1}
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
baseline = (None, 0)  # means from the first instant to t = 0

picks_meg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
                           
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks_meg, baseline=baseline, reject=reject,
                    reject_by_annotation=True)

epochs.drop_bad()

# More efficient ICA based artifact detection

eog_average = mne.create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                picks=picks_meg).average()

# We simplify things by setting the maximum number of components to reject
n_max_eog = 1  # here we bet on finding the vertical EOG components
eog_epochs = mne.create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = mne.ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})
                    
ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
# red -> before, black -> after. Yes! We remove quite a lot!

# to definitely register this component as a bad one to be removed
# there is the ``ica.exclude`` attribute, a simple Python list
ica.exclude.extend(eog_inds)

# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving

# uncomment this for reading and writing
# ica.save('my-ica.fif')
# ica = read_ica('my-ica.fif')

# in absence of eog channel...
#make a bipolar reference from frontal EEG sensors and use as virtual EOG channel. This can be tricky though as you can only hope that the frontal EEG channels only reflect EOG and not brain dynamics in the prefrontal cortex.

# or...
from mne.preprocessing.ica import corrmap  # noqa

# Use corrmap templating












