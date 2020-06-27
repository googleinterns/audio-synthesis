import numpy as np
import scipy.signal
    

def get_maestro_waveform_dataset(path):
    maestro = np.load(path)['arr_0']
    return maestro
    
    
def get_maestro_magnitude_phase_dataset(path, fft_length=512, frame_step=128, log_magnitude=True, instantaneous_frequency=True):
    maestro = get_maestro_waveform_dataset(path)
    
    def process_spectogram(x):
        _, _, stft = scipy.signal.stft(x, fs=16000, nfft=fft_length, nperseg=fft_length, noverlap=(fft_length - frame_step))
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        if log_magnitude:
            magnitude = np.log(magnitude)
            
        if instantaneous_frequency:
            phase = np.unwrap(phase)
            phase = np.concatenate([phase[:0].expand_dims(1), np.diff(phase)], axis=-1)
        
        return np.concatenate([magnitude.expand_dims(2), phase.expand_dims(2)], axis=-1) 
    
    maestro = np.array(map(process_spectogram, maestro))
    return maestro
    