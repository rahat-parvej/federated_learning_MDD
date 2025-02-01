from django.http import HttpResponse
from django.shortcuts import render
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Create your views here.
def predict_view(request):
    return render(request,'prediction/predict_form.html')

def pred_result(request):
    if request.method=='POST':
        # model= tf.keras.models.load_model("models/gamma_15s_50epoch_64batch.h5")
        model= tf.keras.models.load_model("models/with_loss/model.h5")
        
        uploaded_file = request.FILES['edf_file']
        temporary_path = uploaded_file.temporary_file_path()
        # # folder_path = 'uploaded_edf_files/'
        # # file_path = default_storage.save(folder_path + uploaded_file.name, ContentFile(uploaded_file.read()))
        def read_data(file_path):
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.pick_types(meg=False, eeg=True, eog=False, ecg=False) # Selecting EEG, EOG and ECG channels
            # Select a specific channel
            channel_to_keep = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1']  
            # Replace with the name of the channel you want to keep
            raw.pick_channels(channel_to_keep)
            raw.set_eeg_reference()
            raw.filter(l_freq=30,h_freq=100)#1-4=delta, 4-8=theta, 8-12=alpha, 12-30=beta, 30-100=gamma
            epochs=mne.make_fixed_length_epochs(raw,duration=15,overlap=1)
            epochs=epochs.get_data()
            scaler = StandardScaler()
            data = scaler.fit_transform(epochs.reshape(-1,epochs.shape[-1])).reshape(epochs.shape)
            return data #trials,channel,length
        
        data=read_data(temporary_path)
        # # default_storage.delete(file_path)
        data=np.moveaxis(data,1,2)
        result=model.predict(data)
        predict_value=np.mean(result)
        
        
        if predict_value<0.5:
            type="Healthy"
            predict_percent=(1 - predict_value)*100
        else:
            type="MDD"
            predict_percent=(predict_value)*100
        
        
        
        contex={'predict_value': predict_value,'type':type,'all':result,'predict_percent':predict_percent}
        return render(request,'prediction/pred_result.html',contex)