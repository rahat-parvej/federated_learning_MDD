import tempfile
import zipfile
from django.http import HttpResponse
from django.shortcuts import render

import mne
import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import os

import shutil

import time

 

# Create your views here.
def evalute_view(request):
    return render(request,'evaluate/test.html')

def evalute_result(request):
    if request.method=='POST':
        uploaded_file = request.FILES['test_data']
        # temp_dir = uploaded_file.temporary_file_path()
        # Create a temporary directory to extract the files
        temp_dir = tempfile.mkdtemp()

        # Save the uploaded zip file to the temporary directory
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(zip_path, 'wb') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the directory within the temporary directory
        extracted_folder = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        all_files_path=[]
        # Process the files within the extracted folder
        for extracted_file in os.listdir(extracted_folder):
            # Do something with each extracted file
            if extracted_file.endswith('.edf'):
                all_files_path.append(os.path.join(extracted_folder, extracted_file))
        
        healthy_file_path=[i for i in all_files_path if 'H' in i]
        patient_file_path=[i for i in all_files_path if 'MDD' in i]

        def read_data(file_path):
            raw = mne.io.read_raw_edf(file_path, preload=True)
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
        
        control_epochs_array=[read_data(subject) for subject in healthy_file_path]
        patients_epochs_array=[read_data(subject) for subject in patient_file_path]
        control_epochs_labels=[len(i)*[0] for i in control_epochs_array]
        patients_epochs_labels=[len(i)*[1] for i in patients_epochs_array]
        
        data_list=control_epochs_array+patients_epochs_array
        label_list=control_epochs_labels+patients_epochs_labels
        
        data_array=np.vstack(data_list)
        label_array=np.hstack(label_list)
        data_array=np.moveaxis(data_array,1,2)
        model_loss= tf.keras.models.load_model("models/with_loss/model.h5")
        model_sample= tf.keras.models.load_model("models/with_sample/model.h5")
        model_avg= tf.keras.models.load_model("models/with_avg/model.h5")
        model_fednova = tf.keras.models.load_model("models/with_fednova/model.h5")
        
        score_loss = model_loss.evaluate(data_array, label_array)
        
        
        score_sample = model_sample.evaluate(data_array, label_array)
        

        score_avg = model_avg.evaluate(data_array, label_array)

        score_fednova = model_fednova.evaluate(data_array, label_array)

        # Clean up - remove the uploaded zip file and the temporary directory
        shutil.rmtree(temp_dir)
        loss_score={
            'score_loss':score_loss
        }
        sample_score={
            'score_sample':score_sample
        }
        avg_score={
            'score_avg':score_avg
        }
        fednova_score={
            'score_fednova':score_fednova
        }

        return render(request, 'evaluate/test_result.html', {**loss_score, **sample_score, **avg_score, **fednova_score})
        # return render(request, 'evaluate/test_result.html', {'loss': score_loss,'sample': score_sample,'avg': score_avg})
        # return render(request, 'evaluate/test_result.html')
     
    else:
        return HttpResponse("Please upload a zip file.")
        