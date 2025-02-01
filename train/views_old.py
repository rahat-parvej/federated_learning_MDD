from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import random
import shutil
import socket
import tempfile
import zipfile
from django.http import HttpResponse
from django.shortcuts import render
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,\
GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session

# Create your views here.
def train_view(request):
    return render(request,'train/train.html')

def training_going(request):
    if request.method=='POST':
        uploaded_file = request.FILES['test_file']
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

        # combined_list = [item for pair in zip(X, y) for item in pair]
        combined_list = [[a, b] for a, b in zip(data_list, label_list)]

        # Shuffle the combined pairs randomly
        random.shuffle(combined_list)

        data_list = [pair[0] for pair in combined_list]
        label_list = [pair[1] for pair in combined_list]
        
        data_array=np.vstack(data_list)
        label_array=np.hstack(label_list)
        data_array=np.moveaxis(data_array,1,2)
        X=data_array
        y=label_array
        
        
        def load_data(num_clients,X,y):
            data_size = len(X)
            subset_size = data_size // num_clients

            clients_X = [X[i:i + subset_size] for i in range(0, data_size, subset_size)]
            clients_y = [y[i:i + subset_size] for i in range(0, data_size, subset_size)]

            # clients_X, clients_y = np.array_split(X, num_clients), np.array_split(y, num_clients)
            return clients_X, clients_y
        
        def train_local_model(model, data_X, data_y):
            model.compile('adam', loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall', 'AUC'])
            history = model.fit(data_X,data_y,epochs=30,batch_size=25)
            return model, history, len(data_y)
        
        # Initialize global model
        pretrained_model= tf.keras.models.load_model("models/with_loss/model.h5")

        global_model = tf.keras.models.clone_model(pretrained_model)
        # global_model.compile('adam',loss='binary_crossentropy',metrics=['Accuracy', 'Precision', 'Recall','AUC'])
        

        # Number of devices
        num_devices = 3


        # Number of communication rounds
        num_communication_rounds = 1

        clients_X, clients_y=load_data(num_devices,X,y)

        total_loss=0.00
        # Federated Learning
        for round in range(num_communication_rounds):
            for client_id in range(num_devices):
                globals()['client_model%s' % client_id] = tf.keras.models.clone_model(pretrained_model)
                # globals()['client_model%s' % client_id].compile('adam',loss='binary_crossentropy',metrics=['Accuracy', 'Precision', 'Recall','AUC'])

            futures=[]
            with ThreadPoolExecutor() as executor:
                for i in range(num_devices):
                    # globals()['future%s' % i] = executor.submit(train_local_model, globals()['client_model%s' % i], clients_X[i], clients_y[i])
                    futures.append(executor.submit(train_local_model, globals()['client_model%s' % i], clients_X[i], clients_y[i]))

            trained_models=[]
            losses=[]
            # total_loss +=(1-loss)
            samples=[]
            res_sum = [np.zeros_like(w) for w in global_model.get_weights()]
            res_avg = [np.zeros_like(w) for w in global_model.get_weights()]
            for i, future in enumerate(futures):
                model, history, num_sample_i = future.result()
                np.save(f'results/d3/client_model{i}_history.npy', history)
                # total_loss +=(1-loss)
                res=[a - b for a, b in zip(model.get_weights(), global_model.get_weights())]
                res_sum=[np.add(p,q) for p, q in zip(res, res_sum)]

                trained_models.append(model)
                losses.append(np.mean(history.history['loss']))
                samples.append(np.sum(num_sample_i))
        
            for i in range(len(res_sum)):
                res_avg[i]=res_sum[i]/num_devices
            
        # Clean up - remove the uploaded zip file and the temporary directory
        shutil.rmtree(temp_dir)

        updated_weights=[a + b for a, b in zip(pretrained_model.get_weights(), res_avg)]
        pretrained_model.set_weights(updated_weights)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ip_address = socket.gethostbyname(socket.gethostname())

        # Create a PDF file to save the plots with current time in the filename
        filename = "models/updated_weights/weights"+ ip_address+"_" + current_time + ".weights.h5"
        pretrained_model.save_weights(filename, overwrite=False)

        # Save the list to a text file
        samples_file = "models/samples/sample_list.txt"
        with open(samples_file, 'a') as f:
            f.write(str(sum(samples)))
            # Add a line break
            f.write("\n")
                   
        losses_file = "models/losses/losses_list.txt"
        with open(losses_file, 'a') as f:
            loss_sum=sum(losses)
            loss=loss_sum/ len(losses)
            f.write(str(loss))
            # Add a line break
            f.write("\n")
        
        return render(request,'train/success.html',{'list':losses,'samples':samples,'res':res_avg})