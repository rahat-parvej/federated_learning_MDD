from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import pickle
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
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',mode="min", patience=3)
        def train_local_model(model, data_X, data_y):
            model.compile('adam', loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall', 'AUC'])
            model.fit(data_X, data_y,epochs=21,batch_size=25)
            # history = model.fit(data_X,data_y,epochs=30,batch_size=25)
            return model, model.history, len(data_y)
        
        # Initialize global model
        
        global_model_loss = tf.keras.models.load_model("models/with_loss/model.h5")
        # global_model_loss.compile('adam',loss='binary_crossentropy',metrics=['Accuracy', 'Precision', 'Recall','AUC'])

        pretrained_model_sample= tf.keras.models.load_model("models/with_sample/model.h5")
        global_model_sample = tf.keras.models.load_model("models/with_sample/model.h5")
        # global_model_sample.compile('adam',loss='binary_crossentropy',metrics=['Accuracy', 'Precision', 'Recall','AUC'])

        pretrained_model_avg= tf.keras.models.load_model("models/with_avg/model.h5")
        global_model_avg = tf.keras.models.load_model("models/with_avg/model.h5")
               

        # Number of devices
        num_devices = 3


        # Number of communication rounds
        num_communication_rounds = 1

        clients_X, clients_y=load_data(num_devices,X,y)

        total_loss=0.00
        # Federated Learning
        for round in range(num_communication_rounds):
            for client_id in range(num_devices):
                globals()['client_model_loss%s' % client_id] = tf.keras.models.load_model("models/with_loss/model.h5")
                globals()['client_model_sample%s' % client_id] = tf.keras.models.load_model("models/with_sample/model.h5")
                globals()['client_model_avg%s' % client_id] = tf.keras.models.load_model("models/with_avg/model.h5")
                
            futures_loss=[]
            futures_sample=[]
            futures_avg=[]

            for i in range(num_devices):
                 futures_loss.append((train_local_model(globals()['client_model_loss%s' % i], X, y)))
            # with ThreadPoolExecutor() as executor:
            #     for i in range(num_devices):
            #         # globals()['future%s' % i] = executor.submit(train_local_model, globals()['client_model%s' % i], clients_X[i], clients_y[i])
            #         futures_loss.append(executor.submit(train_local_model, globals()['client_model_loss%s' % i], clients_X[i], clients_y[i]))
                    
            for i in range(num_devices):
                futures_sample.append((train_local_model(globals()['client_model_avg%s' % i], X, y)))
            # with ThreadPoolExecutor() as executor:
            #     for i in range(num_devices):
            #         futures_sample.append(executor.submit(train_local_model, globals()['client_model_sample%s' % i], clients_X[i], clients_y[i]))
            for i in range(num_devices):
                futures_avg.append((train_local_model(globals()['client_model_avg%s' % i], X, y)))
            # with ThreadPoolExecutor() as executor:
            #     for i in range(num_devices):
            #         futures_avg.append(executor.submit(train_local_model, globals()['client_model_avg%s' % i], clients_X[i], clients_y[i]))

            trained_models_loss=[]
            trained_models_sample=[]
            trained_models_avg=[]
            losses=[]
            total_loss=0
            total_sample=0
            samples=[]
            avg_global_weights = [tf.zeros_like(w) for w in global_model_loss.get_weights()]
            

            for i, future in enumerate(futures_loss):
                model, history, num_sample_i = future
                # model, history, num_sample_i = future.result()
                # np.save(f'models/results/d2/client_model_loss{i}_history.npy', history.history)
                trained_models_loss.append(model)
                losses.append(history.history['loss'][-1])
                
            
            for i, future in enumerate(futures_sample):
                model, history, num_sample_i = future
                # np.save(f'models/results/d2/client_model_sample{i}_history.npy', history.history)
                trained_models_sample.append(model)
                samples.append(np.sum(num_sample_i))

            for i, future in enumerate(futures_avg):
                model, history, num_sample_i = future
                # np.save(f'models/results/d2/client_model_avg{i}_history.npy', history.history)
                trained_models_avg.append(model)
        
            for i in range(len(losses)):
                total_loss +=(1-losses[i])
                weighted_local_weights_for_loss = [(1-losses[i]) * w for w in trained_models_loss[i].get_weights()]

                total_sample+=samples[i]
                weighted_local_weights= [samples[i] * w for w in trained_models_sample[i].get_weights()]

                avg_global_weights = [tf.add(agw, lw) for agw, lw in zip(avg_global_weights, trained_models_avg[i].get_weights())]
  
        # Clean up - remove the uploaded zip file and the temporary directory
        shutil.rmtree(temp_dir)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ip_address = socket.gethostbyname(socket.gethostname())
        #weight save
        # for loss
        updated_weights_loss=[tf.divide(gws_l, total_loss) for gws_l in weighted_local_weights_for_loss]
        global_model_loss.set_weights(updated_weights_loss)
        
        filename = "models/updated_weights_loss/weights"+ ip_address+"_" + current_time + ".weights.h5"
        global_model_loss.save_weights(filename, overwrite=False)

        #for sample
        updated_weights_sample=[tf.divide(gws_l, total_sample) for gws_l in weighted_local_weights]
        global_model_sample.set_weights(updated_weights_sample)
       
        filename = "models/updated_weights_sample/weights"+ ip_address+"_" + current_time + ".weights.h5"
        global_model_sample.save_weights(filename, overwrite=False)

        #for avg
        updated_weights_avg=[tf.divide(gws_l, num_devices) for gws_l in weighted_local_weights]
        global_model_avg.set_weights(updated_weights_avg)
        
        filename = "models/updated_weights_avg/weights"+ ip_address+"_" + current_time + ".weights.h5"
        global_model_avg.save_weights(filename, overwrite=False)



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
        
        return render(request,'train/success.html',{'list':losses,'samples':samples,'res':updated_weights_loss})