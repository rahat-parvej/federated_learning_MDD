import os
from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,\
GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.backend import clear_session

# Create your views here.
def landing_page_view(request):

    
    def cnnmodel():
        input_layer = Input(shape=(3840, 20))

        conv1 = Conv1D(filters=5, kernel_size=3, strides=1)(input_layer)
        bn1 = BatchNormalization()(conv1)
        relu1 = LeakyReLU()(bn1)
        maxpool1 = MaxPool1D(pool_size=2, strides=2)(relu1)

        conv2 = Conv1D(filters=5, kernel_size=3, strides=1)(maxpool1)
        relu2 = LeakyReLU()(conv2)
        maxpool2 = MaxPool1D(pool_size=2, strides=2)(relu2)

        dropout1 = Dropout(0.5)(maxpool2)

        conv3 = Conv1D(filters=5, kernel_size=3, strides=1)(dropout1)
        relu3 = LeakyReLU()(conv3)
        avgpool1 = AveragePooling1D(pool_size=2, strides=2)(relu3)

        dropout2 = Dropout(0.5)(avgpool1)

        conv4 = Conv1D(filters=5, kernel_size=3, strides=1)(dropout2)
        relu4 = LeakyReLU()(conv4)
        avgpool2 = AveragePooling1D(pool_size=2, strides=2)(relu4)

        conv5 = Conv1D(filters=5, kernel_size=3, strides=1)(avgpool2)
        relu5 = LeakyReLU()(conv5)
        global_avgpool = GlobalAveragePooling1D()(relu5)

        output_layer = Dense(1, activation='sigmoid')(global_avgpool)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile('adam', loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall', 'AUC'])
        
        return model
    
    def fednova_aggregate(weight_files, sample_counts, epochs_list=None, batch_size=25):
        """
        FedNova aggregation from saved weight files
        
        Args:
            weight_files: List of paths to .weights.h5 files
            sample_counts: List of sample counts for each client
            epochs_list: List of epochs each client trained (default: 30)
            batch_size: Batch size used (default: 25)
        
        Returns:
            Aggregated weights
        """
        
        n_clients = len(weight_files)
        
        # Default epochs (from your training_going: epochs=21)
        if epochs_list is None:
            epochs_list = [30] * n_clients
        
        # Parameters for Adam optimizer (which you're using)
        beta = 0.9  # Momentum parameter β₁ for Adam
        
        # Step 1: Load all client weights
        client_weights = []
        for weight_file in weight_files:
            # Create a model instance
            model = cnnmodel()  # Your model architecture function
            model.load_weights(weight_file)
            client_weights.append(model.get_weights())
        
        # Step 2: Calculate FedNova normalization coefficients
        normalization_coeffs = []
        
        for i in range(n_clients):
            n_samples = sample_counts[i]
            n_epochs = epochs_list[i]
            
            # Calculate number of local steps/batches
            n_batches = n_samples // batch_size
            if n_samples % batch_size != 0:
                n_batches += 1
            
            local_steps = n_epochs * n_batches
            
            # FedNova normalization: τ_eff = (1 - β^local_steps) / (1 - β)
            if local_steps > 0:
                tau_eff = (1 - beta**local_steps) / (1 - beta)
            else:
                tau_eff = 1
            
            normalization_coeffs.append(tau_eff)
        
        # Step 3: Calculate normalized weights
        total_normalized_weight = 0
        n_layers = len(client_weights[0])
        
        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(client_weights[0][j]) for j in range(n_layers)]
        
        for i in range(n_clients):
            # FedNova weight: samples / τ_eff
            weight = sample_counts[i] / normalization_coeffs[i]
            total_normalized_weight += weight
            
            # Add weighted contribution
            for j in range(n_layers):
                aggregated_weights[j] += weight * client_weights[i][j]
        
        # Step 4: Normalize by total weight
        for j in range(n_layers):
            aggregated_weights[j] = aggregated_weights[j] / total_normalized_weight
        
        return aggregated_weights



    # Specify the folder containing the model files
    weights_path_loss = "models/updated_weights_loss"
    # base_model_loss=tf.keras.models.load_model("models/with_loss/model.h5")
    weights_path_sample = "models/updated_weights_sample"
    weights_path_avg = "models/updated_weights_avg"
    weights_path_fednova = "models/updated_weights_fednova"

    # Load the models
    models_loss = []
    models_sample = []
    models_avg = []
    models_fednova = []
    with open('models/losses/losses_list.txt', 'r') as file:
        # Read the lines and store them in a list
        losses =  [float(line.strip()) for line in file.readlines()]
    with open('models/samples/sample_list.txt', 'r') as file:
        # Read the lines and store them in a list
        data_lens = [float(line.strip()) for line in file.readlines()]
    i=0

    for  file_name in os.listdir(weights_path_sample):
        
        if file_name.endswith(".h5"):
            model_path_sample = os.path.join(weights_path_sample, file_name)
            model_sample=cnnmodel()
            model_sample.load_weights(model_path_sample)
            models_sample.append((model_sample,data_lens[i]))
            i=i+1
    k=0        
    for  file_name in os.listdir(weights_path_loss):
        
        if file_name.endswith(".h5"):
            model_path_loss = os.path.join(weights_path_loss, file_name)
            
            # model_loss = tf.keras.models.clone_model(base_model_loss)
            model_loss = cnnmodel()
            model_loss.load_weights(model_path_loss)
            models_loss.append((model_loss,losses[k]))
            k=k+1
            
    
    j=0
    for  file_name in os.listdir(weights_path_avg):
        if file_name.endswith(".h5"):
            model_path_avg = os.path.join(weights_path_avg, file_name)
            model_avg=cnnmodel()
            model_avg.load_weights(model_path_avg)
            models_avg.append(model_avg)
            j=j+1
    l = 0
    models_fednova = []
    for file_name in os.listdir(weights_path_fednova):
        if file_name.endswith(".h5"):
            model_path_fednova = os.path.join(weights_path_fednova, file_name)
            model_fednova = cnnmodel()
            model_fednova.load_weights(model_path_fednova)
            # Use the same sample counts as sample models
            if l < len(data_lens):
                models_fednova.append((model_fednova, data_lens[l]))
                l += 1

    weight_files = []
    valid_sample_counts = []

    # Use weights_path_fednova, not weights_path_sample!
    for file_name in os.listdir(weights_path_fednova):  # CHANGED HERE
        if file_name.endswith(".weights.h5"):
            weight_files.append(os.path.join(weights_path_fednova, file_name))  # CHANGED HERE
            
            # Try to match sample count
            if len(valid_sample_counts) < len(data_lens):
                valid_sample_counts.append(data_lens[len(valid_sample_counts)])

    total_clients = max(len(os.listdir(weights_path_loss)), 
                    len(os.listdir(weights_path_sample)),
                    len(os.listdir(weights_path_avg)),
                    len(os.listdir(weights_path_fednova)))

    if total_clients >= 2:
        k=0
        global_weights_sum_for_loss = [tf.zeros_like(w) for w in model_loss.get_weights()]
        global_weights_sum = [tf.zeros_like(w) for w in model_sample.get_weights()]
        avg_global_weights = [tf.zeros_like(w) for w in model_avg.get_weights()]
        
        
    
        total_sample=0
        total_loss=0
        for local_model, loss in models_loss:
            local_weights_loss = local_model.get_weights()

            #proposed method
            total_loss +=(1-loss)
            weighted_local_weights_for_loss = [(1-loss) * w for w in local_weights_loss]
            global_weights_sum_for_loss = [tf.add(gw_l, wlw_l) for gw_l, wlw_l in zip(global_weights_sum_for_loss, weighted_local_weights_for_loss)]

        for local_model, num_samples_i in models_sample:
            local_weights = local_model.get_weights()

            #fedAvg
            total_sample+=num_samples_i
            weighted_local_weights = [num_samples_i * w for w in local_weights]
            global_weights_sum = [tf.add(gw, wlw) for gw, wlw in zip(global_weights_sum, weighted_local_weights)]

        for local_model in models_avg:
            local_weights = local_model.get_weights()

            #for average weight
            avg_global_weights = [tf.add(agw, lw) for agw, lw in zip(avg_global_weights, local_weights)]


        average_weights = [tf.divide(avgw, len(models_avg)) for avgw in avg_global_weights]
        average_weights_for_loss = [tf.divide(gws_l, total_loss) for gws_l in global_weights_sum_for_loss]
        average_weights_fedAvg = [tf.divide(gws, total_sample) for gws in global_weights_sum]

        final_model_loss=tf.keras.models.load_model("models/with_loss/model.h5")
        final_model_loss.set_weights(average_weights_for_loss)
        final_model_loss.save('models/with_loss/model.h5')

        final_model_sample=tf.keras.models.load_model("models/with_sample/model.h5")
        final_model_sample.set_weights(average_weights_fedAvg)
        final_model_sample.save('models/with_sample/model.h5')

        final_model_avg=tf.keras.models.load_model("models/with_avg/model.h5")
        final_model_avg.set_weights(average_weights)
        final_model_avg.save('models/with_avg/model.h5')
        #fednova
        start_time = time.time()
        fednova_weights = fednova_aggregate(
            weight_files=weight_files,
            sample_counts=valid_sample_counts,
            epochs_list=[30] * len(weight_files),  # All trained 21 epochs
            batch_size=25  # Your batch size
        )
        end_time = time.time()
        print("Aggregation time:", end_time - start_time)
        # Create and save FedNova model
        fednova_model = cnnmodel()
        fednova_model.set_weights(fednova_weights)
        fednova_model.save('models/with_fednova/model.h5')


        #deleting
        
        for root, dirs, files in os.walk(weights_path_loss):
            for f in files:
                os.unlink(os.path.join(root, f))
        for root, dirs, files in os.walk(weights_path_sample):
            for f in files:
                os.unlink(os.path.join(root, f))
        for root, dirs, files in os.walk(weights_path_avg):
            for f in files:
                os.unlink(os.path.join(root, f))
        for root, dirs, files in os.walk(weights_path_fednova):
            for f in files:
                os.unlink(os.path.join(root, f))
        
        open("models/losses/losses_list.txt", "w").close()
        open("models/samples/sample_list.txt", "w").close()


    return render(request,'landing_page/landing_page.html')