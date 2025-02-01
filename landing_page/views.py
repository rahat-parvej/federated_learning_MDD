import os
from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf

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
    
    # Specify the folder containing the model files
    weights_path_loss = "models/updated_weights_loss"
    # base_model_loss=tf.keras.models.load_model("models/with_loss/model.h5")
    weights_path_sample = "models/updated_weights_sample"
    weights_path_avg = "models/updated_weights_avg"

    # Load the models
    models_loss = []
    models_sample = []
    models_avg = []
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
    if k>=2:
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


        average_weights = [tf.divide(avgw, i) for avgw in avg_global_weights]
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
        
        open("models/losses/losses_list.txt", "w").close()
        open("models/samples/sample_list.txt", "w").close()


    return render(request,'landing_page/landing_page.html')