import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Input, Lambda, ReLU, Add
from keras.layers import Input, Lambda, ReLU, Add
from keras.models import Model,Sequential
from keras import backend as K
from tensorflow.keras.layers import Conv2D, Reshape, Softmax, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, LeakyReLU, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

#%% process the data files
def process_data(file_path):
  
    data = np.fromfile(file_path, dtype=np.complex64) 
    total_samples = len(data)
    
    samples_per_frame = 13312
    num_frames = total_samples // samples_per_frame

    reshaped_data = data[:num_frames * samples_per_frame].reshape(num_frames, samples_per_frame)
    processed_data = reshaped_data[:, 800:]
    return processed_data

# Directory containing the files
data_dir = r'E:\LoRa Datasets\Distance 29_8_24\Distance 5m'  
data_dir1 = r'E:\LoRa Datasets\Distance 29_8_24\Distance 10m' 
data_dir2 = r'E:\LoRa Datasets\Distance 29_8_24\Distance 15m' 
data_dir3 = r'E:\LoRa Datasets\Distance 29_8_24\Distance 20m' 
data_dir4 = r'E:\LoRa Datasets\Distance 29_8_24\Distance 25m' 
data_dir5 = r'E:\LoRa Datasets\Distance 29_8_24\Distance 30m' 

device_ids = [1, 6, 8, 9]
#%%

def process_lora_data_for_distance(distance, device_ids, data_dir, max_frames=400):
    device_id_to_label = {device_id: label for label, device_id in enumerate(device_ids, start=1)}

    all_data = []
    all_labels = []

    for device_id in device_ids:
    
        file_name = f'lora_data_29_8_{distance}m_device_{device_id}_input0.sigmf-data'
        file_path = os.path.join(data_dir, file_name)
  
        processed_data = process_data(file_path)
        processed_data = processed_data[:max_frames]

        mapped_label = device_id_to_label[device_id]
    
        labels = np.full((processed_data.shape[0], 1), mapped_label)
        
        all_data.append(processed_data)
        all_labels.append(labels)
        
    combined_data = np.vstack(all_data)
    combined_labels = np.vstack(all_labels)

    labeled_dataset = np.hstack((combined_data, combined_labels))

    return labeled_dataset


# Process data for 5m, 10m, 15m, 20m, and 35m
labeled_dataset_5m = process_lora_data_for_distance(5, device_ids, data_dir)
labeled_dataset_10m = process_lora_data_for_distance(10, device_ids, data_dir1)
labeled_dataset_15m = process_lora_data_for_distance(15, device_ids, data_dir2)
labeled_dataset_20m = process_lora_data_for_distance(20, device_ids, data_dir3)
labeled_dataset_25m = process_lora_data_for_distance(25, device_ids, data_dir4)
labeled_dataset_30m = process_lora_data_for_distance(30, device_ids, data_dir5)

# Check shapes
print(labeled_dataset_5m.shape)
print(labeled_dataset_10m.shape)
print(labeled_dataset_15m.shape)
print(labeled_dataset_20m.shape)  
print(labeled_dataset_25m.shape)
#%%
final_data = np.vstack((labeled_dataset_5m,labeled_dataset_10m, labeled_dataset_15m, labeled_dataset_20m))

final_data.shape


data = final_data[:, :-1]
labels = final_data[:, -1].astype(int)
print("Data shape:", data.shape)    
print("Labels shape:", labels.shape)  

unique_classes, class_counts = np.unique(labels, return_counts=True)
print("Unique device IDs (classes):", unique_classes)
print("Counts per device ID:", class_counts)
print("Total number of unique classes:", len(unique_classes))

#%%
#---------------====RMS Normalization Per Sample==========------------------------
def _normalization(data):

    s_norm = np.zeros(data.shape, dtype=complex)

    for i in range(data.shape[0]):
        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude ** 2))
        s_norm[i] = data[i] / rms

    return s_norm

#%%
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

y_train_one_hot = to_categorical(y_train-1, num_classes=4)  
y_test_one_hot = to_categorical(y_test-1, num_classes=4)    
print("One-hot encoded training labels shape:", y_train_one_hot.shape)
print("One-hot encoded test labels shape:", y_test_one_hot.shape)

#%%
X_train_normalized = _normalization(X_train)
X_test_normalized = _normalization(X_test)
print("Training data shape:", X_train_normalized.shape)
print("Test data shape:", X_test_normalized.shape)

#%%
fft_ = np.fft.fft(X_train_normalized)

i = np.abs(fft_)
q = np.angle(fft_)

train_iq = np.stack((i, q), axis=-1)

train_iq.shape

fft_test = np.fft.fft((X_test_normalized))
                      
i_test =np.abs(fft_test)                 
q_test = np.angle(fft_test)

test_iq = np.stack((i_test, q_test), axis=-1)
test_iq.shape


#%%# Define the model
def create_cnn_model(input_shape):
    model = tf.keras.Sequential()

    model.add(Reshape((2, 12512, 1), input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), input_shape=input_shape, kernel_regularizer=l2(0.001), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    
    model.add(Conv2D(16, (3, 3), kernel_regularizer=l2(0.001), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.001),   padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    
    model.add(Flatten())

    model.add(Dense(25))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(4))  
    model.add(Softmax())

    return model

#%%

# Define the optimizer
opt = Adam(learning_rate=1e-2)
# Learning rate scheduler
early_stop = EarlyStopping('val_loss', min_delta=0, patience=30)
reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
callbacks = [early_stop, reduce_lr]

#%%

input_shape = (12512, 2) 
cnn_model = create_cnn_model(input_shape)
cnn_model.compile(optimizer=opt, loss='categorical_crossentropy')
final_epoch = 100 

# Model Summary
cnn_model.summary()
#%%
import time 
start_time = time.time()
history = cnn_model.fit(train_iq,
                        y_train_one_hot,
                        batch_size=32,
                        epochs=100,
                        callbacks=callbacks, 
                        validation_split=0.30)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

#%%

plt.plot(history.history['loss'],'bo-')
plt.plot(history.history['val_loss'], 'r*-')
plt.title('')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()



#%%
#TEST
pred_prob = cnn_model.predict(test_iq)
pred_label_ = pred_prob.argmax(axis=-1)+1
acc= accuracy_score(y_test, pred_label_)
print("Test Accuracy: {:.2f}%".format(acc * 100))

conf_mat = confusion_matrix(y_test, pred_label_)
classes = np.unique(y_test)

plt.figure()
sns.heatmap(conf_mat, annot=True,
          fmt='d', cmap='Blues',
          annot_kws={'size': 7},
          cbar=False,
          xticklabels=classes,
          yticklabels=classes)
plt.xlabel('Predicted label', fontsize=12)
plt.ylabel('True label', fontsize=12)
plt.show()

#%%
data = labeled_dataset_30m[:, :-1]
labels = labeled_dataset_30m[:, -1].astype(int)
print("Data shape:", data.shape)    
print("Labels shape:", labels.shape)  


test_norm=_normalization(data)

fft_test= np.fft.fft((test_norm))
     
i_test_ = np.abs(fft_test)
q_test_ = np.angle(fft_test)

test_iq_ = np.stack((i_test_, q_test_), axis=-1)
test_iq_.shape

pred_prob = cnn_model.predict(test_iq_)
pred_label_ = pred_prob.argmax(axis=-1)+1
acc= accuracy_score(labels, pred_label_)
print("Test Accuracy: {:.2f}%".format(acc * 100))

conf_mat = confusion_matrix(labels, pred_label_)
classes = np.unique(labels)

plt.figure()
sns.heatmap(conf_mat, annot=True,
          fmt='d', cmap='Blues',
          annot_kws={'size': 7},
          cbar=False,
          xticklabels=classes,
          yticklabels=classes)
plt.xlabel('Predicted label', fontsize=12)
plt.ylabel('True label', fontsize=12)
plt.show()
























