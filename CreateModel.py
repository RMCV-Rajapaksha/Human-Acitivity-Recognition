# ============ Imports ============
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# ============ Configuration ============
# Seeds
SEED_CONSTANT = 27
np.random.seed(SEED_CONSTANT)
random.seed(SEED_CONSTANT)
tf.random.set_seed(SEED_CONSTANT)

# Paths and parameters
DATASET_PATH = "C:/Users/ROG/OneDrive/Desktop/Projects/Sem 6 AI/UCF50"
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

# ============ Data Processing ============
def frames_extraction(video_path):
    """Extract and preprocess frames from video."""
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read() 
        
        if not success:
            break
            
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

def create_dataset():
    """Create dataset from video files."""
    features = []
    labels = []
    video_files_paths = []
    
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_PATH, class_name))
        
        for file_name in files_list:
            video_path = os.path.join(DATASET_PATH, class_name, file_name)
            frames = frames_extraction(video_path)
            
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_path)
    
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels, video_files_paths

# ============ Model Creation ============
def create_convlstm_model():
    """Create ConvLSTM model architecture."""
    model = Sequential()
    
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last", recurrent_dropout=0.2,
                         return_sequences=True, input_shape=(SEQUENCE_LENGTH,
                         IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh',
                         data_format="channels_last", recurrent_dropout=0.2,
                         return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(Flatten())
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
    
    return model

# ============ Training and Visualization ============
def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    """Plot training metrics."""
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.show()

# ============ Main Execution ============
if __name__ == "__main__":
    # Create dataset
    features, labels, video_files_paths = create_dataset()
    one_hot_encoded_labels = to_categorical(labels)
    
    # Split dataset
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, one_hot_encoded_labels, test_size=0.25, shuffle=True, 
        random_state=SEED_CONSTANT)
    
    # Create and compile model
    model = create_convlstm_model()
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                                 restore_best_weights=True)
    training_history = model.fit(x=features_train, y=labels_train, epochs=50,
                               batch_size=4, shuffle=True, validation_split=0.2,
                               callbacks=[early_stopping])
    
    # Save model
    date_time = dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    model_file_name = f'convlstm_model_{date_time}.h5'
    model.save(model_file_name)
    
    # Plot results
    plot_metric(training_history, 'loss', 'val_loss', 'Loss vs Validation Loss')
    plot_metric(training_history, 'accuracy', 'val_accuracy', 'Accuracy vs Validation Accuracy')