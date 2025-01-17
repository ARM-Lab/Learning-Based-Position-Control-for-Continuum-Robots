import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
import os
import datetime
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras_tuner import  Hyperband
from sklearn.utils import shuffle


if __name__ == '__main__':

    if not os.path.isdir('modelsIK'):
        os.makedirs('modelsIK')

    MODEL_NAME = 'IKmodel3x3'
    input_dim = 3
    timesteps = 1
    output_dim = 3

    input2 = np.loadtxt("dataset.txt", dtype='f', delimiter=',')
    input = input2[:29600]
    input = shuffle(input)
    y_train = input[:23680, :3]
    X_train = input[:23680, 3:]
   
    y_val = input[23680:30000, :3]
    X_val = input[23680:30000, 3:]
    

# Define the model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=512, step=8),
                           activation='relu', input_dim=input_dim))
    
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=512, step=8),
                            activation='relu'))
 
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=512, step=8),
                             activation='relu'))
    
    model.add(layers.Dense(output_dim))
    model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

# Define the tuner
tuner = Hyperband(
     build_model,
     objective='loss',
     max_epochs=20,
     factor=3,
     hyperband_iterations=1,
     seed=None,
     hyperparameters=None,
     tune_new_entries=True,
     allow_new_entries=True,
     max_retries_per_trial=0,
     max_consecutive_failed_trials=3  
 )

# Search for the best hyperparameters
tuner.search(x=X_train,
             y=y_train,
             epochs=15,
             validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

# Build the final model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)
     
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
# Train final the model
model.fit(X_train, y_train, epochs=1500, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

    #  Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation loss: ", loss)
print("Validation accuracy: ", accuracy)
model.summary()
model.save(f'modelsIK/{MODEL_NAME}.model')
