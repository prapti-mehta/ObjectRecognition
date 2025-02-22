import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
num=10
fig,axs=plt.subplots(1,10,figsize=(15,3))
for i in range(num):
    axs=plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
plt.show()
x_train=x_train.reshape(50000,32,32,3)
x_test=x_test.reshape(10000,32,32,3)

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

model=Sequential([Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),MaxPooling2D(pool_size=(2,2)),
                 Conv2D(64,(3,3),activation='relu'),MaxPooling2D(pool_size=(2,2)),Conv2D(128,(3,3),activation='relu'),
                 Flatten(),Dense(128,activation='relu'),Dropout(0.5),Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
class_names = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse", "ship", "truck"
]

y_predict = model.predict(x_test)
predicted_classes = np.argmax(y_predict, axis=1)

fig,axes=plt.subplots(1,10,figsize=(20,6))
for i in range(10):
    image_index = i
    axes[i].imshow(x_test[image_index])
    predicted_class = class_names[predicted_classes[image_index]]
    actual_class = class_names[np.argmax(y_test[image_index])]
    axes[i].set_title(f"Predicted: {predicted_class}\nActual: {actual_class}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
lost,accuracy=model.evaluate(x_test,y_test)
print(f'accuracy:{accuracy*100}')