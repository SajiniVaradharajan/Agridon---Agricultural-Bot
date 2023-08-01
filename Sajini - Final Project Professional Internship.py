import tkinter as tk2
from tkinter import DISABLED
from tkinter import NORMAL
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from csv import *
from tkinter import NORMAL
import sklearn

print(sklearn.__version__)

from tkinter import *
from tkvideo import tkvideo
import numpy
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
import tkinter as tk
from tkinter import filedialog, Tk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy

from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
def back():
    root2 = tk2.Tk()
    canvas = tk2.Canvas(root2, width=900, height=500)
    canvas.pack()
    root2.title("Agridon")
    label1 = tk2.Label(root2,
                       text="Greetings! I am Agridon, your companion specializing in Crop and Fertilizer Prediction,",font=("Arial",14))
    canvas.create_window(400, 30, window=label1)
    label3 = tk2.Label(root2, text="Plant Disease Prediction and Soil Type Prediction.", font=("Arial", 14))
    canvas.create_window(400, 60, window=label3)
    label2 = tk2.Label(root2, text="Which app would you like to choose?",font=("Arial",16),fg="red")
    canvas.create_window(180, 90, window=label2)

    def CAFP():
        root2.destroy()
        # Create an instance of Tkinter frame
        win = Tk()
        win.title("Crop and Fertilizer Prediction App")
        # Set the geometry of the Tkinter frame
        win.geometry("1000x1000")
        win.configure(bg="green1")


        main_lst = []

        def Submit():
            lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get()]
            main_lst.append(lst)
            with open("data_entry.csv", "w") as file:
                Writer = writer(file)
                Writer.writerow(["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep"])
                Writer.writerows(main_lst)
                messagebox.showinfo("Information", "Saved succesfully")

        def PredictC():
            import pandas as pd
            data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
            print(data.head())
            data['soil type'].value_counts()
            df = data.copy()
            df.drop(["N", "P", "K"], axis=1)

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['soil type'] = le.fit_transform(df['soil type'])
            X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type']].values
            Y = df['label'].values

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            dp = pd.read_csv("data_entry.csv")
            print(dp.head())

            from sklearn.svm import SVC
            svm = SVC()
            svm.fit(x_train, y_train)
            pred = svm.predict(x_test)
            print(pred)

            # Calculate accuracy of model

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, pred) * 100
            print(accuracy)

            le = LabelEncoder()
            dp['SoilTypep'] = le.fit_transform(dp['SoilTypep'])
            global y_pred
            y_pred = svm.predict(dp)
            print(y_pred)
            labelp = ttk.Label(win, text="The crop which can be grown in " + entry.get() + " soil is " + y_pred)
            labelp.pack(pady=5)
            import pyttsx3

            friend = pyttsx3.init()
            friend.runAndWait()
            friend.say("The crop which can be grown in " + entry.get() + " soil is " + y_pred)
            newVolume = 1
            friend.setProperty("volume", newVolume)
            friend.setProperty('pitch', 0.9)
            friend.runAndWait()
            my_label = tk.Label(win)
            my_label.place(relx=0.8, rely=0.5, anchor='center')
            from tkvideo import tkvideo
            player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
            player.play()

        main_lst1 = []

        def PredictF():
            lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get(), y_pred]
            main_lst1.append(lst)
            with open("data_entry2.csv", "w") as file:
                Writer = writer(file)
                Writer.writerow(["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep", "Cropp"])
                Writer.writerows(main_lst)

            import pandas as pd
            data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
            print(data.head())
            data['soil type'].value_counts()
            df = data.copy()
            df.drop(["N", "P", "K"], axis=1)

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['label'])
            df['soil type'] = le.fit_transform(df['soil type'])
            X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type', 'label']].values
            Y = df['fertilizer'].values

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            dp1 = pd.read_csv("data_entry2.csv")
            print(dp1.head())
            dp1['SoilTypep'] = le.fit_transform(dp1['SoilTypep'])
            dp1['Cropp'] = le.fit_transform(dp1['Cropp'])

            from sklearn.svm import SVC
            svm = SVC()
            svm.fit(x_train, y_train)
            pred = svm.predict(x_test)
            print(pred)

            # Calculate accuracy of model

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, pred) * 100
            print(accuracy)

            global y_pred2
            y_pred2 = svm.predict(dp1)
            print(y_pred2)
            labelp = ttk.Label(win, text="The fertilizer which can be used for " + y_pred + " crop is " + y_pred2)
            labelp.pack(pady=5)
            import pyttsx3
            friend = pyttsx3.init()
            friend.runAndWait()
            friend.say("The fertilizer which can be used for " + y_pred + " crop is " + y_pred2)
            newVolume = 1
            friend.setProperty("volume", newVolume)
            friend.setProperty('pitch', 0.9)
            friend.runAndWait()
            my_label = tk.Label(win)
            my_label.place(relx=0.8, rely=0.5, anchor='center')
            from tkvideo import tkvideo
            player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
            player.play()

        main_lst2 = []

        def PredictM():
            lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get(), y_pred, y_pred2]
            main_lst2.append(lst)
            with open("data_entry3.csv", "w") as file:
                Writer = writer(file)
                Writer.writerow(
                    ["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep", "Cropp", "Fertilizerp"])
                Writer.writerows(main_lst)

            import pandas as pd
            data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
            print(data.head())
            data['soil type'].value_counts()
            df = data.copy()
            df.drop(["N", "P", "K"], axis=1)

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['label'])
            df['soil type'] = le.fit_transform(df['soil type'])
            df['fertilizer'] = le.fit_transform(df['fertilizer'])
            X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type', 'label', 'fertilizer']].values
            Y = df['biofertilizer'].values

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            dp2 = pd.read_csv("data_entry3.csv")
            print(dp2.head())
            dp2['SoilTypep'] = le.fit_transform(dp2['SoilTypep'])
            dp2['Cropp'] = le.fit_transform(dp2['Cropp'])
            dp2['Fertilizerp'] = le.fit_transform(dp2['Fertilizerp'])

            from sklearn.svm import SVC
            svm = SVC()
            svm.fit(x_train, y_train)
            pred = svm.predict(x_test)
            print(pred)

            # Calculate accuracy of model

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, pred) * 100
            print(accuracy)

            global y_pred3
            y_pred3 = svm.predict(dp2)
            print(y_pred2)
            labelp = ttk.Label(win,
                               text="Suggestion: The biofertilizer which can be used for " + y_pred + " crop is " + y_pred3 + ". This will ensure soil health and eco-friendliness.")
            labelp.pack(pady=5)
            my_label = tk.Label(win)
            my_label.place(relx=0.8, rely=0.5, anchor='center')
            from tkvideo import tkvideo
            player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
            player.play()
            import pyttsx3
            friend = pyttsx3.init()
            friend.say("Here's a suggestion. The biofertilizer which can be used for " + y_pred + " crop is " + y_pred3 + ". this will ensure soil health and eco-friendliness")
            newVolume = 1
            friend.setProperty("volume", newVolume)
            friend.setProperty('pitch', 0.9)
            friend.runAndWait()

        label1 = ttk.Label(win,
                           text='Welcome to Crop and Fertilizer Prediction App. Enter all the values and then click on save, after which, click on predict crop, predict fertilizer and predict bio-fertilizer.')
        label1.pack(padx=5)

        def back1():
            win.destroy()

        # Define a function to update the entry widget
        def entry_update(text):
            entry.delete(0, END)
            entry.insert(0, text)

        # Create an Entry Widget
        label2 = ttk.Label(win, text='Choose your soil type: ')
        label2.pack(pady=5)
        entry = Entry(win, width=30)
        entry.pack(pady=2)

        # Create Multiple Buttons with different commands
        button_dict = {}
        option = ["Sandy", "Clayey", "Alluvial", "Red Sandy Loam", "Deep Loam", "Sandy Loam", "Clayey Loam", "Loamy"]

        for i in option:
            def func(x=i):
                return entry_update(x)

            button_dict[i] = ttk.Button(win, text=i, command=func)
            button_dict[i].pack()

        def close():
            win.destroy()
            back()

        label5 = ttk.Label(win, text='Enter the temperature (in Celsius): ')
        label5.pack(pady=5)
        entryi = ttk.Entry(win)
        entryi.pack(padx=5)
        label6 = ttk.Label(win, text='Enter the humidity: ')
        label6.pack(pady=5)
        entry1 = ttk.Entry(win)
        entry1.pack(padx=5, pady=0)
        label7 = ttk.Label(win, text='Enter the PH value of the soil: ')
        label7.pack(pady=5)
        entry2 = ttk.Entry(win)
        entry2.pack(padx=5, pady=0)
        label8 = ttk.Label(win, text='Enter the rainfall: ')
        label8.pack(pady=5)
        entry3 = ttk.Entry(win)
        entry3.pack(padx=5, pady=0)
        label9 = ttk.Label(win, text='Enter the moisture content of the soil: ')
        label9.pack(pady=5)
        entry4 = ttk.Entry(win)
        entry4.pack(padx=5, pady=0)
        submit = ttk.Button(win, text="Save", command=Submit)
        submit.pack(pady=2)
        predictc = ttk.Button(win, text="Predict Crop ", command=PredictC)
        predictc.pack(pady=2)
        predictf = ttk.Button(win, text="Predict Fertilizer ", command=PredictF)
        predictf.pack(pady=2)
        predictm = ttk.Button(win, text="Predict Bio-Fertilizer", command=PredictM)
        predictm.pack(pady=2)
        close = ttk.Button(win, text="Back", command=close)
        close.pack(pady=2)
        back1 = ttk.Button(win, text="Close all", command=back1)
        back1.pack(padx=2)
        imgq = ImageTk.PhotoImage(Image.open("Agridon4.png"))
        panel = tk.Label(win, image=imgq)
        panel.place(relx=0.8, rely=0.5, anchor='center')
        panel.image = imgq

        win.mainloop()

    def PDPA():
        root2.destroy()
        my_w: Tk = tk.Tk()
        my_w.geometry("1000x1000")  # Size of the window
        my_w.configure(bg="green")
        my_w.title('Plant Disease Prediction App')
        my_font1 = ('times', 18, 'bold')
        l1 = tk.Label(my_w, text='Welcome to Plant Disease Prediction App', width=30, font=("Comic Sans MS", 18),
                      fg="white", bg="green")
        l1.grid(row=1, column=1)

        def upload():
            global img_processed
            global img
            f_types = [('Jpg Files', '*.jpg')]
            filename = filedialog.askopenfilename(filetypes=f_types)
            img = Image.open(filename)
            img_resized = img.resize((200, 200))
            img = ImageTk.PhotoImage(img_resized)
            predict_button = tk.Button(text="Predict", command=predict, state=NORMAL, font=("Arial", 18), bg="yellow")
            predict_button.place(relx=0.3, rely=0.2, anchor='w')
            b2 = tk.Button(my_w, image=img)  # using Button
            b2.place(relx=0.2, rely=0.3)
            img2 = image.load_img(filename, target_size=(256, 256))
            image_array = numpy.array(img2)
            img_expanded = numpy.expand_dims(img2, axis=0)
            img_processed = preprocess_input(img_expanded)
            return img_processed

        def predict():

            import sklearn
            print(sklearn.__version__)

            # Training Data
            train_path = "C:/Users/LENOVO/PycharmProjects/Trials 3/New Plant Diseases Dataset(Augmented) - Copy/New Plant Diseases Dataset(Augmented)/train"
            # Validation Data
            valid_path = "C:/Users/LENOVO/PycharmProjects/Trials 3/New Plant Diseases Dataset(Augmented) - Copy/New Plant Diseases Dataset(Augmented)/valid"

            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

            # Define the image data generator for rescaling the pixel values
            datagen = ImageDataGenerator(rescale=1. / 255)

            # Define the batch size
            batch_size = 69

            # Load and preprocess the training images
            train_gen = datagen.flow_from_directory(train_path, target_size=(256, 256), batch_size=batch_size,
                                                    class_mode='categorical')

            # Load and preprocess the validation images
            valid_gen = datagen.flow_from_directory(valid_path, target_size=(256, 256), batch_size=batch_size,
                                                    class_mode='categorical')

            # Print the number of classes in the dataset
            num_classes = len(train_gen.class_indices)
            print("Number of classes:", num_classes)

            import numpy as np
            from keras.models import load_model

            model = load_model("PlantDiseasePredictionApp.h5")
            model.summary()
            preds = model.predict(img_processed)
            predicted_class_index = np.argmax(preds)
            class_labels = list(train_gen.class_indices.keys())
            predicted_label = class_labels[predicted_class_index]
            print("Predicted label:", predicted_label)
            result_label.config(text="")
            result_label.config(text="This plant is: " + predicted_label)
            if 'healthy' in predicted_label:
                result_label2 = tk.Label(text="This plant is healthy.", font=("Arial", 15), bg='green', fg='white')
                result_label2.place(relx=0.3, rely=0.74, anchor='center')
            else:
                result_label3 = tk.Label(text="This plant is diseased.", font=("Arial", 15), bg='green', fg='white')
                result_label3.place(relx=0.3, rely=0.74, anchor='center')
            my_label = tk.Label(my_w)
            my_label.place(relx=0.8, rely=0.6, anchor='center')
            from tkvideo import tkvideo
            player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
            player.play()
            predictedlabel2=predicted_label.replace('_',"")
            import pyttsx3

            friend = pyttsx3.init()
            friend.runAndWait()
            friend.say("This plant is: " + predictedlabel2)
            if 'healthy' in predicted_label:
                friend.say("This plant is healthy")
            else:
                friend.say("This plant is diseased")
            newVolume = 1
            friend.setProperty("volume", newVolume)
            friend.setProperty('pitch', 0.9)
            friend.runAndWait()


        def close():
            my_w.destroy()
            back()

        def back1():
            my_w.destroy()

        upload_button = tk.Button(text="Upload", command=upload, font=("Arial", 18), bg="pink")
        upload_button.place(relx=0.2, rely=0.2, anchor='e')
        predict_button = tk.Button(text="Predict", command=predict, state=DISABLED, font=("Arial", 18), bg="yellow")
        predict_button.place(relx=0.3, rely=0.2, anchor='w')

        result_label = tk.Label(text="", font=("Arial", 18), bg="green", fg="white")
        result_label.place(relx=0.3, rely=0.7, anchor='center')
        close = tk.Button(my_w, text="Back", command=close, font=("Arial", 18), bg="cyan")
        close.place(relx=0.6, rely=0.2, anchor='w')
        back2 = tk.Button(my_w, text="Close all", command=back1, font=("Arial", 18), bg="white")
        back2.place(relx=0.8, rely=0.2, anchor='w')
        imgm = ImageTk.PhotoImage(Image.open("Agridon.png"))
        panel = tk.Label(my_w, image=imgm, bg="green")
        panel.place(relx=0.8, rely=0.8, anchor='center')


        my_w.mainloop()

    def STPA():
        root2.destroy()
        w: Tk = tk.Tk()
        w.geometry("1000x1000")  # Size of the window
        w.configure(bg="green")
        w.title('Soil type Prediction App')
        my_font1 = ('times', 18, 'bold')
        w.configure(bg="goldenrod1")

        l1 = tk.Label(w, text='Welcome to Soil Type Prediction App!', width=30, font=("Comic Sans MS", 18),
                      bg="goldenrod1", fg="black")
        l1.grid(row=1, column=1)

        b1 = tk.Button(w, text='Upload File of Soil',
                       width=20, command=lambda: upload(), bg="pink", font=("Arial", 18))
        b1.grid(row=7, column=1)
        b2 = tk.Button(w, text='Predict Soil Type',
                       width=20, command=lambda: predict(), state=DISABLED, bg="green1", font=("Arial", 18))
        b2.grid(row=7, column=3)

        def upload():
            global img_processed
            global img
            f_types = [('Jpg Files', '*.jpg')]
            filename = filedialog.askopenfilename(filetypes=f_types)
            img = Image.open(filename)
            img_resized = img.resize((200, 200))
            img = ImageTk.PhotoImage(img_resized)
            b3 = tk.Button(w, image=img)  # using Button
            b3.grid(row=20, column=1)
            b2 = tk.Button(w, text='Predict Soil Type',
                           width=20, command=lambda: predict(), state=NORMAL, bg="green1", font=("Arial", 18))
            b2.grid(row=7, column=3)
            img2 = image.load_img(filename, target_size=(256, 256))
            image_array = numpy.array(img2)
            img_expanded = numpy.expand_dims(img2, axis=0)
            img_processed = preprocess_input(img_expanded)
            return img_processed

        def predict():
            import sklearn
            print(sklearn.__version__)

            # Training Data
            train_path = 'Soil Types Final/Dataset/Train'
            # Validation Data
            valid_path = 'Soil Types Final/Dataset/Valid'

            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

            # Define the image data generator for rescaling the pixel values
            datagen = ImageDataGenerator(rescale=1. / 255)

            # Define the batch size
            batch_size = 50

            # Load and preprocess the training images
            train_gen = datagen.flow_from_directory(train_path, target_size=(256, 256), batch_size=batch_size,
                                                    class_mode='categorical')

            # Load and preprocess the validation images
            valid_gen = datagen.flow_from_directory(valid_path, target_size=(256, 256), batch_size=batch_size,
                                                    class_mode='categorical')

            # Print the number of classes in the dataset
            num_classes = len(train_gen.class_indices)
            print("Number of classes:", num_classes)

            from keras.models import load_model
            import numpy as np

            model = load_model("Soils.h5")
            model.summary()
            preds = model.predict(img_processed)
            predicted_class_index = np.argmax(preds)
            class_labels = list(train_gen.class_indices.keys())
            predicted_label = class_labels[predicted_class_index]
            print("Predicted label:", predicted_label)
            result_label.config(text="")
            result_label.config(text="The soil type is: " + predicted_label)
            import pyttsx3

            friend = pyttsx3.init()
            friend.runAndWait()
            friend.say("The soil type is: " + predicted_label)
            newVolume = 1
            friend.setProperty("volume", newVolume)
            friend.setProperty('pitch', 0.9)
            friend.runAndWait()
            my_label = tk.Label(w)
            my_label.place(relx=0.8, rely=0.6, anchor='center')
            from tkvideo import tkvideo
            player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
            player.play()

            w.mainloop()

        def close():
            w.destroy()
            back()

        def back1():
            w.destroy()

        result_label = tk.Label(w, text='', width=30, font=("Arial", 18), bg="goldenrod1", fg="black")
        result_label.grid(row=30, column=1)
        b3 = tk.Button(w, text='Back',
                       width=10, command=lambda: close(), font=("Arial", 18), bg="cyan", fg="black")
        b3.grid(row=11, column=3)
        b4 = tk.Button(w, text='Close all',
                       width=8, command=lambda: back1(), font=("Arial", 18), bg="white", fg="black")
        b4.grid(row=11, column=4)
        imgy = ImageTk.PhotoImage(Image.open("Agridon.png"))
        panel = tk.Label(w, image=imgy, bg="goldenrod1")
        panel.place(relx=0.8, rely=0.8, anchor='center')
        panel.image=imgy


    buttony1 = tk2.Button(root2, text="Soil Type Prediction", font = ("Arial", 18),command=STPA, bg="orange")
    canvas.create_window(380, 130, window=buttony1)
    buttonn1 = tk2.Button(root2, text="Crop and Fertilizer Prediction", command=CAFP, font = ("Arial", 18),bg="orange")
    canvas.create_window(380, 190, window=buttonn1)
    buttoni1 = tk2.Button(root2, text="Plant Disease Prediction", command=PDPA,font = ("Arial", 18), bg="orange")
    canvas.create_window(380, 250, window=buttoni1)
    im = PhotoImage(file="Agridon.png")
    label = tk2.Label(root2, image=im)
    canvas.create_window(700, 400, window=label)

    root2.mainloop()


import tkinter as tk2
from tkinter import DISABLED
from tkinter import NORMAL
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from csv import *
from tkinter import NORMAL
import sklearn
print(sklearn.__version__)
import numpy
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
root2=tk2.Tk()
canvas=tk2.Canvas(root2,width=900,height=500)
canvas.pack()
root2.title("Agridon")
label1=tk2.Label(root2,text="Greetings! I am Agridon, your companion specializing in Crop and Fertilizer Prediction,",font=("Arial",14))
canvas.create_window(400,30,window=label1)
label3=tk2.Label(root2,text="Plant Disease Prediction and Soil Type Prediction.",font=("Arial",14))
canvas.create_window(400,60,window=label3)
label2=tk2.Label(root2,text="Which app would you like to choose?",font=("Arial",16),fg="red")
canvas.create_window(180,90,window=label2)

def CAFP():
    root2.destroy()
    # Create an instance of Tkinter frame
    win = Tk()
    win.title("Crop and Fertilizer Prediction App")
    # Set the geometry of the Tkinter frame
    win.geometry("1000x1000")
    win.configure(bg="green1")

    main_lst = []

    def Submit():
        lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get()]
        main_lst.append(lst)
        with open("data_entry.csv", "w") as file:
            Writer = writer(file)
            Writer.writerow(["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep"])
            Writer.writerows(main_lst)
            messagebox.showinfo("Information", "Saved succesfully")

    def PredictC():
        import pandas as pd
        data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
        print(data.head())
        data['soil type'].value_counts()
        df = data.copy()
        df.drop(["N", "P", "K"], axis=1)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['soil type'] = le.fit_transform(df['soil type'])
        X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type']].values
        Y = df['label'].values

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        dp = pd.read_csv("data_entry.csv")
        print(dp.head())

        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(x_train, y_train)
        pred = svm.predict(x_test)
        print(pred)

        # Calculate accuracy of model

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, pred) * 100
        print(accuracy)

        le = LabelEncoder()
        dp['SoilTypep'] = le.fit_transform(dp['SoilTypep'])
        global y_pred
        y_pred = svm.predict(dp)
        print(y_pred)
        labelp = ttk.Label(win, text="The crop which can be grown in " + entry.get() + " soil is " + y_pred)
        labelp.pack(pady=5)
        my_label = tk.Label(win)
        my_label.place(relx=0.8,rely=0.5,anchor='center')
        from tkvideo import tkvideo
        player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
        player.play()
        import pyttsx3
        friend = pyttsx3.init()
        friend.say(
            "The crop which can be grown in " + entry.get() + " soil is " + y_pred)
        newVolume = 1
        friend.setProperty("volume", newVolume)
        friend.setProperty('pitch', 0.9)
        friend.runAndWait()

    main_lst1 = []

    def PredictF():
        lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get(), y_pred]
        main_lst1.append(lst)
        with open("data_entry2.csv", "w") as file:
            Writer = writer(file)
            Writer.writerow(["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep", "Cropp"])
            Writer.writerows(main_lst)

        import pandas as pd
        data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
        print(data.head())
        data['soil type'].value_counts()
        df = data.copy()
        df.drop(["N", "P", "K"], axis=1)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        df['soil type'] = le.fit_transform(df['soil type'])
        X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type', 'label']].values
        Y = df['fertilizer'].values

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        dp1 = pd.read_csv("data_entry2.csv")
        print(dp1.head())
        dp1['SoilTypep'] = le.fit_transform(dp1['SoilTypep'])
        dp1['Cropp'] = le.fit_transform(dp1['Cropp'])

        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(x_train, y_train)
        pred = svm.predict(x_test)
        print(pred)

        # Calculate accuracy of model

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, pred) * 100
        print(accuracy)

        global y_pred2
        y_pred2 = svm.predict(dp1)
        print(y_pred2)
        labelp = ttk.Label(win, text="The fertilizer which can be used for " + y_pred + " crop is " + y_pred2)
        labelp.pack(pady=5)
        my_label = tk.Label(win)
        my_label.place(relx=0.8,rely=0.5,anchor='center')
        from tkvideo import tkvideo
        player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
        player.play()
        import pyttsx3
        friend = pyttsx3.init()
        friend.say(
            "The fertilizer which can be used for " + y_pred + " crop is " + y_pred2)
        newVolume = 1
        friend.setProperty("volume", newVolume)
        friend.setProperty('pitch', 0.9)
        friend.runAndWait()

    main_lst2 = []

    def PredictM():
        lst = [entryi.get(), entry1.get(), entry2.get(), entry3.get(), entry4.get(), entry.get(), y_pred, y_pred2]
        main_lst2.append(lst)
        with open("data_entry3.csv", "w") as file:
            Writer = writer(file)
            Writer.writerow(
                ["Temperaturep", "Humidityp", "PHp", "Rainfallp", "Moisturep", "SoilTypep", "Cropp", "Fertilizerp"])
            Writer.writerows(main_lst)

        import pandas as pd
        data = pd.read_csv("Crop and Fertilizer Recommendation Dataset.csv")
        print(data.head())
        data['soil type'].value_counts()
        df = data.copy()
        df.drop(["N", "P", "K"], axis=1)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        df['soil type'] = le.fit_transform(df['soil type'])
        df['fertilizer'] = le.fit_transform(df['fertilizer'])
        X = df[['temperature', 'humidity', 'ph', 'rainfall', 'moisture', 'soil type', 'label', 'fertilizer']].values
        Y = df['biofertilizer'].values

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        dp2 = pd.read_csv("data_entry3.csv")
        print(dp2.head())
        dp2['SoilTypep'] = le.fit_transform(dp2['SoilTypep'])
        dp2['Cropp'] = le.fit_transform(dp2['Cropp'])
        dp2['Fertilizerp'] = le.fit_transform(dp2['Fertilizerp'])

        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(x_train, y_train)
        pred = svm.predict(x_test)
        print(pred)

        # Calculate accuracy of model

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, pred) * 100
        print(accuracy)

        global y_pred3
        y_pred3 = svm.predict(dp2)
        print(y_pred2)
        labelp = ttk.Label(win,
                           text="Suggestion: The biofertilizer which can be used for " + y_pred + " crop is " + y_pred3 + ". This will ensure soil health and eco-friendliness.")
        labelp.pack(pady=5)
        my_label = tk.Label(win)
        my_label.place(relx=0.8,rely=0.5,anchor='center')
        from tkvideo import tkvideo
        player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
        player.play()
        import pyttsx3
        friend = pyttsx3.init()
        friend.say(
            "Here's a suggestion. The biofertilizer which can be used for " + y_pred + " crop is " + y_pred3 + ". this will ensure soil health and eco-friendliness")
        newVolume = 1
        friend.setProperty("volume", newVolume)
        friend.setProperty('pitch', 0.9)
        friend.runAndWait()


    label1 = ttk.Label(win,
                       text='Welcome to Crop and Fertilizer Prediction App. Enter all the values and then click on save, after which, click on predict crop, predict fertilizer and predict bio-fertilizer.')
    label1.pack(padx=5)

    def back1():
        win.destroy()



    # Define a function to update the entry widget
    def entry_update(text):
        entry.delete(0, END)
        entry.insert(0, text)

    # Create an Entry Widget
    label2 = ttk.Label(win, text='Choose your soil type: ')
    label2.pack(pady=5)
    entry = Entry(win, width=30)
    entry.pack(pady=2)

    # Create Multiple Buttons with different commands
    button_dict = {}
    option = ["Sandy", "Clayey", "Alluvial", "Red Sandy Loam", "Deep Loam", "Sandy Loam", "Clayey Loam", "Loamy"]

    for i in option:
        def func(x=i):
            return entry_update(x)

        button_dict[i] = ttk.Button(win, text=i, command=func)
        button_dict[i].pack()

    def close():
        win.destroy()
        back()

    label5 = ttk.Label(win, text='Enter the temperature (in Celsius): ')
    label5.pack(pady=5)
    entryi = ttk.Entry(win)
    entryi.pack(padx=5)
    label6 = ttk.Label(win, text='Enter the humidity: ')
    label6.pack(pady=5)
    entry1 = ttk.Entry(win)
    entry1.pack(padx=5, pady=0)
    label7 = ttk.Label(win, text='Enter the PH value of the soil: ')
    label7.pack(pady=5)
    entry2 = ttk.Entry(win)
    entry2.pack(padx=5, pady=0)
    label8 = ttk.Label(win, text='Enter the rainfall: ')
    label8.pack(pady=5)
    entry3 = ttk.Entry(win)
    entry3.pack(padx=5, pady=0)
    label9 = ttk.Label(win, text='Enter the moisture content of the soil: ')
    label9.pack(pady=5)
    entry4 = ttk.Entry(win)
    entry4.pack(padx=5, pady=0)
    submit = ttk.Button(win, text="Save", command=Submit)
    submit.pack(pady=2)
    predictc = ttk.Button(win, text="Predict Crop ", command=PredictC)
    predictc.pack(pady=2)
    predictf = ttk.Button(win, text="Predict Fertilizer ", command=PredictF)
    predictf.pack(pady=2)
    predictm = ttk.Button(win, text="Predict Bio-Fertilizer", command=PredictM)
    predictm.pack(pady=2)
    close = ttk.Button(win, text="Back", command=close)
    close.pack(pady=2)
    back1 = ttk.Button(win, text="Close all", command=back1)
    back1.pack(padx=2)
    imgq = ImageTk.PhotoImage(Image.open("Agridon4.png"))
    panel = tk.Label(win, image=imgq)
    panel.place(relx=0.8,rely=0.5,anchor='center')
    panel.image = imgq

    win.mainloop()

import tkinter as tk
from tkinter import filedialog, Tk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy

from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input,decode_predictions
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

def PDPA():
    root2.destroy()
    my_w: Tk = tk.Tk()
    my_w.geometry("1000x1000")  # Size of the window
    my_w.configure(bg="green")
    my_w.title('Plant Disease Prediction App')
    my_font1 = ('times', 18, 'bold')
    l1 = tk.Label(my_w, text='Welcome to Plant Disease Prediction App', width=30, font=("Comic Sans MS",18),fg="white",bg="green")
    l1.grid(row=1, column=1)



    def upload():
        global img_processed
        global img
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        img = Image.open(filename)
        img_resized = img.resize((200, 200))
        img = ImageTk.PhotoImage(img_resized)
        predict_button = tk.Button(text="Predict", command=predict,state=NORMAL,font=("Arial",18),bg="yellow")
        predict_button.place(relx=0.3, rely=0.2, anchor='w')
        b2 = tk.Button(my_w, image=img)  # using Button
        b2.place(relx=0.2,rely=0.3)
        img2 = image.load_img(filename, target_size=(256, 256))
        image_array = numpy.array(img2)
        img_expanded = numpy.expand_dims(img2, axis=0)
        img_processed = preprocess_input(img_expanded)
        return img_processed




    def predict():

        import sklearn
        print(sklearn.__version__)

        # Training Data
        train_path = "C:/Users/LENOVO/PycharmProjects/Trials 3/New Plant Diseases Dataset(Augmented) - Copy/New Plant Diseases Dataset(Augmented)/train"
        # Validation Data
        valid_path = "C:/Users/LENOVO/PycharmProjects/Trials 3/New Plant Diseases Dataset(Augmented) - Copy/New Plant Diseases Dataset(Augmented)/valid"

        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

        # Define the image data generator for rescaling the pixel values
        datagen = ImageDataGenerator(rescale=1. / 255)

        # Define the batch size
        batch_size = 69

        # Load and preprocess the training images
        train_gen = datagen.flow_from_directory(train_path, target_size=(256, 256), batch_size=batch_size,
                                                class_mode='categorical')

        # Load and preprocess the validation images
        valid_gen = datagen.flow_from_directory(valid_path, target_size=(256, 256), batch_size=batch_size,
                                                class_mode='categorical')

        # Print the number of classes in the dataset
        num_classes = len(train_gen.class_indices)
        print("Number of classes:", num_classes)

        import numpy as np
        from keras.models import load_model

        model=load_model("PlantDiseasePredictionApp.h5")
        model.summary()
        preds = model.predict(img_processed)
        predicted_class_index = np.argmax(preds)
        class_labels = list(train_gen.class_indices.keys())
        predicted_label = class_labels[predicted_class_index]
        print("Predicted label:", predicted_label)
        result_label.config(text="")
        result_label.config(text="This plant is: " + predicted_label)
        if 'healthy' in predicted_label:
            result_label2 = tk.Label(text="This plant is healthy.",font=("Arial",15),bg='green',fg='white')
            result_label2.place(relx=0.3, rely=0.74, anchor='center')
        else:
            result_label3 = tk.Label(text="This plant is diseased.",font=("Arial",15),bg='green',fg='white')
            result_label3.place(relx=0.3, rely=0.74, anchor='center')
        predictedlabel2=predicted_label.replace('_',' ')
        import pyttsx3

        friend = pyttsx3.init()
        friend.runAndWait()
        friend.say("This plant is: " + predictedlabel2)
        if 'healthy' in predicted_label:
            friend.say("This plant is healthy")
        else:
            friend.say("This plant is diseased")
        newVolume = 1
        friend.setProperty("volume", newVolume)
        friend.setProperty('pitch', 0.9)
        friend.runAndWait()
        my_label = tk.Label(my_w)
        my_label.place(relx=0.8,rely=0.6,anchor='center')
        from tkvideo import tkvideo
        player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
        player.play()



    def close():
        my_w.destroy()
        back()

    def back1():
        my_w.destroy()

    upload_button = tk.Button(text="Upload", command=upload,font=("Arial",18),bg="pink")
    upload_button.place(relx=0.2, rely=0.2, anchor='e')
    predict_button = tk.Button(text="Predict", command=predict, state=DISABLED,font=("Arial",18),bg="yellow")
    predict_button.place(relx=0.3, rely=0.2, anchor='w')

    result_label = tk.Label(text="",font=("Arial",18),bg="green",fg="white")
    result_label.place(relx=0.3, rely=0.7, anchor='center')
    close = tk.Button(my_w, text="Back", command=close,font=("Arial",18),bg="cyan")
    close.place(relx=0.6,rely=0.2,anchor='w')
    back2 = tk.Button(my_w, text="Close all", command=back1,font=("Arial",18),bg="white")
    back2.place(relx=0.8,rely=0.2,anchor='w')
    imgm = ImageTk.PhotoImage(Image.open("Agridon.png"))
    panel = tk.Label(my_w, image=imgm,bg="green")
    panel.place(relx=0.8,rely=0.8,anchor='center')

    my_w.mainloop()

def STPA():
    root2.destroy()
    w: Tk = tk.Tk()
    w.geometry("1000x1000")  # Size of the window
    w.configure(bg="green")
    w.title('Soil type Prediction App')
    my_font1 = ('times', 18, 'bold')
    w.configure(bg="goldenrod1")

    l1 = tk.Label(w, text='Welcome to Soil Type Prediction App!', width=30, font=("Comic Sans MS",18),bg="goldenrod1",fg="black")
    l1.grid(row=1, column=1)

    b1 = tk.Button(w, text='Upload Soil File',
                   width=20, command=lambda: upload(),bg="pink",font=("Arial",18))
    b1.grid(row=7, column=1)
    b2 = tk.Button(w, text='Predict Soil Type',
                   width=20, command=lambda: predict(), state = DISABLED,bg="green1",font=("Arial",18))
    b2.grid(row=7, column=3)


    def upload():
        global img_processed
        global img
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        img = Image.open(filename)
        img_resized = img.resize((200, 200))
        img = ImageTk.PhotoImage(img_resized)
        b3 = tk.Button(w, image=img)  # using Button
        b3.grid(row=20,column=1)
        b2 = tk.Button(w, text='Predict Soil Type',
                       width=20, command=lambda: predict(), state=NORMAL,bg="green1",font=("Arial",18))
        b2.grid(row=7, column=3)
        img2 = image.load_img(filename, target_size=(256,256))
        image_array = numpy.array(img2)
        img_expanded = numpy.expand_dims(img2, axis=0)
        img_processed = preprocess_input(img_expanded)
        return img_processed

    def predict():
        import sklearn
        print(sklearn.__version__)

        # Training Data
        train_path = 'Soil Types Final/Dataset/Train'
        # Validation Data
        valid_path = 'Soil Types Final/Dataset/Valid'

        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

        # Define the image data generator for rescaling the pixel values
        datagen = ImageDataGenerator(rescale=1. / 255)

        # Define the batch size
        batch_size = 50

        # Load and preprocess the training images
        train_gen = datagen.flow_from_directory(train_path, target_size=(256, 256), batch_size=batch_size,
                                                class_mode='categorical')

        # Load and preprocess the validation images
        valid_gen = datagen.flow_from_directory(valid_path, target_size=(256, 256), batch_size=batch_size,
                                                class_mode='categorical')

        # Print the number of classes in the dataset
        num_classes = len(train_gen.class_indices)
        print("Number of classes:", num_classes)

        from keras.models import load_model
        import numpy as np

        model=load_model("Soils.h5")
        model.summary()
        preds = model.predict(img_processed)
        predicted_class_index = np.argmax(preds)
        class_labels = list(train_gen.class_indices.keys())
        predicted_label = class_labels[predicted_class_index]
        print("Predicted label:", predicted_label)
        my_label = tk.Label(w)
        my_label.place(relx=0.8,rely=0.6,anchor='center')
        result_label.config(text="")
        result_label.config(text="The soil type is: " + predicted_label)
        import pyttsx3

        friend = pyttsx3.init()
        friend.runAndWait()
        friend.say("The soil type is: " + predicted_label)
        newVolume=1
        friend.setProperty("volume",newVolume)
        friend.setProperty('pitch', 0.9)
        friend.runAndWait()
        from tkvideo import tkvideo
        player = tkvideo("Agridon_Spins5_AdobeExpress.mp4", my_label, loop=4, size=(320, 400))
        player.play()

        w.mainloop()
    def close():
        w.destroy()
        back()
    def back1():
        w.destroy()

    result_label = tk.Label(w, text='', width=30, font=("Arial",18),bg="goldenrod1",fg="black")
    result_label.grid(row=30, column=1)

    b3 = tk.Button(w, text='Back',
                   width=10, command=lambda: close(), font=("Arial", 18), bg="cyan", fg="black")
    b3.grid(row=11, column=3)
    b4 = tk.Button(w, text='Close all',
                   width=8, command=lambda: back1(), font=("Arial", 18), bg="white", fg="black")
    b4.grid(row=11, column=4)
    imgy = ImageTk.PhotoImage(Image.open("Agridon.png"))
    panel = tk.Label(w, image=imgy,bg="goldenrod1")
    panel.place(relx=0.8,rely=0.8,anchor='center')
    panel.image = imgy


buttony1=tk2.Button(root2,text="Soil Type Prediction",command=STPA,font = ("Arial", 18),bg="orange")
canvas.create_window(380,130,window=buttony1)
buttonn1=tk2.Button(root2,text="Crop and Fertilizer Prediction",command=CAFP, font = ("Arial", 18),bg="orange")
canvas.create_window(380,190,window=buttonn1)
buttoni1=tk2.Button(root2,text="Plant Disease Prediction",command=PDPA, font = ("Arial", 18),bg="orange")
canvas.create_window(380,250,window=buttoni1)
im=PhotoImage(file="Agridon.png")
label=tk2.Label(root2,image=im)
canvas.create_window(700, 400,window=label)


root2.mainloop()


