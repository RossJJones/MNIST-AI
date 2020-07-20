from NeuralNet import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2

X_Train,X_Test,Y_Train,Y_Test = Dataset_Setup()
model = Load_Model()

def Quit(): #Quits the GUI
    root.destroy()

def loadImage(): #Loads in image and puts it into GUI
    clearFrame()
    ImageDir = tk.filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(('JPG files','*.jpg'),('JPEG files','*.jpeg'),('PNG files','*.png'))) #Runs image upload GUI
    img = cv2.imread(ImageDir)
    computeImage(img)
    img = cv2.resize(img,(256,256))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    imgFrame = tk.Label(frame,image=img)
    imgFrame.image = img
    imgFrame.pack()

def computeImage(image): #Reformats the image and gets predictions from CNN
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(28,28))
    image = SetupImageData(image)
    Prediction = Test_Model(model,image)
    text = tk.Label(frame3,text="I think this number is " + str(Prediction))
    text.pack()

def modelAccuracy(): #Displays the accuracy of the model
    clearFrame()
    accuracy = Evaluate_Model(model,X_Test,Y_Test)
    text = tk.Label(frame3,text="Model accuracy: {:5.2f}%".format(100*accuracy))
    text.pack()
    
def clearFrame(): #Clears the GUI to make space for new a image
    for item in frame.pack_slaves():
        item.pack_forget()
    for item in frame3.pack_slaves():
        item.pack_forget()

root = tk.Tk() #Starts the GUI loop
bg = tk.Canvas(root, height=500, width=600, bg='#000000') #Sets GUI background
bg.pack() #Shows GUI background

#Setup GUI frames
frame = tk.Frame(root, bg="black")
frame.place(relwidth=0.65,relheight=0.45,relx=0.17,rely=0.2)
frame2 = tk.Frame(root, bg="black")
frame2.place(relwidth=0.8,relheight=0.2,relx=0.1,rely=0.7)
frame3 = tk.Frame(root, bg="black")
frame3.place(relwidth=0.8,relheight=0.1,relx=0.1,rely=0.05)

#Setup GUI buttons
upload = tk.Button(frame2, text="Upload Image",padx=10,pady=5,fg="white",bg="black",command=loadImage)
evaluate = tk.Button(frame2, text="Evaluate Model",padx=10,pady=5,fg="white",bg="black",command=modelAccuracy)
end = tk.Button(frame2, text="Exit",padx=10,pady=5,fg="white",bg="black",command=Quit)

#Show GUI buttons
upload.pack()
evaluate.pack()
end.pack()

root.mainloop()
