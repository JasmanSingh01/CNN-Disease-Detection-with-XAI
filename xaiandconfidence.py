import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from keras.models import load_model
from PIL import Image, ImageTk, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = load_model("hybrid_model_350.h5", compile=False)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load the model. Error: {str(e)}")

# Load the labels
class_names = [
    "Normal",  # 0
    "Normal No Cardiovascular Disease",  # 1
    "Bacterial Pneumonia",  # 2
    "Tuberculosis",  # 3
    "Viral Pneumonia",  # 4
    "Cardiovascular Disease Found"  # 5
]

# Disease information dictionary
disease_info = {
    "Normal": {
        "description": "The individual has no noticeable signs of disease.",
        "precautions": "Maintain a healthy lifestyle, balanced diet, and regular exercise.",
        "medications": "None.",
        "diet": "Eat a balanced diet rich in vegetables, fruits, and whole grains."
    },
    "Normal No Cardiovascular Disease": {
        "description": "No cardiovascular disease detected, but should monitor overall health.",
        "precautions": "Maintain a healthy diet and regular physical activity.",
        "medications": "None.",
        "diet": "Consume a low-fat, low-sodium diet with plenty of fruits and vegetables."
    },
    "Bacterial Pneumonia": {
        "description": "A serious bacterial infection of the lungs that causes breathing difficulties.",
        "precautions": "Rest and hydration are essential. Avoid smoking and pollutants.",
        "medications": "Antibiotics like Amoxicillin, Azithromycin.",
        "diet": "Consume foods rich in Vitamin C and fluids like soups and teas."
    },
    "Tuberculosis": {
        "description": "A serious infectious disease that primarily affects the lungs.",
        "precautions": "Follow treatment regimens strictly. Avoid close contact with others during active stages.",
        "medications": "Isoniazid, Rifampin, Pyrazinamide.",
        "diet": "High-protein foods, iron-rich foods like leafy greens, lean meats."
    },
    "Viral Pneumonia": {
        "description": "Inflammation of the lungs caused by a viral infection, such as the flu.",
        "precautions": "Rest and stay hydrated. Isolate to avoid spreading the virus.",
        "medications": "Antiviral drugs if prescribed. Pain relievers like acetaminophen.",
        "diet": "Increase fluid intake, eat easily digestible food like soups."
    },
    "Cardiovascular Disease Found": {
        "description": "Heart-related conditions that affect the cardiovascular system.",
        "precautions": "Monitor blood pressure, cholesterol levels, and maintain a low-stress lifestyle.",
        "medications": "Aspirin, Beta-blockers, Statins, ACE inhibitors.",
        "diet": "Low-sodium, low-fat diet. High fiber from vegetables, fruits, and whole grains."
    }
}

# Classify the uploaded image
def classify_image(image):
    try:
        # Resize the image to be at least 224x224 and crop from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # Convert image to RGB if it is not already in that mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Display the input image
        input_img = ImageTk.PhotoImage(image)
        input_img_label.config(image=input_img)
        input_img_label.image = input_img

        # Turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict with the model
        prediction = model.predict(data)
        index = np.argmax(prediction)

        # Explainable AI: Show probabilities
        probabilities = prediction[0]
        explanation = "\n".join([f"{class_names[i]}: {probabilities[i]*100:.2f}%" for i in range(len(class_names))])

        # Get the disease name
        disease_name = class_names[index]
        confidence_score = probabilities[index]  # Get the confidence score of the predicted class

        # Update the result label with the predicted class name
        result_label.config(text=f"Predicted Disease: {disease_name}", fg="#ffffff")

        # Check if the confidence score is greater than 0.98
        if confidence_score > 0.98:
            confidence_label.config(text=f"Confidence Score: {confidence_score:.4f}", fg="#000000")
            confidence_label.pack()
            result_label.pack()  # Display the Predicted Class label
        else:
            confidence_label.pack_forget()  # Hide the Confidence Score label
            result_label.pack_forget()  # Hide the Predicted Class label

        # Display probabilities and disease information
        explanation_label.config(text=f"Probabilities:\n{explanation}", fg="#000000")
        show_disease_info(disease_name)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during classification: {str(e)}")

# Display detailed disease information
def show_disease_info(disease_name):
    info = disease_info.get(disease_name, None)
    if info:
        description_label.config(text=f"Description: {info['description']}", fg="#ffffff")
        precautions_label.config(text=f"Precautions: {info['precautions']}", fg="#ffffff")
        medications_label.config(text=f"Medications: {info['medications']}", fg="#ffffff")
        diet_label.config(text=f"Diet: {info['diet']}", fg="#ffffff")
        description_label.pack(pady=5)
        precautions_label.pack(pady=5)
        medications_label.pack(pady=5)
        diet_label.pack(pady=5)
    else:
        messagebox.showerror("Error", "No information available for the detected disease.")

# Select an image to classify
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        classify_image(image)

# Create the main window
root = tk.Tk()
root.title("Disease Detection Classifier")
root.geometry("800x800")
root.configure(bg="#808080")

# Title
title_label = tk.Label(root, text="Disease Detection", font=("Arial", 15, "bold"), bg="#808080", fg="#ffffff")#673AB7
title_label.pack(pady=20)

# Input Frame
input_frame = tk.Frame(root, bg="#808080")
input_frame.pack(pady=20)
input_label = tk.Label(input_frame, text="Input Image", font=("Arial", 12, "bold"), bg="#808080", fg="#ffffff")#3F51B5
input_label.pack()
input_img_label = tk.Label(input_frame, bg="#808080")
input_img_label.pack()

# Result Frame
result_frame = tk.Frame(root, bg="#808080")
result_frame.pack(pady=20)
result_label = tk.Label(result_frame, text="", font=("Arial", 14), bg="#808080", fg="#ffffff")#FF5722
result_label.pack()
explanation_label = tk.Label(result_frame, text="", font=("Arial", 12), bg="#808080", fg="#ffffff", justify="left")
explanation_label.pack()
confidence_label = tk.Label(result_frame, text="", font=("Arial", 12), bg="#808080", fg="#ffffff")#FF5722
confidence_label.pack_forget()

# Disease Info Labels
description_label = tk.Label(root, text="", font=("Arial", 10), bg="#808080")
precautions_label = tk.Label(root, text="", font=("Arial", 10), bg="#808080")
medications_label = tk.Label(root, text="", font=("Arial", 10), bg="#808080")
diet_label = tk.Label(root, text="", font=("Arial", 10), bg="#808080")

# Select Image Button
select_button = tk.Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief="raised", padx=20, pady=10)
select_button.pack(pady=0)

# Run the Tkinter application
root.mainloop()
