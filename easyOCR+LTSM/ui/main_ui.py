from tkinter import Tk, filedialog, Button, Label, Listbox, Scrollbar
from ocr.easyocr_extraction import extract_text_from_image
from models.inference import predict_answer_score
from evaluation.performance_metrics import generate_performance_report

def process_images():
    """Uploads images, extracts text, and evaluates answers."""
    Tk().withdraw()  # Hide root window for file selection
    image_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    if not image_paths:
        print("⚠️ No images selected. Exiting...")
        return

    all_scores = []
    y_true, y_pred = [], []  # ✅ Ensure lists are initialized

    for image_path in image_paths:
        try:
            text_list = extract_text_from_image(image_path)

            # ✅ Ensure OCR returns a valid text string
            if not text_list or not isinstance(text_list, list):
                print(f"⚠️ No text detected in {image_path}. Skipping...")
                continue

            text = " ".join(text_list)  # Convert OCR list output to string
            split_text = text.split("?")

            if len(split_text) < 2:
                print(f"⚠️ Could not extract valid Q&A from {image_path}. Skipping...")
                continue  # Skip this image
            
            question = split_text[0].strip() + "?"  # Ensure clean formatting
            user_answer = split_text[1].strip()

            # ✅ Ensure extracted text is valid
            if not question or not user_answer:
                print(f"⚠️ Invalid text format in {image_path}. Skipping...")
                continue

            score = predict_answer_score(question, user_answer)
            all_scores.append(score)

            y_true.append(1)  # Assuming ground truth is "correct"
            y_pred.append(1 if score > 5 else 0)  # ✅ Fix: Avoid appending to NoneType

            # ✅ Update UI listbox asynchronously to keep UI responsive
            root.after(0, listbox.insert, "end", f"Q: {question}\nA: {user_answer}\nScore: {score}/10\n")
        
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")

    # ✅ Ensure y_true and y_pred are not empty before generating a report
    if y_true and y_pred:
        generate_performance_report(y_true, y_pred)
    else:
        print("⚠️ No valid predictions to generate a report.")

# Create UI Window
root = Tk()
root.title("QA Evaluation")
root.geometry("600x400")

Label(root, text="Upload Images to Evaluate Answers", font=("Arial", 14)).pack(pady=10)
Button(root, text="Upload Images", command=process_images, font=("Arial", 12)).pack(pady=5)

# Result Listbox
listbox = Listbox(root, width=80, height=10)
listbox.pack(pady=10)

# Scrollbar
scrollbar = Scrollbar(root)
scrollbar.pack(side="right", fill="y")
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

root.mainloop()
from tkinter import Tk, filedialog, Button, Label, Listbox, Scrollbar
from ocr.easyocr_extraction import extract_text_from_image
from models.inference import predict_answer_score
from evaluation.performance_metrics import generate_performance_report

def process_images():
    """Uploads images, extracts text, and evaluates answers."""
    Tk().withdraw()  # Hide root window for file selection
    image_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    if not image_paths:
        print("⚠️ No images selected. Exiting...")
        return

    all_scores = []
    y_true, y_pred = [], []  # ✅ Ensure lists are initialized

    for image_path in image_paths:
        try:
            text_list = extract_text_from_image(image_path)

            # ✅ Ensure OCR returns a valid text string
            if not text_list or not isinstance(text_list, list):
                print(f"⚠️ No text detected in {image_path}. Skipping...")
                continue

            text = " ".join(text_list)  # Convert OCR list output to string
            split_text = text.split("?")

            if len(split_text) < 2:
                print(f"⚠️ Could not extract valid Q&A from {image_path}. Skipping...")
                continue  # Skip this image
            
            question = split_text[0].strip() + "?"  # Ensure clean formatting
            user_answer = split_text[1].strip()

            # ✅ Ensure extracted text is valid
            if not question or not user_answer:
                print(f"⚠️ Invalid text format in {image_path}. Skipping...")
                continue

            score = predict_answer_score(question, user_answer)
            all_scores.append(score)

            y_true.append(1)  # Assuming ground truth is "correct"
            y_pred.append(1 if score > 5 else 0)  # ✅ Fix: Avoid appending to NoneType

            # ✅ Update UI listbox asynchronously to keep UI responsive
            root.after(0, listbox.insert, "end", f"Q: {question}\nA: {user_answer}\nScore: {score}/10\n")
        
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")

    # ✅ Ensure y_true and y_pred are not empty before generating a report
    if y_true and y_pred:
        generate_performance_report(y_true, y_pred)
    else:
        print("⚠️ No valid predictions to generate a report.")

# Create UI Window
root = Tk()
root.title("QA Evaluation")
root.geometry("600x400")

Label(root, text="Upload Images to Evaluate Answers", font=("Arial", 14)).pack(pady=10)
Button(root, text="Upload Images", command=process_images, font=("Arial", 12)).pack(pady=5)

# Result Listbox
listbox = Listbox(root, width=80, height=10)
listbox.pack(pady=10)

# Scrollbar
scrollbar = Scrollbar(root)
scrollbar.pack(side="right", fill="y")
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

root.mainloop()
