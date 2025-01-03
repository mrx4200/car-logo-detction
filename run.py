import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\ckw10\OneDrive\Desktop\Car logo Detection\car_logo_best_model2.pt")


root = tk.Tk()
root.title("Resim ya da Camera ")
root.geometry("400x250")

output_label = tk.Label(root, text="Araba logosunu tanıma yöntemini seçin", font=("Arial", 12))
output_label.pack(pady=10)

def analyze_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        results = model.predict(source=image_path, show=True)
        output_label.config(text="Image analyzed. Check console for results.")

def analyze_camera():
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model.predict(source=frame, verbose=False)
            for result in results:
                classes = result.names
                for det in result.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = det[:6]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    name = classes[int(class_id)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Camera Feed - Press 'q' to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    output_label.config(text="Camera analysis ended. Check console for results.")


def analyze_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        cap = cv2.VideoCapture(video_path) 
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = model.predict(source=frame, verbose=False)
                for result in results:
                    classes = result.names
                    for det in result.boxes.data.tolist():
                        x1, y1, x2, y2, confidence, class_id = det[:6]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        name = classes[int(class_id)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{name} ({confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Video Analysis - Press 'q' to exit", frame)

                if cv2.waitKey(20) & 0xFF == ord('q'): 
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        output_label.config(text="Video analysis ended. Check console for results.")


image_button = tk.Button(root, text="Resim yükleyerek tanıma.", command=analyze_image)
image_button.pack(pady=5)

camera_button = tk.Button(root, text="Kamara ile tanıma.", command=analyze_camera)
camera_button.pack(pady=5)

video_button = tk.Button(root, text="Video dosyasından tanıma.", command=analyze_video)
video_button.pack(pady=5)

root.mainloop()
