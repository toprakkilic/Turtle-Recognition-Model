import tensorflow as tf
import numpy as np
import cv2
import os
from config import TrainingConfig
from keras.applications.mobilenet_v2 import preprocess_input 

class TurtleRecognizer:
    def __init__(self, config):
        self.cfg = config
        
        if not os.path.exists(self.cfg.MODEL_PATH):
            print(f"❌ Hata: {self.cfg.MODEL_PATH} bulunamadı!")
            return

        self.model = tf.keras.models.load_model(
            self.cfg.MODEL_PATH,
            custom_objects={'preprocess_input': preprocess_input}
        )
        
        with open(self.cfg.CLASSES_PATH, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.threshold = 60.0 
        print(f"✅ Model yüklendi. {len(self.class_names)} kaplumbağa tanınabilir.")

    def predict(self, img_path):
        if not os.path.exists(img_path):
            print("❌ Dosya bulunamadı!")
            return None, 0

        img_raw = cv2.imread(img_path)
        if img_raw is None:
            print("❌ Resim okunamadı!")
            return None, 0

        img_resized = cv2.resize(img_raw, self.cfg.IMG_SIZE)
        img_array = img_resized.astype('float32') 
        img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array, verbose=0)
        confidence = 100 * np.max(preds[0])
        name = self.class_names[np.argmax(preds[0])]

        print("\n" + "═"*40)
        if confidence >= self.threshold:
            status_msg = f"✅ ONAYLANDI: {name} (%{confidence:.1f})"
        else:
            status_msg = f"❌ REDDEDILDI (Guven: %{confidence:.1f})"
        
        print(status_msg)
        print("═"*40)

        return name, confidence

if __name__ == "__main__":
    config = TrainingConfig()
    recognizer = TurtleRecognizer(config)
    
    test_img = "proje_data/test/t016/dluhVmVVbR.JPG"
    recognizer.predict(test_img)    