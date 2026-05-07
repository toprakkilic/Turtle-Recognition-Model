import os
import tensorflow as tf
from config import TrainingConfig
from data_loader import TurtleDataLoader
from model_factory import ModelFactory

class TurtleTrainer:
    def __init__(self, config):
        self.cfg = config
        self.data_loader = TurtleDataLoader(config)
        self.model = None

    def setup(self):
        self.train_ds, self.val_ds = self.data_loader.load_data()
        self.num_classes = len(self.train_ds.class_names)
        self.model = ModelFactory.build_mobilenet_v2(self.num_classes, self.cfg)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def run(self):
        print(f"🚀 {self.num_classes} sınıf için eğitim başlıyor...")
        self.model.fit(
            self.train_ds, 
            validation_data=self.val_ds, 
            epochs=self.cfg.EPOCHS
        )
        self._save_results()

    def _save_results(self):
        if not os.path.exists("models"): 
            os.makedirs("models")
            
        self.model.save(self.cfg.MODEL_PATH)
        with open(self.cfg.CLASSES_PATH, "w") as f:
            for name in self.train_ds.class_names:
                f.write(f"{name}\n")
        print(f"✅ Eğitim tamamlandı. Model ve sınıflar '{self.cfg.MODEL_PATH}' konumuna kaydedildi.")

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = TurtleTrainer(config)
    trainer.setup()
    trainer.run()