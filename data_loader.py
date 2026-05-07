import tensorflow as tf

class TurtleDataLoader:
    def __init__(self, config):
        self.cfg = config

    def load_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.cfg.TRAIN_DIR,
            image_size=self.cfg.IMG_SIZE,
            batch_size=self.cfg.BATCH_SIZE,
            label_mode='int'
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.cfg.VAL_DIR,
            image_size=self.cfg.IMG_SIZE,
            batch_size=self.cfg.BATCH_SIZE,
            label_mode='int'
        )
        return train_ds, val_ds