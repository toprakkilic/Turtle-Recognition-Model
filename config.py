class TrainingConfig:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 4
    EPOCHS = 40
    LEARNING_RATE = 0.00005
    TRAIN_DIR = "proje_data/train"
    VAL_DIR = "proje_data/val"
    MODEL_PATH = "models/seaturtle.h5"
    CLASSES_PATH = "models/classes.txt"
