from altair import layer
from keras import layers, models, applications

class ModelFactory:
    @staticmethod
    def build_mobilenet_v2(num_classes, config):
        from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
        
        base = MobileNetV2(
            input_shape=(*config.IMG_SIZE, 3), 
            include_top=False, 
            weights='imagenet'
        )
        base.trainable = True

        for layer in base.layers[:-30]:
            layer.trainable = False

        model = models.Sequential([
            layers.Input(shape=(*config.IMG_SIZE, 3)),
            layers.Lambda(preprocess_input),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model