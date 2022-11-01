IMG_PATH = "../data/coa_resized/"
IMG_HEIGHT = 128  # The images are already resized here
IMG_WIDTH = 128  # The images are already resized here

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 20
TRAIN_BATCH_SIZE = 32  # Let's see, I don't have GPU, Google Colab is best hope
TEST_BATCH_SIZE = 32  # Let's see, I don't have GPU, Google Colab is best hope
FULL_BATCH_SIZE = 32

###### Train and Test time #########

DATA_PATH = "../data/coa_resized/"
AUTOENCODER_MODEL_PATH = "baseline_autoencoder.pt"
ENCODER_MODEL_PATH = "../data/models/deep_encoder.pt"
DECODER_MODEL_PATH = "../data/models/deep_decoder.pt"
EMBEDDING_PATH = "../data/models/data_embedding_f.npy"
IMG_DICT_PATH = "../data/models/img_dict.pkl"
# EMBEDDING_SHAPE = (1, 256, 16, 16)
EMBEDDING_SHAPE = (1, 64, 16, 16)
# TEST_RATIO = 0.2

###### Test time #########
NUM_IMAGES = 10
TEST_IMAGE_PATH = "../data/coa_resized/-1_O B lion rampant.jpg"
