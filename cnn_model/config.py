# REL_PATH = ""
REL_PATH = "../"
IMG_PATH = "{}data/coa_cutout/".format(REL_PATH)
IMG_PATH_OUTPUT = "{}data/coa_cutout/".format(REL_PATH)
IMG_HEIGHT = 128  # The images are already resized here
IMG_WIDTH = 128  # The images are already resized here

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 15
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

###### Train and Test time #########
DATA_PATH = "{}data/coa_renamed/".format(REL_PATH)
MODEL_NAME = "_transformed17"
AUTOENCODER_MODEL_PATH = "baseline_autoencoder{}.pt".format(REL_PATH, MODEL_NAME)
ENCODER_MODEL_PATH = "{}data/models/deep_encoder{}.pt".format(REL_PATH, MODEL_NAME)
DECODER_MODEL_PATH = "{}data/models/deep_decoder{}.pt".format(REL_PATH, MODEL_NAME)
EMBEDDING_PATH = "{}data/models/data_embedding_f{}.npy".format(REL_PATH, MODEL_NAME)
IMG_DICT_PATH = "{}data/models/img_dict{}.pkl".format(REL_PATH, MODEL_NAME)
ANGLE_DICT_PATH = "{}data/coa_rotation_angle_rounded-dict.csv".format(REL_PATH)
EMBEDDING_SHAPE = (1, 64, 16, 16)
# TEST_RATIO = 0.2

###### Test time #########
NUM_IMAGES = 10
TEST_IMAGE_PATH = "../data/coa_resized/-1_O B lion rampant.jpg"
