from cnn_model.torch_infer import compute_similar_images, set_vars
import csv

# Parameter
MODEL_NAME = "_transformed"
REL_PATH = "../"

# Set variable paths
DATA_PATH = "{}data/coa/".format(REL_PATH)
src_encoder = "{}data/models/deep_encoder{}.pt".format(REL_PATH, MODEL_NAME)
src_dict = "{}data/models/img_dict{}.pkl".format(REL_PATH, MODEL_NAME)
src_embedding = "{}data/models/data_embedding_f{}.npy".format(REL_PATH, MODEL_NAME)

# Get test data
test_data = list(csv.reader(open("../data/test_data.csv")))
test_data_secondary = list(csv.reader(open("../data/test_data_secondary.csv")))

# Set system vars
encoder, img_dict, embedding, device = set_vars(src_encoder, src_dict, src_embedding)
score = -len(test_data)
score_secondary = -len(test_data_secondary)

# Iterate over test_data
for idx, test in enumerate(test_data):
    image_list = compute_similar_images('../data/coa/' + test[0], 10, embedding, encoder, device, img_dict)

    for img in image_list:
        score += 1 if img[0] in test else 0
        score_secondary += 1 if img[0] in test_data_secondary[idx] else 0

print(f"{MODEL_NAME} score: {score}, secondary score: {score_secondary}")
# _transformed score: 11, secondary score: 5
# _stn: 4, secondary: 1

# Epochs = 7, Training Loss : 0.03284949064254761
# Validation Loss decreased, saving new best model
# Epochs = 7, Validation Loss : 0.035060491412878036