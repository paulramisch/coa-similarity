from cnn_model.torch_infer import compute_similar_images, set_vars, plot_similar_cnn
import csv


def test_model(encoder, img_dict, embedding, device, data_path='../data/coa_renamed/',
               test_data_path="../data/test_data.csv", test_data_secondary_path="../data/test_data_secondary.csv"):
    # Get test data
    test_data = list(csv.reader(open(test_data_path)))
    test_data_secondary = list(csv.reader(open(test_data_secondary_path)))

    # Set Score vars
    score = -len(test_data)
    score_secondary = -len(test_data_secondary)

    # Iterate over test_data
    for idx, test in enumerate(test_data):
        image_list = compute_similar_images(data_path + test[0], 10, embedding, encoder, device, img_dict)

        for img in image_list:
            score += 1 if img[0] in test else 0
            score_secondary += 1 if img[0] in test_data_secondary[idx] else 0
        plot_similar_cnn(data_path + test[0], embedding, encoder, device, img_dict, 20).show()
    return f"{MODEL_NAME} score: {score}, secondary score: {score_secondary}"

if __name__ == "__main__":
    # Parameter
    MODEL_NAME = "_transformed3_cut"
    REL_PATH = "../"

    # Set variable paths
    DATA_PATH = "{}data/coa_renamed/".format(REL_PATH)
    src_encoder = "{}data/models/deep_encoder{}.pt".format(REL_PATH, MODEL_NAME)
    src_dict = "{}data/models/img_dict{}.pkl".format(REL_PATH, MODEL_NAME)
    src_embedding = "{}data/models/data_embedding_f{}.npy".format(REL_PATH, MODEL_NAME)

    encoder, img_dict, embedding, device = set_vars(src_encoder, src_dict, src_embedding)
    result = test_model(encoder, img_dict, embedding, device, data_path=DATA_PATH)
    print(result)