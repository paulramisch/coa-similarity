from cnn_model.torch_infer import compute_similar_images, set_vars, plot_similar_cnn, plot_similar_images_grid
import csv


def test_model(encoder, img_dict, embedding, device, data_path='../data/coa_renamed/',
               test_data_path="../data/test_data.csv", test_data_secondary_path="../data/test_data_secondary.csv",
               angle=True, angle_dict_path="../data/coa_rotation_angle_rounded-dict.csv"):
    # Get test data
    test_data = list(csv.reader(open(test_data_path)))
    test_data_secondary = list(csv.reader(open(test_data_secondary_path)))

    # Set Score vars
    self = 0
    score = 0
    score_secondary = 0

    if angle:
        # Load angle dictionary
        with open(angle_dict_path, mode='r') as infile:
            reader = csv.reader(infile)
            angle_dict = dict((rows[0], rows[1]) for rows in reader)
    else:
        angle_dict = None

    # Iterate over test_data
    for idx, test in enumerate(test_data):
        image_list = compute_similar_images(data_path + test[0], 10, embedding, encoder, device, img_dict,
                                            test[0], angle_dict)
        self += 1 if image_list[0][0] == test[0] else 0

        for idy, img in enumerate(image_list):
            # the first row is the img name itself
            if idy > 0:
                score += 1 if img[0] in test else 0
                score_secondary += 1 if img[0] in test_data_secondary[idx] else 0
        # plot_similar_cnn(data_path + test[0], embedding, encoder, device, img_dict, 20, test[0], angle_dict).show()

        # Set model_name
        if 'model_name' not in globals():
            model_name = 'Autoencoder'

    return f"{model_name} score: {score}, secondary score: {score_secondary}, self: {self}/{len(test_data)}"

if __name__ == "__main__":
    # Parameter
    model_name = "_transformed20"
    REL_PATH = "../"
    angle = True

    # Set variable paths
    data = "{}data/coa_renamed/".format(REL_PATH)
    src_encoder = "{}data/models/deep_encoder{}.pt".format(REL_PATH, model_name)
    src_dict = "{}data/models/img_dict{}.pkl".format(REL_PATH, model_name)
    src_embedding = "{}data/models/data_embedding_f{}.npy".format(REL_PATH, model_name)

    encoder, img_dict, embedding, device = set_vars(src_encoder, src_dict, src_embedding)
    result = test_model(encoder, img_dict, embedding, device, data, angle=angle)
    print(result)

    # for idx, img in enumerate(test_data):
    #     list = []
    #     for img_sim in img:
    #         if img_sim != "":
    #             list.append([img_sim, 0])
    #     plot_similar_images_grid("../data/coa_renamed/" + img[0], list, 'Similar images').save(f"../data/plots/similar/{idx}.jpg")