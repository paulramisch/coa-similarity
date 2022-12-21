# COA-Sim APP
from flask import Flask, request, jsonify
from cnn_model.torch_infer import set_vars, compute_similar_images
import cnn_model.config as config
import os

# Set the App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
print("App started")

# Load encoder, img_dict, embeddings
encoder, img_dict, embedding, device = set_vars()
print("Loaded model and embeddings")


# For the home route and health check
@app.route("/")
def index():
    return "App is running"


@app.route("/similar_coa", methods=["GET", "POST"])
def similar_coa():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'The server did not receive an image!'

        image = request.files["image"]
        similar_coa = compute_similar_images(image, 20, embedding, encoder, device, img_dict)

        # Change to json structure
        similar_coa_json = [{"img": coa[0], "similarity": coa[1]} for coa in similar_coa]

        # Return json object
        return jsonify(similar_coa_json)

    return '''
    <h1>Upload a COA to compare</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image">
      <input type="submit">
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=False, port=9000)
