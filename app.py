import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils import load_image_as_matrix, flatten_matrix
from similarity import calculate_euclidean_distance, calculate_similarity_percentage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    distance = None
    similarity = None
    error = None

    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            error = 'No file part'
            return render_template('index.html', error=error)
        
        file1 = request.files['image1']
        file2 = request.files['image2']

        if file1.filename == '' or file2.filename == '':
            error = 'No selected file'
            return render_template('index.html', error=error)

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            
            file1.save(path1)
            file2.save(path2)
            
            try:
                matrix1 = load_image_as_matrix(path1)
                vec1 = flatten_matrix(matrix1)
                
                matrix2 = load_image_as_matrix(path2)
                vec2 = flatten_matrix(matrix2)
                
                distance = calculate_euclidean_distance(vec1, vec2)
                similarity = calculate_similarity_percentage(distance)
                
            except Exception as e:
                error = f"Error processing images: {str(e)}"
        else:
            error = 'Allowed file types are png, jpg, jpeg'

    return render_template('index.html', distance=distance, similarity=similarity, error=error)

if __name__ == '__main__':
    app.run(debug=True)
