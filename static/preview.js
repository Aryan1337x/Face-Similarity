function readURL(input, previewId) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function(e) {
            var preview = document.getElementById(previewId);
            preview.src = e.target.result;
            preview.style.display = 'block';
        }

        reader.readAsDataURL(input.files[0]);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var image1Input = document.getElementById('image1');
    var image2Input = document.getElementById('image2');

    if (image1Input) {
        image1Input.addEventListener('change', function() {
            readURL(this, 'preview-image1');
        });
    }

    if (image2Input) {
        image2Input.addEventListener('change', function() {
            readURL(this, 'preview-image2');
        });
    }
});
