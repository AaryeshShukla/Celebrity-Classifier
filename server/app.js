function uploadImage() {
  const fileInput = document.getElementById("imageUpload");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image!");
    return;
  }

  const reader = new FileReader();
  reader.onloadend = function () {
    const base64data = reader.result;

    const formData = new FormData();
    formData.append("image_data", base64data);

    fetch("http://127.0.0.1:5000/classify_image", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        let topClass = data[0].class;
        let probs = data[0].class_probability;

        let html = `<h5>🎯 Predicted: <span class="text-warning">${topClass}</span></h5>`;
        html += `<p class="text-light">Class probabilities:</p>`;
        html += `<ul class="list-group">`;

        data[0].probabilities &&
          Object.entries(data[0].probabilities).forEach(([cls, prob]) => {
            html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                      ${cls}
                      <span class="badge bg-primary rounded-pill">${(prob*100).toFixed(2)}%</span>
                     </li>`;
          });

        html += `</ul>`;

        document.getElementById("result").innerHTML = html;
      })
      .catch((err) => {
        console.error(err);
        document.getElementById("result").innerHTML =
          `<div class="alert alert-danger">❌ Error: Could not classify image.</div>`;
      });
  };
  reader.readAsDataURL(file);
}

document.getElementById("uploadBtn").addEventListener("click", uploadImage);
