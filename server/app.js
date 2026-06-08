function uploadImage() {
  const fileInput = document.getElementById("imageUpload");
  const result = document.getElementById("result");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image!");
    return;
  }

  const reader = new FileReader();
  reader.onloadend = function () {
    const base64data = reader.result;
    result.innerHTML = `<div class="text-light">Classifying image...</div>`;

    const formData = new FormData();
    formData.append("image_data", base64data);

    fetch("/classify_image", {
      method: "POST",
      body: formData,
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Could not classify image.");
        }
        return data;
      })
      .then((data) => {
        if (!Array.isArray(data) || data.length === 0) {
          result.innerHTML =
            `<div class="alert alert-warning">No clear face with two eyes was detected. Try another image.</div>`;
          return;
        }

        const topClass = data[0].class;
        let html = `<h5>Predicted: <span class="text-warning">${topClass}</span></h5>`;
        html += `<p class="text-light">Class probabilities:</p>`;
        html += `<ul class="list-group">`;

        Object.entries(data[0].probabilities || {}).forEach(([cls, prob]) => {
          html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                    ${cls}
                    <span class="badge bg-primary rounded-pill">${(prob * 100).toFixed(2)}%</span>
                   </li>`;
        });

        html += `</ul>`;
        result.innerHTML = html;
      })
      .catch((err) => {
        console.error(err);
        result.innerHTML =
          `<div class="alert alert-danger">Error: ${err.message}</div>`;
      });
  };
  reader.readAsDataURL(file);
}
