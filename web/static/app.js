const singleFile = document.getElementById("single-file");
const btnPredict = document.getElementById("btn-predict");
const singlePreview = document.getElementById("single-preview");
const singleResult = document.getElementById("single-result");
const btnCamera = document.getElementById("btn-camera");
const btnCapture = document.getElementById("btn-capture");
const cameraStatus = document.getElementById("camera-status");
const cameraPanel = document.getElementById("camera-panel");
const cameraVideo = document.getElementById("camera-video");
const cameraCanvas = document.getElementById("camera-canvas");

const fileA = document.getElementById("file-a");
const fileB = document.getElementById("file-b");
const btnCompare = document.getElementById("btn-compare");
const previewA = document.getElementById("preview-a");
const previewB = document.getElementById("preview-b");
const compareResult = document.getElementById("compare-result");
const samplesEl = document.getElementById("samples");

let selectedSampleUrl = null;
let capturedCameraFile = null;
let cameraStream = null;

function showPreview(container, file) {
  if (!file) {
    container.innerHTML = "";
    return;
  }
  const url = URL.createObjectURL(file);
  container.innerHTML = `<img src="${url}" alt="preview" />`;
}

singleFile.addEventListener("change", () => {
  showPreview(singlePreview, singleFile.files[0]);
  selectedSampleUrl = null;
  capturedCameraFile = null;
});

fileA.addEventListener("change", () => showPreview(previewA, fileA.files[0]));
fileB.addEventListener("change", () => showPreview(previewB, fileB.files[0]));

async function predictFromFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/predict", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

btnCamera.addEventListener("click", async () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    cameraVideo.srcObject = null;
    cameraPanel.classList.add("hidden");
    btnCamera.textContent = "Start camera";
    btnCapture.disabled = true;
    cameraStatus.textContent = "Camera is off.";
    return;
  }

  cameraStatus.textContent = "Requesting camera permission...";
  singleResult.textContent = "Your browser may ask for camera permission.";

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    cameraStatus.textContent =
      "Camera unavailable: open this page at http://localhost:8000 or http://127.0.0.1:8000 in a regular browser.";
    singleResult.textContent =
      "Camera access requires a secure browser context. File upload still works.";
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraPanel.classList.remove("hidden");
    btnCamera.textContent = "Stop camera";
    btnCapture.disabled = false;
    cameraStatus.textContent = "Camera is on. Center your head in the guide box.";
    singleResult.textContent = "Camera ready. Center your head in the guide box, then capture.";
  } catch (e) {
    cameraStatus.textContent = `Camera error: ${e.name || "Error"} - ${e.message}`;
    singleResult.textContent = "Camera did not start. You can still upload an image.";
  }
});

btnCapture.addEventListener("click", async () => {
  if (!cameraStream || !cameraVideo.videoWidth) {
    cameraStatus.textContent = "Camera is not ready yet.";
    singleResult.textContent = "Camera is not ready yet.";
    return;
  }

  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

  cameraCanvas.toBlob((blob) => {
    if (!blob) {
      singleResult.textContent = "Could not capture image from camera.";
      return;
    }
    capturedCameraFile = new File([blob], "webcam-capture.jpg", {
      type: "image/jpeg",
    });
    selectedSampleUrl = null;
    singleFile.value = "";
    showPreview(singlePreview, capturedCameraFile);
    cameraStatus.textContent = "Captured a webcam image.";
    singleResult.textContent = "Captured webcam image. Click Predict BMI.";
  }, "image/jpeg", 0.92);
});

btnPredict.addEventListener("click", async () => {
  singleResult.textContent = "Predicting…";
  btnPredict.disabled = true;
  try {
    let file = singleFile.files[0] || capturedCameraFile;
    if (!file && selectedSampleUrl) {
      const blob = await fetch(selectedSampleUrl).then((r) => r.blob());
      file = new File([blob], "sample.bmp", { type: blob.type });
    }
    if (!file) {
      singleResult.textContent = "Please upload an image or select a sample.";
      return;
    }
    const data = await predictFromFile(file);
    singleResult.innerHTML = `
      <strong>Predicted BMI:</strong> ${data.predicted_bmi}<br/>
      <strong>Category:</strong> ${data.bmi_category}
    `;
  } catch (e) {
    singleResult.textContent = `Error: ${e.message}`;
  } finally {
    btnPredict.disabled = false;
  }
});

btnCompare.addEventListener("click", async () => {
  const a = fileA.files[0];
  const b = fileB.files[0];
  if (!a || !b) {
    compareResult.textContent = "Please select both images.";
    return;
  }
  compareResult.textContent = "Comparing…";
  btnCompare.disabled = true;
  try {
    const form = new FormData();
    form.append("file_a", a);
    form.append("file_b", b);
    const res = await fetch("/api/compare", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const label =
      data.heavier_image === "tie"
        ? "Tie (equal predicted BMI)"
        : `Image ${data.heavier_image} predicted higher BMI`;
    compareResult.innerHTML = `
      <p><strong>${label}</strong> (Δ = ${data.bmi_difference})</p>
      <p>A: BMI ${data.image_a.predicted_bmi} (${data.image_a.bmi_category})</p>
      <p>B: BMI ${data.image_b.predicted_bmi} (${data.image_b.bmi_category})</p>
    `;
  } catch (e) {
    compareResult.textContent = `Error: ${e.message}`;
  } finally {
    btnCompare.disabled = false;
  }
});

async function loadSamples() {
  try {
    const res = await fetch("/api/samples?n=12");
    const samples = await res.json();
    samplesEl.innerHTML = samples
      .map(
        (s) => `
        <div class="sample-card" data-url="${s.url}" data-name="${s.name}">
          <img src="${s.url}" alt="${s.name}" />
          <div class="meta">BMI ${s.bmi.toFixed(1)} · ${s.gender}</div>
        </div>
      `
      )
      .join("");

    samplesEl.querySelectorAll(".sample-card").forEach((card) => {
      card.addEventListener("click", async () => {
        selectedSampleUrl = card.dataset.url;
        const img = card.querySelector("img");
        singlePreview.innerHTML = `<img src="${img.src}" alt="sample" />`;
        singleFile.value = "";
        singleResult.textContent = `Loaded ${card.dataset.name}. Click Predict BMI.`;
      });
    });
  } catch (e) {
    samplesEl.textContent = `Could not load samples: ${e.message}`;
  }
}

loadSamples();
