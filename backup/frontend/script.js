const dropZoneElement = document.getElementById("drop-zone");
const inputElement = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const detectBtn = document.getElementById("detect-btn");
const loadingIndicator = document.getElementById("loading");
const resultsSection = document.getElementById("results-section");
const resultImage = document.getElementById("result-image");
const resultsBody = document.getElementById("results-body");
const errorMessage = document.getElementById("error-message");

let selectedFile = null;

// Drag and Drop Events
dropZoneElement.addEventListener("click", () => inputElement.click());

dropZoneElement.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZoneElement.classList.add("drop-zone--over");
});

["dragleave", "dragend"].forEach((type) => {
    dropZoneElement.addEventListener(type, (e) => {
        dropZoneElement.classList.remove("drop-zone--over");
    });
});

dropZoneElement.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZoneElement.classList.remove("drop-zone--over");

    if (e.dataTransfer.files.length) {
        inputElement.files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0]);
    }
});

inputElement.addEventListener("change", (e) => {
    if (inputElement.files.length) {
        handleFile(inputElement.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("Please select a valid image file.");
        return;
    }

    selectedFile = file;
    clearError();
    resultsSection.classList.add("hidden");

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
        imagePreview.src = reader.result;
        previewContainer.classList.remove("hidden");
        detectBtn.disabled = false;
    };
}

detectBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    detectBtn.disabled = true;
    loadingIndicator.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    clearError();

    try {
        const response = await fetch("http://127.0.0.1:8000/detect", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError("Failed to connect to the server. Ensure the backend is running.");
        console.error(error);
    } finally {
        detectBtn.disabled = false;
        loadingIndicator.classList.add("hidden");
    }
});

function displayResults(data) {
    resultImage.src = data.image_base64;
    resultsBody.innerHTML = "";

    if (data.detections.length === 0) {
        resultsBody.innerHTML = "<tr><td colspan='2' style='text-align:center;'>No objects detected.</td></tr>";
    } else {
        data.detections.forEach(det => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${det.class_name}</td>
                <td>${det.confidence}%</td>
            `;
            resultsBody.appendChild(row);
        });
    }

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(msg) {
    errorMessage.textContent = msg;
}

function clearError() {
    errorMessage.textContent = "";
}