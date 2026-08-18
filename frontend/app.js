const API_URL = "/api";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const cameraButton = document.getElementById("cameraButton");
const scanButton = document.getElementById("scanButton");
const fileInput = document.getElementById("fileInput");
const cameraMessage = document.getElementById("cameraMessage");
const cameraStatus = document.getElementById("cameraStatus");
const enrollmentForm = document.getElementById("enrollmentForm");
const enrollmentMessage = document.getElementById("enrollmentMessage");
const enrollButton = document.getElementById("enrollButton");

function formatCount(count, label) {
  return `${count} ${label}${count === 1 ? "" : "s"}`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.message || "Something went wrong. Please try again.");
  }
  return data;
}

async function loadDashboard() {
  try {
    const [health, attendance] = await Promise.all([
      fetchJson(`${API_URL}/health`),
      fetchJson(`${API_URL}/attendance`),
    ]);
    document.getElementById("peopleCount").textContent = health.registered_people;
    const samples = health.people.reduce((total, person) => total + person.samples, 0);
    document.getElementById("sampleCount").textContent = formatCount(samples, "face sample");
    renderAttendance(attendance.records);
  } catch {
    document.getElementById("sampleCount").textContent = "Service unavailable";
  }
}

function renderAttendance(records) {
  const attendanceList = document.getElementById("attendanceList");
  if (!records.length) {
    attendanceList.innerHTML = '<p class="muted">No attendance recorded yet.</p>';
    return;
  }

  attendanceList.replaceChildren(
    ...records.map((record) => {
      const item = document.createElement("article");
      item.className = "attendance-item";
      const name = document.createElement("strong");
      name.textContent = record.Name;
      const details = document.createElement("span");
      details.textContent = `${record["Date and Time"]} · ${record.Confidence}`;
      item.append(name, details);
      return item;
    }),
  );
}

cameraButton.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    cameraMessage.hidden = true;
    cameraStatus.textContent = "Camera ready";
    cameraStatus.classList.add("is-active");
    cameraButton.textContent = "Camera active";
    cameraButton.disabled = true;
    scanButton.disabled = false;
  } catch {
    cameraMessage.hidden = false;
    cameraMessage.textContent = "Camera access was not granted. You can still upload a photo.";
  }
});

scanButton.addEventListener("click", () => {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => blob && sendImageToApi(blob), "image/jpeg", 0.9);
});

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) sendImageToApi(file);
  event.target.value = "";
});

async function sendImageToApi(image) {
  scanButton.disabled = true;
  scanButton.textContent = "Scanning…";
  const formData = new FormData();
  formData.append("image", image, "face.jpg");

  try {
    const result = await fetchJson(`${API_URL}/recognize`, { method: "POST", body: formData });
    showResult(result);
    if (result.match) loadDashboard();
  } catch (error) {
    showResult({ match: false, message: error.message, detected_faces: 0 });
  } finally {
    scanButton.disabled = !video.srcObject;
    scanButton.innerHTML = 'Scan identity <span aria-hidden="true">→</span>';
  }
}

function showResult(result) {
  const matched = result.match === true;
  document.getElementById("waitingState").classList.add("hidden");
  document.getElementById("resultState").classList.remove("hidden");
  const name = matched ? result.name : "No match";
  document.getElementById("personName").textContent = name;
  document.getElementById("resultLabel").textContent = matched ? "Identity verified" : "Scan needs attention";
  document.getElementById("resultMessage").textContent = result.message || "";
  const confidence = result.confidence ?? 0;
  document.getElementById("confidenceText").textContent = result.confidence !== undefined ? `${confidence}%` : "—";
  document.getElementById("progressBar").style.width = `${confidence}%`;
  document.getElementById("faceCount").textContent = result.detected_faces ?? 0;
  document.getElementById("statusText").textContent = matched ? "Recorded" : "Not verified";
  document.getElementById("initials").textContent = matched
    ? name.split(" ").map((part) => part[0]).join("").slice(0, 2).toUpperCase()
    : "?";
}

enrollmentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  enrollButton.disabled = true;
  enrollButton.textContent = "Adding person…";
  enrollmentMessage.textContent = "";

  try {
    const result = await fetchJson(`${API_URL}/people`, {
      method: "POST",
      body: new FormData(enrollmentForm),
    });
    enrollmentForm.reset();
    enrollmentMessage.className = "form-message success";
    enrollmentMessage.textContent = result.message;
    loadDashboard();
  } catch (error) {
    enrollmentMessage.className = "form-message error";
    enrollmentMessage.textContent = error.message;
  } finally {
    enrollButton.disabled = false;
    enrollButton.textContent = "Add to recognition";
  }
});

loadDashboard();
