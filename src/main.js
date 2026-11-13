import * as ort from "onnxruntime-web/webgpu";
import { createDetectionStabilizer } from "./detection-stabilizer.js";
import "./style.css";

const INPUT_SIZE = 320;
const INFERENCE_INTERVAL_MS = 225;
const CENTER_REGION_LIMIT = 0.72;
const MODEL_URL = "/models/waste-yolov8n-12cls-320-fp16.onnx";
const CLASS_NAMES = [
  "battery",
  "biological",
  "brown-glass",
  "cardboard",
  "clothes",
  "green-glass",
  "metal",
  "paper",
  "plastic",
  "shoes",
  "trash",
  "white-glass",
];
const ORGANIC_CLASSES = new Set([
  "biological",
  "food",
  "food scraps",
  "food waste",
  "fruit",
  "fruits",
  "organic",
  "organic waste",
  "peel",
  "plant",
  "plants",
  "vegetable",
  "vegetables",
]);
const CATEGORY_COLORS = {
  "Basura Orgánica": "#10b981",
  "Basura Inorgánica": "#3b82f6",
};
const CLASS_THRESHOLDS = {
  clothes: 0.65,
  "brown-glass": 0.55,
  "green-glass": 0.55,
  "white-glass": 0.55,
};

const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const placeholder = document.querySelector("#cameraPlaceholder");
const cameraButton = document.querySelector("#cameraButton");
const confidenceInput = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidenceValue");
const statusMessage = document.querySelector("#statusMessage");
const liveIndicator = document.querySelector("#liveIndicator");
const resultList = document.querySelector("#resultList");
const resultEmpty = document.querySelector("#resultEmpty");
const overlayContext = overlay.getContext("2d");
const inputCanvas = document.createElement("canvas");
inputCanvas.width = INPUT_SIZE;
inputCanvas.height = INPUT_SIZE;
const inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });
const detectionStabilizer = createDetectionStabilizer();

let session;
let cameraStream;
let running = false;
let animationFrameId;
let lastInferenceAt = 0;
let isProcessing = false;
let inferenceProvider = "wasm";

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = `${confidenceInput.value}%`;
});

cameraButton.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    return;
  }
  await startCamera();
});

async function loadModel() {
  statusMessage.textContent =
    "Preparando el modelo; la primera vez se descargará en este dispositivo.";
  let gpuError;

  if (navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      inferenceProvider = "webgpu";
      return;
    } catch (error) {
      gpuError = error;
      statusMessage.textContent =
        "La aceleración gráfica no está disponible; preparando el modo compatible.";
    }
  }

  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    inferenceProvider = "wasm";
  } catch (error) {
    const gpuMessage = gpuError?.message || "WebGPU no disponible";
    throw new Error(
      `No se pudo cargar el modelo. GPU: ${gpuMessage}. Procesador: ${error.message}`
    );
  }
}

async function startCamera() {
  cameraButton.disabled = true;
  statusMessage.classList.remove("error");
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "No se puede acceder a la cámara. Abre esta aplicación con HTTPS en un navegador actualizado."
      );
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 } },
    });
    video.srcObject = cameraStream;
    await video.play();

    if (!session) {
      await loadModel();
    }

    running = true;
    cameraButton.textContent = "Detener cámara";
    cameraButton.classList.add("stop-button");
    cameraButton.disabled = false;
    placeholder.hidden = true;
    liveIndicator.textContent = "EN VIVO";
    liveIndicator.classList.add("active");
    statusMessage.textContent =
      inferenceProvider === "wasm"
        ? "Cámara activa. El análisis se ejecutará en el procesador de este dispositivo."
        : "Cámara activa. Mantén un objeto centrado en el encuadre.";
    resizeOverlay();
    detectionStabilizer.reset();
    lastInferenceAt = 0;
    animationFrameId = requestAnimationFrame(processFrame);
  } catch (error) {
    stopCamera();
    statusMessage.textContent = getStartupErrorMessage(error);
    statusMessage.classList.add("error");
    cameraButton.disabled = false;
  }
}

function getStartupErrorMessage(error) {
  if (error.name === "NotAllowedError" || error.name === "SecurityError") {
    return "No se autorizó el acceso a la cámara. Permite el acceso en los ajustes del navegador y vuelve a intentarlo.";
  }
  if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
    return "No se encontró una cámara compatible en este dispositivo.";
  }
  if (error.name === "NotReadableError" || error.name === "AbortError") {
    return "No se pudo iniciar la cámara. Cierra otras aplicaciones que la estén usando y vuelve a intentarlo.";
  }
  return error.message || "No se pudo iniciar el clasificador.";
}

function stopCamera() {
  running = false;
  detectionStabilizer.reset();
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  cameraButton.textContent = "Activar cámara";
  cameraButton.classList.remove("stop-button");
  cameraButton.disabled = false;
  placeholder.hidden = false;
  liveIndicator.textContent = "INACTIVA";
  liveIndicator.classList.remove("active");
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  resultList.replaceChildren();
  resultEmpty.hidden = false;
}

function processFrame(timestamp) {
  if (!running) {
    return;
  }

  if (
    !isProcessing &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    timestamp - lastInferenceAt >= INFERENCE_INTERVAL_MS
  ) {
    lastInferenceAt = timestamp;
    isProcessing = true;
    void inferFrame();
  }

  animationFrameId = requestAnimationFrame(processFrame);
}

async function inferFrame() {
  try {
    const { scale, padX, padY } = prepareInput();
    const input = new ort.Tensor("float32", inputPixels(), [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const output = await session.run({ images: input });
    if (!running) {
      return;
    }
    const detections = detectionStabilizer.stabilize(
      decodeDetections(output.output0, scale, padX, padY)
    );
    drawDetections(detections);
    renderResults(detections);
    statusMessage.textContent =
      inferenceProvider === "wasm"
        ? "Análisis local activo en el procesador; las imágenes no salen de tu dispositivo."
        : "Análisis local activo; las imágenes no salen de tu dispositivo.";
  } catch (error) {
    statusMessage.textContent = error.message || "Ocurrió un error durante el análisis.";
    statusMessage.classList.add("error");
    stopCamera();
  } finally {
    isProcessing = false;
  }
}

function prepareInput() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const padX = Math.floor((INPUT_SIZE - resizedWidth) / 2);
  const padY = Math.floor((INPUT_SIZE - resizedHeight) / 2);

  inputContext.fillStyle = "rgb(114, 114, 114)";
  inputContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  inputContext.drawImage(video, padX, padY, resizedWidth, resizedHeight);
  return { scale, padX, padY };
}

function inputPixels() {
  const { data } = inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const source = pixel * 4;
    pixels[pixel] = data[source] / 255;
    pixels[planeSize + pixel] = data[source + 1] / 255;
    pixels[planeSize * 2 + pixel] = data[source + 2] / 255;
  }
  return pixels;
}

function decodeDetections(data, scale, padX, padY) {
  const outputData = data.data;
  const classCount = data.dims[1] - 4;
  const detectionCount = data.dims[2];
  if (classCount !== CLASS_NAMES.length) {
    throw new Error("La salida del modelo no coincide con sus clases de residuos.");
  }

  const candidates = [];
  const minimumConfidence = Math.min(Number(confidenceInput.value), 75) / 100;

  for (let index = 0; index < detectionCount; index += 1) {
    let classId = 0;
    let score = 0;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const classScore = outputData[(classIndex + 4) * detectionCount + index];
      if (classScore > score) {
        score = classScore;
        classId = classIndex;
      }
    }
    const name = CLASS_NAMES[classId];
    const category = classifyCategory(name);
    if (score < Math.max(minimumConfidence, CLASS_THRESHOLDS[name] || 0)) {
      continue;
    }

    const centerX = outputData[index];
    const centerY = outputData[detectionCount + index];
    const width = outputData[2 * detectionCount + index];
    const height = outputData[3 * detectionCount + index];
    const left = Math.max(0, centerX - width / 2);
    const top = Math.max(0, centerY - height / 2);
    const right = Math.min(INPUT_SIZE, centerX + width / 2);
    const bottom = Math.min(INPUT_SIZE, centerY + height / 2);
    const x1 = Math.max(0, (left - padX) / scale);
    const y1 = Math.max(0, (top - padY) / scale);
    const x2 = Math.min(video.videoWidth, (right - padX) / scale);
    const y2 = Math.min(video.videoHeight, (bottom - padY) / scale);
    const area = (x2 - x1) * (y2 - y1);
    if (area <= 0 || area / (video.videoWidth * video.videoHeight) > 0.75) {
      continue;
    }
    const boxCenterX = (x1 + x2) / 2 / video.videoWidth;
    const boxCenterY = (y1 + y2) / 2 / video.videoHeight;
    const offsetX = (boxCenterX - 0.5) * 2;
    const offsetY = (boxCenterY - 0.5) * 2;
    if (
      Math.abs(offsetX) > CENTER_REGION_LIMIT ||
      Math.abs(offsetY) > CENTER_REGION_LIMIT
    ) {
      continue;
    }

    candidates.push({
      category,
      score,
      centerX: boxCenterX,
      centerY: boxCenterY,
      centerDistance: Math.hypot(offsetX, offsetY),
      x1,
      y1,
      x2,
      y2,
    });
  }

  return nonMaximumSuppression(candidates)
    .sort(
      (first, second) =>
        first.centerDistance - second.centerDistance || second.score - first.score
    )
    .slice(0, 1);
}

function classifyCategory(className) {
  const normalizedName = className.toLowerCase().replace(/[-_]+/g, " ");
  return ORGANIC_CLASSES.has(normalizedName)
    ? "Basura Orgánica"
    : "Basura Inorgánica";
}

function nonMaximumSuppression(candidates) {
  candidates.sort((first, second) => second.score - first.score);
  const selected = [];
  for (const candidate of candidates.slice(0, 300)) {
    if (selected.every((item) => intersectionOverUnion(item, candidate) <= 0.45)) {
      selected.push(candidate);
    }
  }
  return selected;
}

function intersectionOverUnion(first, second) {
  const intersectionWidth = Math.max(
    0,
    Math.min(first.x2, second.x2) - Math.max(first.x1, second.x1)
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(first.y2, second.y2) - Math.max(first.y1, second.y1)
  );
  const intersection = intersectionWidth * intersectionHeight;
  const firstArea = (first.x2 - first.x1) * (first.y2 - first.y1);
  const secondArea = (second.x2 - second.x1) * (second.y2 - second.y1);
  const union = firstArea + secondArea - intersection;
  return union > 0 ? intersection / union : 0;
}

function drawDetections(detections) {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlayContext.lineWidth = Math.max(2, overlay.width / 320);
  overlayContext.font = `600 ${Math.max(14, overlay.width / 42)}px Inter, sans-serif`;

  for (const detection of detections) {
    const color = CATEGORY_COLORS[detection.category];
    const label = `${detection.category} · ${Math.round(detection.score * 100)}%`;
    const labelHeight = Math.max(24, overlay.width / 25);
    const labelWidth = Math.min(
      overlay.width,
      overlayContext.measureText(label).width + 16
    );
    const labelX = Math.min(
      Math.max(0, detection.x1),
      overlay.width - labelWidth
    );
    const labelTop =
      detection.y1 >= labelHeight
        ? detection.y1 - labelHeight
        : detection.y1 + overlayContext.lineWidth;
    overlayContext.strokeStyle = color;
    overlayContext.fillStyle = color;
    overlayContext.strokeRect(
      detection.x1,
      detection.y1,
      detection.x2 - detection.x1,
      detection.y2 - detection.y1
    );
    overlayContext.fillStyle = color;
    overlayContext.fillRect(labelX, labelTop, labelWidth, labelHeight);
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(label, labelX + 8, labelTop + labelHeight * 0.68);
  }
}

function renderResults(detections) {
  resultList.replaceChildren(
    ...detections.map((detection) => {
      const item = document.createElement("li");
      const organic = detection.category === "Basura Orgánica";
      item.className = `result-item ${organic ? "organic-result" : "inorganic-result"}`;
      const dot = document.createElement("span");
      dot.className = `result-dot ${organic ? "organic-dot" : "inorganic-dot"}`;
      const text = document.createElement("span");
      text.className = "result-category";
      text.textContent = `${detection.category} · ${Math.round(detection.score * 100)}%`;
      item.append(dot, text);
      return item;
    })
  );
  resultEmpty.hidden = detections.length > 0;
}

function resizeOverlay() {
  if (video.videoWidth && video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
}

video.addEventListener("loadedmetadata", resizeOverlay);
window.addEventListener("beforeunload", stopCamera);
