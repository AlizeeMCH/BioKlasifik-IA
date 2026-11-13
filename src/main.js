import * as ort from "onnxruntime-web/webgpu";
import "./style.css";

const INPUT_SIZE = 640;
const CLASS_COUNT = 12;
const DETECTION_COUNT = 8400;
const MODEL_URL = "/models/waste-yolov8n-12cls-fp16.onnx";
const CLASSIFICATIONS = [
  ["battery", "Batería", "Basura Inorgánica"],
  ["biological", "Orgánico", "Basura Orgánica"],
  ["brown-glass", "Vidrio marrón", "Basura Inorgánica"],
  ["cardboard", "Cartón", "Basura Inorgánica"],
  ["clothes", "Residuo Inorgánico", "Basura Inorgánica"],
  ["green-glass", "Vidrio verde", "Basura Inorgánica"],
  ["metal", "Metal", "Basura Inorgánica"],
  ["paper", "Papel", "Basura Inorgánica"],
  ["plastic", "Plástico", "Basura Inorgánica"],
  ["shoes", "Calzado", "Basura Inorgánica"],
  ["trash", "Residuo mixto", "Basura Inorgánica"],
  ["white-glass", "Vidrio transparente", "Basura Inorgánica"],
];
const CLASS_THRESHOLDS = {
  clothes: 0.65,
  "brown-glass": 0.55,
  "green-glass": 0.55,
  "white-glass": 0.55,
};

const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const stage = document.querySelector("#cameraStage");
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

let session;
let cameraStream;
let running = false;
let timer;
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
    await processFrame();
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
  clearTimeout(timer);
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

async function processFrame() {
  if (!running || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    timer = setTimeout(processFrame, 250);
    return;
  }

  try {
    const { scale, padX, padY } = prepareInput();
    const input = new ort.Tensor("float32", inputPixels(), [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const output = (await session.run({ images: input })).output0;
    const detections = decodeDetections(output.data, scale, padX, padY);
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
    return;
  }

  timer = setTimeout(processFrame, 900);
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
  const candidates = [];
  const minimumConfidence = Number(confidenceInput.value) / 100;

  for (let index = 0; index < DETECTION_COUNT; index += 1) {
    let classId = 0;
    let score = 0;
    for (let classIndex = 0; classIndex < CLASS_COUNT; classIndex += 1) {
      const classScore = data[(classIndex + 4) * DETECTION_COUNT + index];
      if (classScore > score) {
        score = classScore;
        classId = classIndex;
      }
    }
    const [name, object, category] = CLASSIFICATIONS[classId];
    if (score < Math.max(minimumConfidence, CLASS_THRESHOLDS[name] || 0)) {
      continue;
    }

    const centerX = data[index];
    const centerY = data[DETECTION_COUNT + index];
    const width = data[2 * DETECTION_COUNT + index];
    const height = data[3 * DETECTION_COUNT + index];
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
    if (
      name === "biological" &&
      skinRatio(left, top, right, bottom) > 0.45
    ) {
      continue;
    }

    candidates.push({ name, object, category, score, x1, y1, x2, y2 });
  }

  return nonMaximumSuppression(candidates).slice(0, 10);
}

function skinRatio(left, top, right, bottom) {
  const x = Math.max(0, Math.floor(left));
  const y = Math.max(0, Math.floor(top));
  const width = Math.min(INPUT_SIZE - x, Math.ceil(right) - x);
  const height = Math.min(INPUT_SIZE - y, Math.ceil(bottom) - y);
  if (width <= 0 || height <= 0) {
    return 0;
  }

  const { data } = inputContext.getImageData(x, y, width, height);
  let skinPixels = 0;
  const pixelCount = width * height;
  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue;
    const cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue;
    if (cr >= 133 && cr <= 173 && cb >= 77 && cb <= 127) {
      skinPixels += 1;
    }
  }
  return skinPixels / pixelCount;
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
    const color = detection.category === "Basura Orgánica" ? "#42c58a" : "#5796ff";
    const label = `${detection.object} · ${Math.round(detection.score * 100)}%`;
    const labelHeight = Math.max(24, overlay.width / 25);
    const labelWidth = overlayContext.measureText(label).width + 16;
    const labelTop = Math.max(0, detection.y1 - labelHeight);
    overlayContext.strokeStyle = color;
    overlayContext.fillStyle = color;
    overlayContext.strokeRect(
      detection.x1,
      detection.y1,
      detection.x2 - detection.x1,
      detection.y2 - detection.y1
    );
    overlayContext.fillRect(detection.x1, labelTop, labelWidth, labelHeight);
    overlayContext.fillStyle = "#102019";
    overlayContext.fillText(label, detection.x1 + 8, labelTop + labelHeight * 0.68);
  }
}

function renderResults(detections) {
  resultList.replaceChildren(
    ...detections.map((detection) => {
      const item = document.createElement("li");
      item.className = "result-item";
      const dot = document.createElement("span");
      dot.className =
        detection.category === "Basura Orgánica"
          ? "result-dot organic-dot"
          : "result-dot inorganic-dot";
      const text = document.createElement("span");
      text.className = "result-label";
      const object = document.createElement("strong");
      object.textContent = detection.object;
      const category = document.createElement("small");
      category.textContent = detection.category;
      text.append(object, category);
      const score = document.createElement("span");
      score.className = "result-score";
      score.textContent = `${Math.round(detection.score * 100)}%`;
      item.append(dot, text, score);
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
