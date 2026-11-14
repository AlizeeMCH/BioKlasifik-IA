import { createDetectionStabilizer } from "./detection-stabilizer.js";
import {
  getCoverTransform,
  mapBoxToCanvas,
  placeLabel,
} from "./canvas-geometry.js";
import {
  analyzeObjectAppearance,
  classifyLabel,
  classifyWastePredictions,
} from "./vision-classifier.js";
import "./style.css";

const MAX_ROI_SIZE = 320;
const INFERENCE_INTERVAL_MS = 350;
const CENTER_REGION_LIMIT = 0.72;
const CATEGORY_COLORS = {
  "Basura Orgánica": "#10b981",
  "Basura Inorgánica": "#3b82f6",
};

const video = document.querySelector("#camera");
const pythonStream = document.querySelector("#pythonStream");
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
const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d");
const roiCanvas = document.createElement("canvas");
const roiContext = roiCanvas.getContext("2d", { willReadFrequently: true });
const detectionStabilizer = createDetectionStabilizer();

let tf;
let objectDetector;
let mobileNet;
let cameraStream;
let running = false;
let animationFrameId;
let lastInferenceAt = 0;
let isProcessing = false;
let pythonMode = false;
let pythonRunning = false;
let pythonStatusTimer;
const pythonBackendReady = detectPythonBackend();

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = `${confidenceInput.value}%`;
  if (pythonRunning) {
    void setPythonConfidence();
  }
});

cameraButton.addEventListener("click", async () => {
  if (await pythonBackendReady) {
    if (running) {
      stopCamera();
    } else {
      await startPythonCamera();
    }
    return;
  }
  if (running) {
    stopCamera();
    return;
  }
  await startCamera();
});

async function loadModels() {
  statusMessage.textContent = "Preparando TensorFlow.js y los modelos visuales.";
  const [tensorflow, cocoSsd, mobilenet] = await Promise.all([
    import("@tensorflow/tfjs"),
    import("@tensorflow-models/coco-ssd"),
    import("@tensorflow-models/mobilenet"),
  ]);
  tf = tensorflow;
  try {
    if (!(await tf.setBackend("webgl"))) {
      throw new Error("WebGL no está disponible.");
    }
  } catch (error) {
    if (!(await tf.setBackend("cpu"))) {
      throw new Error(`No se pudo iniciar TensorFlow.js: ${error.message}`);
    }
  }
  await tf.ready();
  statusMessage.textContent =
    "Descargando COCO-SSD y MobileNetV2; la primera carga puede tardar.";
  objectDetector = await cocoSsd.load({ base: "lite_mobilenet_v2" });
  try {
    mobileNet = await mobilenet.load({ version: 2, alpha: 1.0 });
  } catch (error) {
    objectDetector.dispose();
    objectDetector = undefined;
    throw new Error(`No se pudo cargar MobileNetV2: ${error.message}`);
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
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 640, max: 960 },
        height: { ideal: 480, max: 720 },
      },
    });
    video.srcObject = cameraStream;
    await video.play();

    if (!objectDetector || !mobileNet) {
      await loadModels();
    }

    running = true;
    cameraButton.textContent = "Detener cámara";
    cameraButton.classList.add("stop-button");
    cameraButton.disabled = false;
    placeholder.hidden = true;
    liveIndicator.textContent = "EN VIVO";
    liveIndicator.classList.add("active");
    statusMessage.textContent =
      "Cámara activa. Mantén un objeto centrado en el encuadre.";
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

async function detectPythonBackend() {
  if (!["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    return false;
  }
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      return false;
    }
    const status = await response.json();
    pythonMode = status.python_backend === true;
    if (pythonMode) {
      statusMessage.textContent =
        "Servidor Python detectado. La cámara y los modelos se ejecutan localmente.";
    }
    return pythonMode;
  } catch {
    return false;
  }
}

async function startPythonCamera() {
  cameraButton.disabled = true;
  statusMessage.classList.remove("error");
  try {
    const response = await fetch("/api/start", { method: "POST" });
    const status = await response.json();
    if (!response.ok) {
      throw new Error(status.error || status.message || "No se pudo iniciar Python.");
    }

    running = true;
    pythonRunning = true;
    video.hidden = true;
    overlay.hidden = true;
    pythonStream.hidden = false;
    pythonStream.src = "/video_feed";
    placeholder.hidden = true;
    cameraButton.textContent = "Detener cámara";
    cameraButton.classList.add("stop-button");
    cameraButton.disabled = false;
    liveIndicator.textContent = "INICIANDO PYTHON";
    liveIndicator.classList.add("active");
    pythonStatusTimer = window.setInterval(
      () => void updatePythonStatus(),
      500
    );
    await updatePythonStatus();
  } catch (error) {
    cameraButton.disabled = false;
    statusMessage.textContent = error.message || "No se pudo iniciar Python.";
    statusMessage.classList.add("error");
  }
}

async function updatePythonStatus() {
  if (!pythonRunning) {
    return;
  }
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    if (!response.ok) {
      throw new Error(status.error || "No se pudo consultar el motor Python.");
    }
    if (status.state === "error") {
      statusMessage.textContent = status.message;
      statusMessage.classList.add("error");
      stopCamera();
      return;
    }
    statusMessage.textContent = status.message;
    statusMessage.classList.remove("error");
    liveIndicator.textContent =
      status.state === "listo" ? "EN VIVO · PYTHON" : "INICIANDO PYTHON";
    renderResults(
      status.detection
        ? [
            {
              category: status.detection.category,
              score: status.detection.confidence,
            },
          ]
        : []
    );
  } catch (error) {
    statusMessage.textContent =
      error.message || "Se perdió la conexión con el servidor Python.";
    statusMessage.classList.add("error");
  }
}

async function setPythonConfidence() {
  try {
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confidence: Number(confidenceInput.value) / 100,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "No se pudo actualizar la confianza.");
    }
  } catch (error) {
    statusMessage.textContent =
      error.message || "No se pudo actualizar la confianza en Python.";
    statusMessage.classList.add("error");
  }
}

async function stopPythonBackend() {
  try {
    const response = await fetch("/api/stop", { method: "POST" });
    if (!response.ok) {
      throw new Error("No se pudo detener la cámara Python.");
    }
  } catch (error) {
    statusMessage.textContent =
      error.message || "No se pudo detener la cámara Python.";
    statusMessage.classList.add("error");
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
  const stopPython = pythonRunning;
  running = false;
  pythonRunning = false;
  if (pythonStatusTimer !== undefined) {
    window.clearInterval(pythonStatusTimer);
    pythonStatusTimer = undefined;
  }
  detectionStabilizer.reset();
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  video.hidden = false;
  pythonStream.hidden = true;
  pythonStream.removeAttribute("src");
  overlay.hidden = false;
  cameraButton.textContent = "Activar cámara";
  cameraButton.classList.remove("stop-button");
  cameraButton.disabled = false;
  placeholder.hidden = false;
  liveIndicator.textContent = "INACTIVA";
  liveIndicator.classList.remove("active");
  const overlaySize = resizeOverlay();
  if (overlaySize) {
    overlayContext.clearRect(0, 0, overlaySize.width, overlaySize.height);
  }
  resultList.replaceChildren();
  resultEmpty.hidden = false;
  if (stopPython) {
    void stopPythonBackend();
  }
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
    captureCurrentFrame();
    const objects = await objectDetector.detect(frameCanvas, 10, 0.15);
    if (!running) {
      return;
    }
    const selectedObject = selectPrimaryObject(objects);
    const { box, colorCrop } = cropObject(selectedObject);
    const mobilePredictions = await mobileNet.classify(roiCanvas, 5);
    if (!running) {
      return;
    }
    const confidenceThreshold = Number(confidenceInput.value) / 100;
    const appearance = analyzeObjectAppearance(
      roiContext.getImageData(
        colorCrop.x,
        colorCrop.y,
        colorCrop.width,
        colorCrop.height
      )
    );
    const classification = classifyWastePredictions(
      mobilePredictions,
      selectedObject,
      appearance,
      confidenceThreshold
    );
    if (!running) {
      return;
    }
    const detections = detectionStabilizer.stabilize(
      classification
        ? [
            {
              ...classification,
              ...box,
              centerX: (box.x1 + box.x2) / 2 / video.videoWidth,
              centerY: (box.y1 + box.y2) / 2 / video.videoHeight,
              centerDistance: Math.hypot(
                ((box.x1 + box.x2) / 2 / video.videoWidth - 0.5) * 2,
                ((box.y1 + box.y2) / 2 / video.videoHeight - 0.5) * 2
              ),
            },
          ]
        : []
    );
    drawDetections(detections);
    renderResults(detections);
    statusMessage.textContent =
      "Análisis local activo; las imágenes no salen de tu dispositivo.";
  } catch (error) {
    statusMessage.textContent = error.message || "Ocurrió un error durante el análisis.";
    statusMessage.classList.add("error");
    stopCamera();
  } finally {
    isProcessing = false;
  }
}

function captureCurrentFrame() {
  if (
    frameCanvas.width !== video.videoWidth ||
    frameCanvas.height !== video.videoHeight
  ) {
    frameCanvas.width = video.videoWidth;
    frameCanvas.height = video.videoHeight;
  }
  frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
}

function selectPrimaryObject(predictions) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  return predictions
    .filter((prediction) => prediction.class !== "person")
    .map((prediction) => {
      const [x, y, boxWidth, boxHeight] = prediction.bbox;
      const centerX = (x + boxWidth / 2) / width;
      const centerY = (y + boxHeight / 2) / height;
      return {
        ...prediction,
        centerDistance: Math.hypot((centerX - 0.5) * 2, (centerY - 0.5) * 2),
      };
    })
    .filter((prediction) => prediction.centerDistance <= CENTER_REGION_LIMIT)
    .sort(
      (first, second) =>
        first.centerDistance - second.centerDistance ||
        second.score - first.score
    )[0];
}

function cropObject(detection) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  let [x, y, boxWidth, boxHeight] = detection?.bbox || [];
  if (!detection) {
    const side = Math.min(width, height) * 0.72;
    x = (width - side) / 2;
    y = (height - side) / 2;
    boxWidth = side;
    boxHeight = side;
  }

  const boxX1 = Math.max(0, x);
  const boxY1 = Math.max(0, y);
  const boxX2 = Math.min(width, x + boxWidth);
  const boxY2 = Math.min(height, y + boxHeight);
  boxWidth = Math.max(1, boxX2 - boxX1);
  boxHeight = Math.max(1, boxY2 - boxY1);
  const paddingX = boxWidth * 0.08;
  const paddingY = boxHeight * 0.08;
  const left = Math.max(0, boxX1 - paddingX);
  const top = Math.max(0, boxY1 - paddingY);
  const right = Math.min(width, boxX2 + paddingX);
  const bottom = Math.min(height, boxY2 + paddingY);
  const cropWidth = Math.max(1, right - left);
  const cropHeight = Math.max(1, bottom - top);
  const scale = Math.min(1, MAX_ROI_SIZE / Math.max(cropWidth, cropHeight));
  roiCanvas.width = Math.max(1, Math.round(cropWidth * scale));
  roiCanvas.height = Math.max(1, Math.round(cropHeight * scale));
  roiContext.drawImage(
    frameCanvas,
    left,
    top,
    cropWidth,
    cropHeight,
    0,
    0,
    roiCanvas.width,
    roiCanvas.height
  );

  const colorX = Math.max(
    0,
    Math.min(
      roiCanvas.width - 1,
      Math.round((boxX1 + boxWidth * 0.2 - left) * scale)
    )
  );
  const colorY = Math.max(
    0,
    Math.min(
      roiCanvas.height - 1,
      Math.round((boxY1 + boxHeight * 0.2 - top) * scale)
    )
  );
  return {
    box: { x1: boxX1, y1: boxY1, x2: boxX2, y2: boxY2 },
    colorCrop: {
      x: colorX,
      y: colorY,
      width: Math.max(
        1,
        Math.min(
          roiCanvas.width - colorX,
          Math.round(boxWidth * 0.6 * scale)
        )
      ),
      height: Math.max(
        1,
        Math.min(
          roiCanvas.height - colorY,
          Math.round(boxHeight * 0.6 * scale)
        )
      ),
    },
  };
}

function drawDetections(detections) {
  const overlaySize = resizeOverlay();
  if (!overlaySize || !video.videoWidth || !video.videoHeight) {
    return;
  }
  const { width, height } = overlaySize;
  const transform = getCoverTransform(
    video.videoWidth,
    video.videoHeight,
    width,
    height
  );
  overlayContext.clearRect(0, 0, width, height);
  overlayContext.lineWidth = Math.max(2, width / 320);
  overlayContext.font = `600 ${Math.max(14, width / 42)}px Inter, sans-serif`;

  for (const detection of detections) {
    const box = mapBoxToCanvas(detection, transform, width, height);
    if (!box) {
      continue;
    }
    const color = CATEGORY_COLORS[detection.category];
    const label = `${detection.category} · ${Math.round(detection.score * 100)}%`;
    const labelHeight = Math.max(24, width / 25);
    const labelPosition = placeLabel(
      box,
      overlayContext.measureText(label).width + 16,
      labelHeight,
      width,
      height
    );
    overlayContext.strokeStyle = color;
    overlayContext.fillStyle = color;
    overlayContext.strokeRect(
      box.x1,
      box.y1,
      box.x2 - box.x1,
      box.y2 - box.y1
    );
    overlayContext.fillStyle = color;
    overlayContext.fillRect(
      labelPosition.x,
      labelPosition.y,
      labelPosition.width,
      labelHeight
    );
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(
      label,
      labelPosition.x + 8,
      labelPosition.y + labelHeight * 0.68,
      Math.max(0, labelPosition.width - 16)
    );
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
  const width = overlay.clientWidth;
  const height = overlay.clientHeight;
  if (!width || !height) {
    return undefined;
  }

  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const backingWidth = Math.round(width * pixelRatio);
  const backingHeight = Math.round(height * pixelRatio);
  if (overlay.width !== backingWidth || overlay.height !== backingHeight) {
    overlay.width = backingWidth;
    overlay.height = backingHeight;
  }
  overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return { width, height };
}

video.addEventListener("loadedmetadata", resizeOverlay);
window.addEventListener("resize", resizeOverlay);
window.addEventListener("beforeunload", () => {
  if (pythonRunning) {
    navigator.sendBeacon("/api/stop");
    pythonRunning = false;
  }
  stopCamera();
});
