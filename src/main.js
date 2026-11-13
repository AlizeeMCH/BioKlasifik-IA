import { createDetectionStabilizer } from "./detection-stabilizer.js";
import {
  classifyLabel,
  classifyWastePredictions,
  estimateOrganicColorEvidence,
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
    captureCurrentFrame();
    const objects = await objectDetector.detect(frameCanvas, 10, 0.15);
    if (!running) {
      return;
    }
    const selectedObject = selectPrimaryObject(objects);
    const roi = cropObject(selectedObject);
    const mobilePredictions = await mobileNet.classify(roiCanvas, 5);
    if (!running) {
      return;
    }
    const confidenceThreshold = Number(confidenceInput.value) / 100;
    const colorEvidence = estimateOrganicColorEvidence(
      roiContext.getImageData(0, 0, roiCanvas.width, roiCanvas.height)
    );
    const classification = classifyWastePredictions(
      mobilePredictions,
      selectedObject,
      colorEvidence,
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
              ...roi,
              centerX: (roi.x1 + roi.x2) / 2 / video.videoWidth,
              centerY: (roi.y1 + roi.y2) / 2 / video.videoHeight,
              centerDistance: Math.hypot(
                ((roi.x1 + roi.x2) / 2 / video.videoWidth - 0.5) * 2,
                ((roi.y1 + roi.y2) / 2 / video.videoHeight - 0.5) * 2
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

  const paddingX = boxWidth * 0.08;
  const paddingY = boxHeight * 0.08;
  const left = Math.max(0, x - paddingX);
  const top = Math.max(0, y - paddingY);
  const right = Math.min(width, x + boxWidth + paddingX);
  const bottom = Math.min(height, y + boxHeight + paddingY);
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

  return { x1: left, y1: top, x2: right, y2: bottom };
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
      detection.y1 < 32
        ? detection.y1 + overlayContext.lineWidth
        : detection.y1 - labelHeight;
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
