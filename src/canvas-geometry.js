const BOX_LEFT_MARGIN = 12;
const BOX_TOP_MARGIN = 38;
const BOX_RIGHT_MARGIN = 12;
const BOX_BOTTOM_MARGIN = 12;

export function getCoverTransform(
  sourceWidth,
  sourceHeight,
  canvasWidth,
  canvasHeight
) {
  const scale = Math.max(
    canvasWidth / sourceWidth,
    canvasHeight / sourceHeight
  );
  return {
    scale,
    offsetX: (canvasWidth - sourceWidth * scale) / 2,
    offsetY: (canvasHeight - sourceHeight * scale) / 2,
  };
}

export function mapBoxToCanvas(
  box,
  transform,
  canvasWidth,
  canvasHeight
) {
  const rawBox = {
    x1: box.x1 * transform.scale + transform.offsetX,
    y1: box.y1 * transform.scale + transform.offsetY,
    x2: box.x2 * transform.scale + transform.offsetX,
    y2: box.y2 * transform.scale + transform.offsetY,
  };
  if (
    rawBox.x2 <= BOX_LEFT_MARGIN ||
    rawBox.x1 >= canvasWidth - BOX_RIGHT_MARGIN ||
    rawBox.y2 <= BOX_TOP_MARGIN ||
    rawBox.y1 >= canvasHeight - BOX_BOTTOM_MARGIN
  ) {
    return undefined;
  }

  const mappedBox = {
    x1: Math.max(BOX_LEFT_MARGIN, rawBox.x1),
    y1: Math.max(BOX_TOP_MARGIN, rawBox.y1),
    x2: Math.min(canvasWidth - BOX_RIGHT_MARGIN, rawBox.x2),
    y2: Math.min(canvasHeight - BOX_BOTTOM_MARGIN, rawBox.y2),
  };
  return mappedBox.x2 > mappedBox.x1 && mappedBox.y2 > mappedBox.y1
    ? mappedBox
    : undefined;
}

export function placeLabel(box, labelWidth, labelHeight, canvasWidth, canvasHeight) {
  const safeLabelWidth = Math.min(labelWidth, canvasWidth - 24);
  const x = Math.min(
    Math.max(BOX_LEFT_MARGIN, box.x1),
    canvasWidth - BOX_RIGHT_MARGIN - safeLabelWidth
  );
  const preferredY =
    box.y1 <= BOX_TOP_MARGIN
      ? Math.max(BOX_LEFT_MARGIN, box.y1 + 8)
      : Math.max(BOX_LEFT_MARGIN, box.y1 - labelHeight);
  const maxY = canvasHeight - BOX_BOTTOM_MARGIN - labelHeight;

  return {
    x,
    y: Math.min(preferredY, maxY),
    width: safeLabelWidth,
  };
}
