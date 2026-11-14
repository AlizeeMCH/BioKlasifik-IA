import assert from "node:assert/strict";
import test from "node:test";
import {
  getCoverTransform,
  mapBoxToCanvas,
  placeLabel,
} from "../src/canvas-geometry.js";

test("maps portrait video coordinates through object-fit cover offsets", () => {
  const transform = getCoverTransform(480, 640, 320, 240);
  const box = mapBoxToCanvas(
    { x1: 120, y1: 0, x2: 360, y2: 640 },
    transform,
    320,
    240
  );

  assert.equal(transform.offsetX, 0);
  assert.ok(transform.offsetY < 0);
  assert.deepEqual(box, { x1: 80, y1: 38, x2: 240, y2: 228 });
});

test("clamps boxes to all four visible canvas margins", () => {
  const box = mapBoxToCanvas(
    { x1: -20, y1: -40, x2: 400, y2: 300 },
    { scale: 1, offsetX: 0, offsetY: 0 },
    320,
    240
  );

  assert.deepEqual(box, { x1: 12, y1: 38, x2: 308, y2: 228 });
});

test("keeps the top-edge label inside the canvas and within the box", () => {
  const label = placeLabel(
    { x1: 4, y1: 38, x2: 120, y2: 180 },
    180,
    24,
    320,
    240
  );

  assert.deepEqual(label, { x: 12, y: 46, width: 180 });
});

test("discards boxes fully outside the visible crop", () => {
  const box = mapBoxToCanvas(
    { x1: 100, y1: 0, x2: 200, y2: 20 },
    { scale: 1, offsetX: 0, offsetY: 0 },
    320,
    240
  );

  assert.equal(box, undefined);
});
