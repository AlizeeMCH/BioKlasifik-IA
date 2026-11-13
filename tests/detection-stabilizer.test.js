import assert from "node:assert/strict";
import test from "node:test";
import { createDetectionStabilizer } from "../src/detection-stabilizer.js";

function detection(category, score = 0.56, centerX = 0.5) {
  return {
    category,
    score,
    x1: centerX * 100 - 20,
    y1: 30,
    x2: centerX * 100 + 20,
    y2: 70,
    centerX,
    centerY: 0.5,
    centerDistance: Math.abs(centerX - 0.5) * 2,
  };
}

test("keeps the active category through isolated near-tie predictions", () => {
  const stabilizer = createDetectionStabilizer();
  for (let frame = 0; frame < 3; frame += 1) {
    stabilizer.stabilize([detection("Basura Orgánica", 0.57)]);
  }

  const result = stabilizer.stabilize([detection("Basura Inorgánica", 0.55)]);

  assert.equal(result[0].category, "Basura Orgánica");
  assert.equal(result[0].score, 0.57);
});

test("waits for repeated evidence before showing an initial category", () => {
  const stabilizer = createDetectionStabilizer();

  assert.deepEqual(
    stabilizer.stabilize([detection("Basura Inorgánica")]),
    []
  );
  assert.deepEqual(
    stabilizer.stabilize([detection("Basura Orgánica")]),
    []
  );
  assert.deepEqual(
    stabilizer.stabilize([detection("Basura Orgánica")]),
    []
  );
  assert.equal(
    stabilizer.stabilize([detection("Basura Orgánica")])[0].category,
    "Basura Orgánica"
  );
});

test("switches after four of six frames consistently support the other category", () => {
  const stabilizer = createDetectionStabilizer();
  for (let frame = 0; frame < 3; frame += 1) {
    stabilizer.stabilize([detection("Basura Orgánica")]);
  }

  for (let frame = 0; frame < 3; frame += 1) {
    assert.equal(
      stabilizer.stabilize([detection("Basura Inorgánica")])[0].category,
      "Basura Orgánica"
    );
  }

  assert.equal(
    stabilizer.stabilize([detection("Basura Inorgánica")])[0].category,
    "Basura Inorgánica"
  );
});

test("resets category history when the centered object changes", () => {
  const stabilizer = createDetectionStabilizer();
  for (let frame = 0; frame < 3; frame += 1) {
    stabilizer.stabilize([detection("Basura Orgánica")]);
  }

  stabilizer.stabilize([detection("Basura Inorgánica", 0.7, 0.9)]);
  stabilizer.stabilize([detection("Basura Inorgánica", 0.7, 0.9)]);
  const result = stabilizer.stabilize([
    detection("Basura Inorgánica", 0.7, 0.9),
  ]);

  assert.equal(result[0].category, "Basura Inorgánica");
});
