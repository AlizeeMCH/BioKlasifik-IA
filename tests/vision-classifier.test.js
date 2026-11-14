import assert from "node:assert/strict";
import test from "node:test";
import {
  analyzeObjectAppearance,
  classifyLabel,
  classifyWastePredictions,
} from "../src/vision-classifier.js";

const NEUTRAL_APPEARANCE = {
  organicEvidence: 0,
  inorganicEvidence: 0,
  isHardInorganic: false,
};

test("recognizes fruit and produce labels without matching substrings", () => {
  assert.equal(classifyLabel("bell pepper"), "Basura Orgánica");
  assert.equal(classifyLabel("green jalapeno"), "Basura Orgánica");
  assert.equal(classifyLabel("chile jalapeño"), "Basura Orgánica");
  assert.equal(classifyLabel("dark avocado"), "Basura Orgánica");
  assert.equal(classifyLabel("tomato"), "Basura Orgánica");
  assert.equal(classifyLabel("plastic bag"), "Basura Inorgánica");
  assert.equal(classifyLabel("cantaloupe"), undefined);
});

test("prioritizes organic top-five evidence above fifteen percent", () => {
  const result = classifyWastePredictions(
    [{ className: "lemon", probability: 0.18 }],
    undefined,
    NEUTRAL_APPEARANCE,
    0.35
  );

  assert.equal(result.category, "Basura Orgánica");
  assert.ok(Math.abs(result.score - 0.18) < Number.EPSILON * 2);
});

test("keeps a strong container prediction over a weak color fallback", () => {
  const result = classifyWastePredictions(
    [{ className: "water bottle", probability: 0.8 }],
    { class: "bottle", score: 0.9 },
    { ...NEUTRAL_APPEARANCE, organicEvidence: 0.65 },
    0.35
  );

  assert.equal(result.category, "Basura Inorgánica");
});

test("recognizes cutlery-class predictions as inorganic instead of using color fallback", () => {
  const result = classifyWastePredictions(
    [{ className: "letter opener, paper knife", probability: 0.327 }],
    { class: "scissors", score: 0.36 },
    { ...NEUTRAL_APPEARANCE, organicEvidence: 0.59 },
    0.35
  );

  assert.equal(result.category, "Basura Inorgánica");
});

test("uses organic color evidence but ignores neutral specular highlights", () => {
  const width = 32;
  const height = 32;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    pixels[offset] = 25;
    pixels[offset + 1] = 85;
    pixels[offset + 2] = 20;
    pixels[offset + 3] = 255;
  }
  for (let y = 2; y < 6; y += 1) {
    for (let x = 2; x < 6; x += 1) {
      const offset = (y * width + x) * 4;
      pixels[offset] = 255;
      pixels[offset + 1] = 255;
      pixels[offset + 2] = 255;
    }
  }

  const appearance = analyzeObjectAppearance({
    data: pixels,
    width,
    height,
  });

  assert.ok(appearance.organicEvidence >= 0.3);
  assert.equal(appearance.isHardInorganic, false);
});

test("classifies a mostly white object as inorganic despite neutral highlights", () => {
  const width = 32;
  const height = 32;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    pixels[offset] = 248;
    pixels[offset + 1] = 248;
    pixels[offset + 2] = 248;
    pixels[offset + 3] = 255;
  }

  const appearance = analyzeObjectAppearance({ data: pixels, width, height });
  const result = classifyWastePredictions(
    [],
    undefined,
    appearance,
    0.35
  );

  assert.equal(appearance.isHardInorganic, true);
  assert.equal(result.category, "Basura Inorgánica");
});

test("classifies pastel pink material as inorganic", () => {
  const width = 32;
  const height = 32;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    pixels[offset] = 220;
    pixels[offset + 1] = 150;
    pixels[offset + 2] = 165;
    pixels[offset + 3] = 255;
  }

  const appearance = analyzeObjectAppearance({ data: pixels, width, height });
  assert.equal(appearance.isHardInorganic, true);
});
