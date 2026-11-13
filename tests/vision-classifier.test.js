import assert from "node:assert/strict";
import test from "node:test";
import {
  classifyLabel,
  classifyWastePredictions,
  estimateOrganicColorEvidence,
} from "../src/vision-classifier.js";

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
    0,
    0.35
  );

  assert.equal(result.category, "Basura Orgánica");
  assert.ok(Math.abs(result.score - 0.18) < Number.EPSILON * 2);
});

test("keeps a strong container prediction over a weak color fallback", () => {
  const result = classifyWastePredictions(
    [{ className: "water bottle", probability: 0.8 }],
    { class: "bottle", score: 0.9 },
    0.65,
    0.35
  );

  assert.equal(result.category, "Basura Inorgánica");
});

test("recognizes cutlery-class predictions as inorganic instead of using color fallback", () => {
  const result = classifyWastePredictions(
    [{ className: "letter opener, paper knife", probability: 0.327 }],
    { class: "scissors", score: 0.36 },
    0.59,
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
    pixels[offset] = 255;
    pixels[offset + 1] = 255;
    pixels[offset + 2] = 255;
    pixels[offset + 3] = 255;
  }
  for (let y = 8; y < 24; y += 1) {
    for (let x = 8; x < 24; x += 1) {
      const offset = (y * width + x) * 4;
      pixels[offset] = 25;
      pixels[offset + 1] = 85;
      pixels[offset + 2] = 20;
    }
  }

  const evidence = estimateOrganicColorEvidence({
    data: pixels,
    width,
    height,
  });

  assert.ok(evidence >= 0.3);
  assert.equal(
    estimateOrganicColorEvidence({
      data: new Uint8ClampedArray(width * height * 4).fill(255),
      width,
      height,
    }),
    0
  );
});

test("does not treat a pink plastic object as warm organic produce", () => {
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

  assert.ok(
    estimateOrganicColorEvidence({ data: pixels, width, height }) <
      0.3
  );
});
