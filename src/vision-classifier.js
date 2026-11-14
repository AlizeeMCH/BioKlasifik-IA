const ORGANIC_KEYWORDS = [
  "banana",
  "orange",
  "lemon",
  "lime",
  "fig",
  "pineapple",
  "apple",
  "strawberry",
  "pomegranate",
  "jackfruit",
  "custard apple",
  "bell pepper",
  "cucumber",
  "zucchini",
  "acorn squash",
  "butternut squash",
  "broccoli",
  "cauliflower",
  "cabbage",
  "artichoke",
  "cardoon",
  "mushroom",
  "granny smith",
  "corn",
  "ear",
  "vegetable",
  "fruit",
  "food",
  "guacamole",
  "potpie",
  "burrito",
  "pizza",
  "bagel",
  "pretzel",
  "mashed potato",
  "head cabbage",
  "carrot",
  "hip",
  "rosehip",
  "acorn",
  "buckeye",
  "coral fungus",
  "agaric",
  "gyromitra",
  "stinkhorn",
  "earthstar",
  "hen of the woods",
  "bolete",
  "plant",
  "leaf",
  "flower",
  "pot",
  "tomato",
  "avocado",
  "jalapeno",
  "chili",
  "chile",
  "pepper",
  "sweet pepper",
  "produce",
  "peppers",
  "jalapenos",
  "tomatoes",
  "avocados",
];
const INORGANIC_KEYWORDS = [
  "bottle",
  "water bottle",
  "pop bottle",
  "beer bottle",
  "wine bottle",
  "plastic bag",
  "packet",
  "carton",
  "paper towel",
  "toilet tissue",
  "envelope",
  "book",
  "letter opener",
  "paper knife",
  "paperknife",
  "can opener",
  "tin opener",
  "scissors",
  "binder",
  "notebook",
  "menu",
  "comic book",
  "ballpoint",
  "fountain pen",
  "quill",
  "marker",
  "rubber eraser",
  "pencil",
  "can",
  "beer glass",
  "goblet",
  "cup",
  "coffee mug",
  "measuring cup",
  "tray",
  "plate",
  "pill bottle",
  "lotion",
  "sunscreen",
  "hair clip",
  "hair slide",
  "hairpin",
  "clothespin",
  "clothes peg",
  "paper clip",
  "clamp",
  "buckle",
  "lighter",
  "mouse",
  "remote",
  "cellular telephone",
  "keyboard",
  "metal",
  "plastic",
  "glass",
];

const CATEGORY_ORGANIC = "Basura Orgánica";
const CATEGORY_INORGANIC = "Basura Inorgánica";
const ORGANIC_EVIDENCE_THRESHOLD = 0.15;

function normalizeLabel(label) {
  const asciiLabel = label
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
  return ` ${asciiLabel.replace(/[^a-z0-9]+/g, " ").trim()} `;
}

function matchesKeyword(label, keywords) {
  const normalizedLabel = normalizeLabel(label);
  return keywords.some((keyword) =>
    normalizedLabel.includes(normalizeLabel(keyword))
  );
}

export function classifyLabel(label) {
  if (matchesKeyword(label, ORGANIC_KEYWORDS)) {
    return CATEGORY_ORGANIC;
  }
  if (matchesKeyword(label, INORGANIC_KEYWORDS)) {
    return CATEGORY_INORGANIC;
  }
  return undefined;
}

function combineEvidence(mobileScore, cocoScore) {
  return 1 - (1 - mobileScore) * (1 - cocoScore * 0.7);
}

export function classifyWastePredictions(
  mobilePredictions,
  cocoPrediction,
  appearance,
  minimumConfidence
) {
  const mobileEvidence = {
    [CATEGORY_ORGANIC]: 0,
    [CATEGORY_INORGANIC]: 0,
  };
  for (const prediction of mobilePredictions) {
    const category = classifyLabel(prediction.className);
    if (category && Number.isFinite(prediction.probability)) {
      mobileEvidence[category] += prediction.probability;
    }
  }

  const cocoCategory = cocoPrediction
    ? classifyLabel(cocoPrediction.class)
    : undefined;
  const cocoScore =
    cocoCategory && Number.isFinite(cocoPrediction.score)
      ? cocoPrediction.score
      : 0;
  const organicEvidence = combineEvidence(
    Math.min(1, mobileEvidence[CATEGORY_ORGANIC]),
    cocoCategory === CATEGORY_ORGANIC ? cocoScore : 0
  );
  const inorganicEvidence = combineEvidence(
    Math.min(1, mobileEvidence[CATEGORY_INORGANIC]),
    cocoCategory === CATEGORY_INORGANIC ? cocoScore : 0
  );
  const threshold = Math.min(0.75, Math.max(0.15, minimumConfidence));

  if (appearance.isHardInorganic) {
    return {
      category: CATEGORY_INORGANIC,
      score: Math.max(0.61, appearance.inorganicEvidence),
    };
  }
  if (
    organicEvidence >= ORGANIC_EVIDENCE_THRESHOLD &&
    (organicEvidence >= inorganicEvidence || inorganicEvidence < threshold)
  ) {
    return { category: CATEGORY_ORGANIC, score: organicEvidence };
  }
  if (inorganicEvidence >= threshold && inorganicEvidence > organicEvidence) {
    return { category: CATEGORY_INORGANIC, score: inorganicEvidence };
  }
  if (
    inorganicEvidence < threshold &&
    appearance.organicEvidence >= 0.3
  ) {
    return {
      category: CATEGORY_ORGANIC,
      score: appearance.organicEvidence,
    };
  }
  return undefined;
}

function rgbToHsv(red, green, blue) {
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  const delta = maximum - minimum;
  let hue = 0;

  if (delta > 0) {
    if (maximum === red) {
      hue = 60 * (((green - blue) / delta) % 6);
    } else if (maximum === green) {
      hue = 60 * ((blue - red) / delta + 2);
    } else {
      hue = 60 * ((red - green) / delta + 4);
    }
  }

  return {
    hue: (hue + 360) % 360,
    saturation: maximum === 0 ? 0 : delta / maximum,
    value: maximum,
  };
}

export function analyzeObjectAppearance(imageData) {
  const { data, width, height } = imageData;
  const sampleWidth = Math.ceil(width / 2);
  const sampleHeight = Math.ceil(height / 2);
  const sampledLuminance = new Float32Array(sampleWidth * sampleHeight);
  const darkBrownMask = new Uint8Array(sampleWidth * sampleHeight);
  let eligiblePixels = 0;
  let organicPixels = 0;
  let sampledPixels = 0;
  let darkBrownPixels = 0;
  let neutralPixels = 0;
  let pastelPinkPixels = 0;

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const offset = (y * width + x) * 4;
      const sampleIndex = (y >> 1) * sampleWidth + (x >> 1);
      const red = data[offset] / 255;
      const green = data[offset + 1] / 255;
      const blue = data[offset + 2] / 255;
      const { hue, saturation, value } = rgbToHsv(red, green, blue);
      sampledLuminance[sampleIndex] =
        0.2126 * red + 0.7152 * green + 0.0722 * blue;

      sampledPixels += 1;
      if (value >= 0.62 && saturation <= 0.2) {
        neutralPixels += 1;
      }
      if (
        hue >= 320 &&
        hue <= 355 &&
        saturation >= 0.1 &&
        saturation <= 0.65 &&
        value >= 0.45
      ) {
        pastelPinkPixels += 1;
      }
      if (value > 0.9 && saturation < 0.24) {
        continue;
      }
      const greenProduce =
        hue >= 35 && hue <= 165 && saturation >= 0.18 && value > 0.1;
      const warmProduce =
        (hue <= 40 || hue >= 355) && saturation >= 0.3 && value > 0.12;
      const yellowProduce =
        hue >= 20 && hue <= 70 && saturation >= 0.3 && value > 0.12;
      const darkBrownProduce =
        hue >= 5 && hue <= 50 && saturation >= 0.2 && value <= 0.48;
      if (value <= 0.5 && saturation >= 0.18 && hue >= 5 && hue <= 50) {
        darkBrownMask[sampleIndex] = 1;
        darkBrownPixels += 1;
      }

      if (saturation >= 0.08 && value >= 0.07) {
        eligiblePixels += 1;
      }
      if (greenProduce || warmProduce || yellowProduce || darkBrownProduce) {
        organicPixels += 1;
      }
    }
  }

  if (sampledPixels === 0) {
    return {
      organicEvidence: 0,
      inorganicEvidence: 0,
      isHardInorganic: false,
    };
  }

  let textureEdges = 0;
  for (let y = 0; y < sampleHeight; y += 1) {
    for (let x = 0; x < sampleWidth; x += 1) {
      const sampleIndex = y * sampleWidth + x;
      if (!darkBrownMask[sampleIndex]) {
        continue;
      }
      if (
        x + 1 < sampleWidth &&
        Math.abs(sampledLuminance[sampleIndex] - sampledLuminance[sampleIndex + 1]) >=
          0.08
      ) {
        textureEdges += 1;
      }
      if (
        y + 1 < sampleHeight &&
        Math.abs(
          sampledLuminance[sampleIndex] -
            sampledLuminance[sampleIndex + sampleWidth]
        ) >= 0.08
      ) {
        textureEdges += 1;
      }
    }
  }

  const chromaticShare =
    eligiblePixels === 0 ? 0 : organicPixels / eligiblePixels;
  const objectCoverage = Math.min(1, organicPixels / sampledPixels / 0.3);
  const baseEvidence = 0.65 * Math.sqrt(chromaticShare * objectCoverage);
  const textureEvidence =
    darkBrownPixels > 0
      ? Math.min(
          0.12,
          (textureEdges / (darkBrownPixels * 2)) *
            (darkBrownPixels / sampledPixels) *
            0.3
        )
      : 0;
  const organicEvidence =
    eligiblePixels === 0
      ? 0
      : Math.min(0.65, baseEvidence + textureEvidence);
  const neutralShare = neutralPixels / sampledPixels;
  const pinkShare = pastelPinkPixels / sampledPixels;
  const isHardInorganic = neutralShare >= 0.62 || pinkShare >= 0.3;

  return {
    organicEvidence,
    inorganicEvidence: isHardInorganic
      ? Math.max(neutralShare, pinkShare)
      : 0,
    isHardInorganic,
  };
}
