const HISTORY_SIZE = 6;
const CATEGORY_INITIAL_VOTES = 3;
const CATEGORY_SWITCH_VOTES = 4;
const HIGH_CONFIDENCE_SWITCH_THRESHOLD = 0.6;
const MAX_MISSED_FRAMES = 3;
const SAME_OBJECT_CENTER_DISTANCE = 0.18;

export function createDetectionStabilizer() {
  let history = [];
  let activeCategory;
  let missedFrames = 0;

  function reset() {
    history = [];
    activeCategory = undefined;
    missedFrames = 0;
  }

  function stabilize(detections) {
    const detection = detections[0];
    if (!detection) {
      missedFrames += 1;
      history.push(null);
      trimHistory();
      if (missedFrames >= MAX_MISSED_FRAMES) {
        reset();
        return [];
      }
      return getSmoothedDetection();
    }

    missedFrames = 0;
    const previousDetection = findLatestDetection();
    if (previousDetection && !isSameObject(previousDetection, detection)) {
      reset();
    }

    history.push(detection);
    trimHistory();
    if (!activeCategory) {
      const initialCategory = getMostVotedCategory();
      if (
        initialCategory &&
        countCategoryVotes(initialCategory) >= CATEGORY_INITIAL_VOTES
      ) {
        activeCategory = initialCategory;
      } else {
        return [];
      }
    } else if (
      detection.category !== activeCategory &&
      (countCategoryVotes(detection.category) >= CATEGORY_SWITCH_VOTES ||
        detection.score > HIGH_CONFIDENCE_SWITCH_THRESHOLD)
    ) {
      activeCategory = detection.category;
    }

    return getSmoothedDetection();
  }

  function trimHistory() {
    if (history.length > HISTORY_SIZE) {
      history.shift();
    }
  }

  function findLatestDetection() {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      if (history[index]) {
        return history[index];
      }
    }
    return undefined;
  }

  function countCategoryVotes(category) {
    return history.filter((item) => item?.category === category).length;
  }

  function getMostVotedCategory() {
    const counts = new Map();
    for (const item of history) {
      if (item) {
        counts.set(item.category, (counts.get(item.category) || 0) + 1);
      }
    }
    return [...counts.entries()].sort(
      (first, second) => second[1] - first[1]
    )[0]?.[0];
  }

  function getSmoothedDetection() {
    const categoryHistory = history.filter(
      (item) => item?.category === activeCategory
    );
    const latestDetection = categoryHistory[categoryHistory.length - 1];
    if (!latestDetection) {
      return [];
    }

    const smoothedDetection = { ...latestDetection, category: activeCategory };
    for (const property of [
      "score",
      "x1",
      "y1",
      "x2",
      "y2",
      "centerX",
      "centerY",
      "centerDistance",
    ]) {
      smoothedDetection[property] =
        categoryHistory.reduce((total, item) => total + item[property], 0) /
        categoryHistory.length;
    }
    return [smoothedDetection];
  }

  function isSameObject(first, second) {
    const centerDistance = Math.hypot(
      first.centerX - second.centerX,
      first.centerY - second.centerY
    );
    if (centerDistance <= SAME_OBJECT_CENTER_DISTANCE) {
      return true;
    }

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
    return union > 0 && intersection / union >= 0.15;
  }

  return { reset, stabilize };
}
