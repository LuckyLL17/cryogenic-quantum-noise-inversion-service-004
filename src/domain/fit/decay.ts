import type { DecayFit, DecayFitReason, SynchronizedSample } from '../types.ts';
import { noDecayFit } from '../types.ts';

// Robust exponential decay fitter.
//
// Models compared on the same set of inliers (u is normalized time in [0, 1]):
//   M0   : y = c                         constant / pure baseline
//   Mlin : y = d*u + c                   linear baseline drift, no decay
//   Mexp : y = A*exp(-u/r) + c           decay on a constant baseline
//   Md   : y = A*exp(-u/r) + d*u + c     decay on a drifting baseline
// where r = tau / windowSpan.
//
// Outliers (preparation-stage residue, glitches) are removed by an iteratively
// reweighted least-squares fit with Tukey bisquare weights seeded by a least
// median-of-squares estimate, so rejection scales with the data's own residual
// scale instead of a fixed amplitude threshold. Models are chosen by BIC on the
// robust scale, so flat / rising / drifting curves are not forced to report a
// time constant.

type Point = { t: number; y: number; baseWeight: number; variance: number };
type ModelName = 'constant' | 'linear' | 'exponential' | 'exponential-drift';

const TUKEY_K = 4.685; // bisquare weight reaches zero at ~4.7 residual standard deviations
const GRID_POINTS = 80;
const R_MIN = 1 / 300; // decay can complete inside the first sampling interval
const R_MAX = 12; // decay barely visible across the window
const MIN_INLIERS = 3;
const BIC_PROMOTION = 2; // require a meaningful likelihood gain before adding parameters
const CI_Z = 1.96;

const TAU_GRID: number[] = Array.from({ length: GRID_POINTS }, (_, index) => {
  const fraction = index / (GRID_POINTS - 1);
  return R_MIN * (R_MAX / R_MIN) ** fraction;
});

function columns(model: ModelName, u: number, r: number): number[] {
  const decay = Math.exp(-u / r);
  switch (model) {
    case 'constant': return [1];
    case 'linear': return [u, 1];
    case 'exponential': return [decay, 1];
    case 'exponential-drift': return [decay, u, 1];
  }
}

const MODEL_K: Record<ModelName, number> = { constant: 1, linear: 2, exponential: 3, 'exponential-drift': 4 };
const MODEL_HAS_TAU: Record<ModelName, boolean> = { constant: false, linear: false, exponential: true, 'exponential-drift': true };

/** Solves (X'WX) beta = X'Wy for at most 3 predictors; null when rank deficient. */
function weightedLeastSquares(design: number[][], y: number[], weights: number[]): number[] | null {
  const p = design[0].length;
  const matrix = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const vector = new Array<number>(p).fill(0);
  let scale = 1;
  for (let i = 0; i < design.length; i += 1) {
    const w = weights[i];
    if (w <= 0) continue;
    for (let a = 0; a < p; a += 1) {
      for (let b = 0; b < p; b += 1) matrix[a][b] += w * design[i][a] * design[i][b];
      vector[a] += w * design[i][a] * y[i];
      scale = Math.max(scale, Math.abs(design[i][a]));
    }
  }
  const tolerance = 1e-12 * scale * scale;
  for (let col = 0; col < p; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < p; row += 1) if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
    if (Math.abs(matrix[pivot][col]) < tolerance) return null;
    if (pivot !== col) {
      [matrix[pivot], matrix[col]] = [matrix[col], matrix[pivot]];
      [vector[pivot], vector[col]] = [vector[col], vector[pivot]];
    }
    for (let row = col + 1; row < p; row += 1) {
      const factor = matrix[row][col] / matrix[col][col];
      for (let b = col; b < p; b += 1) matrix[row][b] -= factor * matrix[col][b];
      vector[row] -= factor * vector[col];
    }
  }
  const beta = new Array<number>(p).fill(0);
  for (let row = p - 1; row >= 0; row -= 1) {
    let sum = vector[row];
    for (let b = row + 1; b < p; b += 1) sum -= matrix[row][b] * beta[b];
    beta[row] = sum / matrix[row][row];
  }
  return beta.every(Number.isFinite) ? beta : null;
}

function weightedQuantile(values: number[], weights: number[], quantile: number): number {
  const pairs = values.map((value, index) => ({ value, weight: Math.max(0, weights[index]) })).filter((pair) => pair.weight > 0).sort((a, b) => a.value - b.value);
  const total = pairs.reduce((sum, pair) => sum + pair.weight, 0);
  if (total <= 0) return 0;
  let cumulative = 0;
  for (const pair of pairs) {
    cumulative += pair.weight;
    if (cumulative >= quantile * total) return pair.value;
  }
  return pairs.at(-1)!.value;
}

type CandidateModel = {
  model: ModelName;
  beta: number[];
  r: number | null;
  weights: number[];
  residuals: number[];
  rss: number;
  noiseFloor: number;
  inliers: number;
  rSquared: number;
  bic: number;
};

function predict(model: ModelName, beta: number[], u: number, r: number | null): number {
  const cols = columns(model, u, r ?? R_MIN);
  return beta.reduce((sum, coefficient, index) => sum + coefficient * cols[index], 0);
}

/** Fits one model with fixed weights, optimizing tau on the grid for decay models. */
function fitWithWeights(points: Point[], u: number[], model: ModelName, weights: number[]): { beta: number[]; r: number | null; residuals: number[]; rss: number } | null {
  const evaluate = (r: number): { beta: number[]; r: number; residuals: number[]; rss: number } | null => {
    const design = points.map((_, index) => columns(model, u[index], r));
    const beta = weightedLeastSquares(design, points.map((point) => point.y), weights);
    if (!beta) return null;
    const amplitude = beta[0]; // first column is exp(-u/r) for every decay model
    if (!(amplitude > 0) || !beta.every(Number.isFinite)) return null;
    const residuals = points.map((point, index) => point.y - predict(model, beta, u[index], r));
    const rss = residuals.reduce((sum, residual, index) => sum + weights[index] * residual ** 2, 0);
    return { beta, r, residuals, rss };
  };
  if (!MODEL_HAS_TAU[model]) {
    const design = points.map((_, index) => columns(model, u[index], R_MIN));
    const beta = weightedLeastSquares(design, points.map((point) => point.y), weights);
    if (!beta) return null;
    const residuals = points.map((point, index) => point.y - predict(model, beta, u[index], null));
    const rss = residuals.reduce((sum, residual, index) => sum + weights[index] * residual ** 2, 0);
    return { beta, r: null, residuals, rss };
  }
  let best: { beta: number[]; r: number; residuals: number[]; rss: number } | null = null;
  for (const r of TAU_GRID) {
    const candidate = evaluate(r);
    if (candidate && (!best || candidate.rss < best.rss)) best = candidate;
  }
  if (!best) return null;
  // Golden-section refinement in log-r around the best grid cell keeps the coarse grid
  // from leaking discretization error into the baseline-drift column.
  const gridIndex = TAU_GRID.findIndex((gridR) => gridR === best.r);
  let lo = Math.log(gridIndex > 0 ? TAU_GRID[gridIndex - 1] : R_MIN);
  let hi = Math.log(gridIndex < TAU_GRID.length - 1 ? TAU_GRID[gridIndex + 1] : R_MAX);
  const golden = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - golden * (hi - lo);
  let x2 = lo + golden * (hi - lo);
  let f1 = evaluate(Math.exp(x1));
  let f2 = evaluate(Math.exp(x2));
  for (let iteration = 0; iteration < 30 && hi - lo > 1e-4; iteration += 1) {
    if (f2 && (!f1 || f2.rss < f1.rss)) {
      lo = x1; x1 = x2; f1 = f2;
      x2 = lo + golden * (hi - lo);
      f2 = evaluate(Math.exp(x2));
    } else {
      hi = x2; x2 = x1; f2 = f1;
      x1 = hi - golden * (hi - lo);
      f1 = evaluate(Math.exp(x1));
    }
  }
  const refined = [best, f1, f2, evaluate(Math.exp((lo + hi) / 2))]
    .filter((item): item is NonNullable<typeof item> => item !== null && item.beta[0] > 0);
  return refined.reduce((winner, item) => item.rss < winner.rss ? item : winner, best);
}

/** Weighted solve at one fixed tau (no grid/refinement), null when unsolvable. */
function fitFixedR(points: Point[], u: number[], model: ModelName, weights: number[], fixedR: number | null): { beta: number[]; r: number | null; residuals: number[]; rss: number } | null {
  const r = fixedR ?? R_MIN;
  const design = points.map((_, index) => columns(model, u[index], r));
  const beta = weightedLeastSquares(design, points.map((point) => point.y), weights);
  if (!beta) return null;
  const residuals = points.map((point, index) => point.y - predict(model, beta, u[index], fixedR));
  const rss = residuals.reduce((sum, residual, index) => sum + weights[index] * residual ** 2, 0);
  return { beta, r: fixedR, residuals, rss };
}

/** Robust residual scale: MAD estimate, never below the weighted stated noise level. */
function robustSigma(points: Point[], residuals: number[], weights: number[]): number {
  const absResidual = residuals.map((residual) => Math.abs(residual));
  const madScale = 1.4826 * weightedQuantile(absResidual, weights, 0.5);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const statedScale = Math.sqrt(points.reduce((sum, point, index) => sum + weights[index] * point.variance, 0) / Math.max(totalWeight, 1e-300));
  return Math.max(madScale, statedScale);
}

/**
 * Iteratively reweighted fit. Warm-up iterations keep tau pinned to the LMedS seed and
 * use Huber weights that shrink gross outliers without zeroing them, so a poor seed
 * cannot hard-reject genuine early-decay points. Bisquare iterations then free tau and
 * are allowed to reject points beyond the robust scale.
 */
function robustFit(points: Point[], u: number[], model: ModelName, seedR: number | null): CandidateModel | null {
  let weights = points.map((point) => point.baseWeight);
  let mask = weights.map((weight) => weight > 0);
  let current: { beta: number[]; r: number | null; residuals: number[]; rss: number } | null = null;
  const HUBER_K = 1.345;
  for (let iteration = 0; iteration < 20; iteration += 1) {
    const warmUp = iteration < 3;
    const fit = warmUp
      ? fitFixedR(points, u, model, weights, MODEL_HAS_TAU[model] ? seedR : null)
      : fitWithWeights(points, u, model, weights);
    if (!fit) return null;
    const sigma = robustSigma(points, fit.residuals, weights);
    const nextWeights = points.map((point, index) => {
      const standardized = Math.abs(fit.residuals[index]) / Math.max(sigma, 1e-300);
      if (warmUp) {
        const factor = Math.min(1, HUBER_K / Math.max(standardized, 1e-12));
        return point.baseWeight * factor;
      }
      if (standardized >= TUKEY_K) return 0;
      const factor = 1 - (standardized / TUKEY_K) ** 2;
      return point.baseWeight * factor * factor;
    });
    const nextMask = nextWeights.map((weight) => weight > 0);
    current = fit;
    if (!warmUp && nextMask.every((inside, index) => inside === mask[index])) break;
    mask = nextMask;
    weights = nextWeights;
  }
  if (!current) return null;
  return finalizeCandidate(points, u, model, current, weights);
}

function finalizeCandidate(points: Point[], u: number[], model: ModelName, current: { beta: number[]; r: number | null; residuals: number[]; rss: number }, weights: number[]): CandidateModel | null {
  // Refit once with converged weights so coefficients describe the inliers only.
  const refit = fitWithWeights(points, u, model, weights) ?? current;
  const inliers = weights.filter((weight) => weight > 0).length;
  if (inliers < MIN_INLIERS) return null;
  const weightedY = points.map((point, index) => ({ y: point.y, w: weights[index] })).filter((item) => item.w > 0);
  const totalWeight = weightedY.reduce((sum, item) => sum + item.w, 0);
  const meanY = weightedY.reduce((sum, item) => sum + item.w * item.y, 0) / totalWeight;
  const totalSs = weightedY.reduce((sum, item) => sum + item.w * (item.y - meanY) ** 2, 0);
  const rss = refit.residuals.reduce((sum, residual, index) => sum + weights[index] * residual ** 2, 0);
  const rSquared = totalSs > 0 ? Math.max(0, Math.min(1, 1 - rss / totalSs)) : 1;
  // RSS is dimensionless chi^2 (weights are 1/variance), so its floor is the inlier
  // count. Flooring keeps sub-noise residual differences (rounding, grid error) from
  // earning extra parameters through BIC alone.
  const inlierWeightCount = weights.reduce((sum, weight) => sum + (weight > 0 ? 1 : 0), 0);
  const noiseFloor = Math.sqrt(points.reduce((sum, point, index) => sum + weights[index] * point.variance, 0) / Math.max(totalWeight, 1e-300));
  const bicRss = Math.max(rss, inlierWeightCount);
  const bic = totalWeight > 0
    ? totalWeight * Math.log(bicRss / totalWeight) + MODEL_K[model] * Math.log(totalWeight)
    : -Infinity;
  return { model, beta: refit.beta, r: refit.r, weights, residuals: refit.residuals, rss, noiseFloor, inliers, rSquared, bic };
}

/** Least-median-of-squares tau seed: tolerates nearly half gross outliers without a rejection threshold. */
function lmedsSeed(points: Point[], u: number[]): number | null {
  let seed: { r: number; median: number } | null = null;
  for (const r of TAU_GRID) {
    const design = points.map((_, index) => columns('exponential', u[index], r));
    const weights = points.map((point) => point.baseWeight);
    const beta = weightedLeastSquares(design, points.map((point) => point.y), weights);
    if (!beta || !(beta[0] > 0)) continue;
    const squaredResidual = points.map((point, index) => (point.y - predict('exponential', beta, u[index], r)) ** 2);
    const median = weightedQuantile(squaredResidual, weights, 0.5);
    if (!seed || median < seed.median) seed = { r, median };
  }
  return seed?.r ?? null;
}

function lag1ResidualStructure(residuals: number[], weights: number[], noiseFloor: number): boolean {
  const active = residuals.map((_, index) => index).filter((index) => weights[index] > 0);
  if (active.length < 5) return false;
  const values = active.map((index) => residuals[index]);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let variance = 0; let lag1 = 0;
  for (let i = 0; i < values.length; i += 1) {
    variance += (values[i] - mean) ** 2;
    if (i > 0) lag1 += (values[i] - mean) * (values[i - 1] - mean);
  }
  const rms = Math.sqrt(variance / values.length);
  // Correlation of sub-noise residuals (rounding, near-exact fits) is meaningless.
  if (rms <= noiseFloor || variance === 0) return false;
  return Math.abs(lag1 / variance) > 2 / Math.sqrt(values.length - 1);
}

/**
 * Profile-likelihood interval for tau: refit the linear coefficients at each trial r and
 * find where the weighted chi^2 rises by z^2 over its best-fit value (floored at the
 * expected chi^2 of one residual per inlier). This profiles the exact reported model, so
 * the interval always brackets the point estimate.
 */
function timeConstantInterval(points: Point[], u: number[], weights: number[], model: ModelName, bestR: number, span: number): { lowerNs: number | null; upperNs: number | null } {
  const rssAt = (r: number): number => {
    const design = points.map((_, index) => columns(model, u[index], r));
    const beta = weightedLeastSquares(design, points.map((point) => point.y), weights);
    if (!beta || !(beta[0] > 0)) return Number.POSITIVE_INFINITY;
    return points.reduce((sum, point, index) => {
      if (weights[index] <= 0) return sum;
      const residual = point.y - predict(model, beta, u[index], r);
      return sum + weights[index] * residual ** 2;
    }, 0);
  };
  const inlierCountCi = weights.reduce((sum, weight) => sum + (weight > 0 ? 1 : 0), 0);
  // Weighted RSS is chi^2; floor at its residual degrees of freedom so an exact-looking
  // fit (including an exactly determined 3-point fit) still reports an honest interval.
  const residualDof = Math.max(1, inlierCountCi - MODEL_K[model]);
  const bestRss = Math.max(rssAt(bestR), residualDof);
  const target = bestRss + CI_Z ** 2;
  const crossing = (direction: -1 | 1): number | null => {
    let boundary = bestR;
    for (let step = 1; step <= 60; step += 1) {
      boundary = direction === 1
        ? Math.min(R_MAX, bestR * 2 ** step)
        : Math.max(R_MIN, bestR / 2 ** step);
      if (boundary === (direction === 1 ? R_MAX : R_MIN)) break;
      if (rssAt(boundary) >= target) break;
    }
    if (rssAt(boundary) < target) return null; // bound unconstrained within the observable range
    let lo = direction === 1 ? bestR : boundary;
    let hi = direction === 1 ? boundary : bestR;
    for (let iteration = 0; iteration < 40; iteration += 1) {
      const mid = Math.exp((Math.log(lo) + Math.log(hi)) / 2);
      if (rssAt(mid) >= target) {
        if (direction === 1) hi = mid; else lo = mid;
      } else if (direction === 1) lo = mid; else hi = mid;
      if (Math.abs(Math.log(hi / lo)) < 1e-4) break;
    }
    return Math.exp((Math.log(lo) + Math.log(hi)) / 2);
  };
  const lowerR = crossing(-1);
  const upperR = crossing(1);
  return {
    lowerNs: lowerR !== null ? lowerR * span : null,
    upperNs: upperR !== null ? upperR * span : null,
  };
}

function assembleFailure(reasons: DecayFitReason[], points: Point[], rejectedPoints: number, status: DecayFit['status']): DecayFit {
  const weights = points.map((point) => point.baseWeight);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const offset = total > 0 ? points.reduce((sum, point, index) => sum + weights[index] * point.y, 0) / total : 0;
  return noDecayFit(offset, points.length, rejectedPoints, status, reasons);
}

export function fitExponential(samples: SynchronizedSample[], startNs?: number, endNs?: number): DecayFit {
  const selected = samples
    .filter((sample) => (startNs === undefined || sample.timestampNs >= startNs) && (endNs === undefined || sample.timestampNs < endNs))
    .sort((a, b) => a.timestampNs - b.timestampNs || a.syncIndex - b.syncIndex);
  if (selected.length === 0) return noDecayFit(0, 0, 0, 'invalid', ['no-samples-in-window']);

  const reasons: DecayFitReason[] = [];
  let rejectedPoints = 0;
  const points: Point[] = [];
  for (const sample of selected) {
    if (!Number.isFinite(sample.amplitude) || !Number.isFinite(sample.timestampNs)) {
      rejectedPoints += 1;
      continue;
    }
    const qualityWeight = sample.quality === 'suspect' ? 0.5 : 1;
    const variance = Math.max(sample.uncertainty, 1e-9) ** 2;
    points.push({ t: sample.timestampNs, y: sample.amplitude, baseWeight: qualityWeight / variance, variance });
  }
  if (rejectedPoints > 0) reasons.push('nonfinite-points-removed');
  if (points.length === 0) {
    return noDecayFit(0, 0, rejectedPoints, 'invalid', [...reasons, 'few-usable-points']);
  }
  if (points.length < MIN_INLIERS) return assembleFailure([...reasons, 'few-usable-points'], points, rejectedPoints, 'insufficient-points');

  const t0 = points[0].t;
  const span = points.at(-1)!.t - t0;
  if (!(span > 0) || !Number.isFinite(span)) return assembleFailure([...reasons, 'numerical-failure'], points, rejectedPoints, 'invalid');
  const u = points.map((point) => (point.t - t0) / span);

  const seedR = lmedsSeed(points, u);
  // Build outlier masks with two robust views. A constant-baseline decay masks glitches
  // well but treats a genuinely drifting baseline as outliers, while the drift-tolerant
  // view absorbs that trend. Prefer the drift mask only when it rescues a meaningful
  // group of points the tight mask rejected; otherwise the tight mask is safer.
  const exponentialMask = seedR !== null ? robustFit(points, u, 'exponential', seedR) : null;
  const driftMask = seedR !== null ? robustFit(points, u, 'exponential-drift', seedR) : null;
  let robust: CandidateModel | null = null;
  if (exponentialMask && driftMask) {
    const expInliers = exponentialMask.inliers;
    const driftInliers = driftMask.inliers;
    const rescuedCount = points.reduce(
      (count, _, index) => count + (exponentialMask.weights[index] <= 0 && driftMask.weights[index] > 0 ? 1 : 0),
      0,
    );
    robust = rescuedCount >= 3 && driftInliers > expInliers ? driftMask : exponentialMask;
  } else {
    // No positive-amplitude exponential can explain the data at all: mask with a line.
    robust = exponentialMask ?? driftMask ?? robustFit(points, u, 'linear', null) ?? robustFit(points, u, 'constant', null);
  }
  if (!robust || robust.inliers < MIN_INLIERS) {
    return assembleFailure([...reasons, 'few-usable-points'], [], rejectedPoints + points.length, 'insufficient-points');
  }
  const inlierCount = robust.inliers;
  rejectedPoints += points.length - inlierCount;
  const usedPoints = inlierCount;
  if (usedPoints < MIN_INLIERS) return assembleFailure([...reasons, 'few-usable-points'], points.filter((_, index) => robust.weights[index] > 0), rejectedPoints, 'insufficient-points');

  // Refit every nested model on the converged mask and select by BIC (promotion needs a real gain).
  const candidates = new Map<ModelName, CandidateModel>();
  for (const model of ['constant', 'linear', 'exponential', 'exponential-drift'] as ModelName[]) {
    if (MODEL_K[model] > usedPoints) continue;
    if (MODEL_HAS_TAU[model] && seedR === null) continue;
    const fit = fitWithWeights(points, u, model, robust.weights);
    if (!fit) continue;
    const candidate = finalizeCandidate(points, u, model, fit, robust.weights);
    if (candidate) candidates.set(model, candidate);
  }
  // Walk from the simplest model; a complex model is only adopted when its BIC improves
  // on the current winner by the promotion margin. Added drift must also clear a
  // likelihood-ratio gate (chi^2, 1 dof, 95%), and a decay amplitude must exceed the
  // noise floor, so trend/flat data cannot smuggle in a time constant.
  const CHI2_1DF_95 = 3.84;
  const modelOrder: ModelName[] = ['constant', 'linear', 'exponential', 'exponential-drift'];
  const constantModel = candidates.get('constant');
  if (!constantModel) return assembleFailure([...reasons, 'numerical-failure'], [], rejectedPoints + points.length, 'invalid');
  let chosen: CandidateModel = constantModel;
  for (const name of modelOrder.slice(1)) {
    const candidate = candidates.get(name);
    if (!candidate) continue;
    if (chosen.bic - candidate.bic < BIC_PROMOTION) continue;
    if (name === 'exponential-drift') {
      // A spurious drift slope can absorb ~1-sigma residual scatter with many points;
      // require a significant chi^2 reduction over Mexp before accepting it.
      const decayOnly = candidates.get('exponential');
      if (!decayOnly || decayOnly.rss - candidate.rss < CHI2_1DF_95) continue;
    }
    if ((name === 'exponential' || name === 'exponential-drift')
      && candidate.noiseFloor > 0 && candidate.beta[0] / candidate.noiseFloor < CI_Z) continue;
    chosen = candidate;
  }

  const inlierPoints = points.filter((_, index) => robust.weights[index] > 0);
  const robustOffset = (() => {
    const inlierWeights = robust.weights.filter((weight) => weight > 0);
    const totalWeight = inlierWeights.reduce((sum, weight) => sum + weight, 0);
    return inlierPoints.reduce((sum, point, index) => sum + inlierWeights[index] * point.y, 0) / totalWeight;
  })();

  // A tau pinned to the long edge means no visible decay happened in the window: do not invent one.
  const tauPinnedLong = chosen.r !== null && chosen.r >= R_MAX * (1 - 1e-9);
  const tauPinnedShort = chosen.r !== null && chosen.r <= R_MIN * (1 + 1e-9);

  if (chosen.model === 'constant' || (chosen.model === 'exponential' && tauPinnedLong)) {
    return noDecayFit(robustOffset, usedPoints, rejectedPoints, 'non-decaying', [...reasons, 'no-decay-detected']);
  }
  if (chosen.model === 'linear' || (chosen.model === 'exponential-drift' && tauPinnedLong)) {
    // A significant moving baseline with no identifiable decay on top of it.
    const slopePerNs = chosen.model === 'linear' ? chosen.beta[0] / span : chosen.beta[1] / span;
    const driftReasons: DecayFitReason[] = [...reasons, 'drift-without-decay', ...(tauPinnedLong ? ['no-decay-detected' as const] : [])];
    return {
      model: 'exponential', status: 'baseline-drift', confidence: 'none', reasons: driftReasons,
      amplitude0: null, timeConstantNs: null, offset: chosen.beta[chosen.beta.length - 1],
      baselineSlopePerNs: slopePerNs, rSquared: chosen.rSquared, usedPoints, rejectedPoints,
      confidenceInterval: { lowerNs: null, upperNs: null },
    };
  }

  const decayModel = chosen.model as 'exponential' | 'exponential-drift';
  const amplitude = chosen.beta[0];
  const tauNs = chosen.r! * span;
  const offset = chosen.beta[chosen.beta.length - 1];
  const slopePerNs = decayModel === 'exponential-drift' ? chosen.beta[1] / span : 0;

  if (!(amplitude > 0) || !(tauNs > 0) || !Number.isFinite(tauNs)) {
    return noDecayFit(robustOffset, usedPoints, rejectedPoints, 'invalid', [...reasons, 'decay-did-not-converge']);
  }

  // The drift coefficient survived the likelihood-ratio gate, so the decay is identifiable
  // even though the baseline moved: report tau, but at reduced confidence and say so.
  const hasDrift = decayModel === 'exponential-drift';
  const interval = timeConstantInterval(points, u, robust.weights, decayModel, chosen.r!, span);
  const wideInterval = interval.lowerNs === null || interval.upperNs === null
    || Math.log(interval.upperNs / Math.max(interval.lowerNs, 1e-300)) > 2; // wider than a factor of ~7.4
  const structured = lag1ResidualStructure(chosen.residuals, robust.weights, chosen.noiseFloor);

  // At least two points beyond the fitted coefficients are needed to check a decay model
  // against the data; an exactly determined fit is reported but never "high" confidence.
  const hasResidualChecks = usedPoints >= MODEL_K[decayModel] + 2;

  if (tauPinnedShort) reasons.push('time-constant-at-window-edge');
  if (hasDrift) reasons.push('baseline-drift-corrected');
  if (!hasResidualChecks) reasons.push('few-usable-points');
  if (wideInterval) {
    reasons.push(interval.lowerNs === null || interval.upperNs === null
      ? 'time-constant-unconstrained'
      : 'wide-confidence-interval');
  }
  if (structured) reasons.push('structured-residuals');

  const constantBic = candidates.get('constant')?.bic;
  const decisive = constantBic === undefined || constantBic === -Infinity || constantBic - chosen.bic >= 10;
  const highConfidence = !hasDrift
    && hasResidualChecks
    && !tauPinnedShort
    && !structured
    && !wideInterval
    && decisive;

  return {
    model: 'exponential',
    status: hasDrift ? 'baseline-drift' : 'fitted',
    confidence: highConfidence ? 'high' : 'low',
    reasons,
    amplitude0: amplitude,
    timeConstantNs: tauNs,
    offset,
    baselineSlopePerNs: slopePerNs,
    rSquared: chosen.rSquared,
    usedPoints,
    rejectedPoints,
    confidenceInterval: { lowerNs: interval.lowerNs, upperNs: interval.upperNs },
  };
}
