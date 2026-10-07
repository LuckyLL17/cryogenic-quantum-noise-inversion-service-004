import type {
  DecayBaselineStrategy,
  DecayFit,
  DecayFitStatus,
  DecayOutlierStrategy,
  SynchronizedSample,
} from '../types.ts';

type Point = {
  x: number;
  y: number;
  sigma: number;
  qualityWeight: number;
};

type LinearSolution = { intercept: number; slope: number };
type ExponentialKind = 'constant' | 'drift';

type ExponentialParameters = {
  sign: 1 | -1;
  logAmplitude: number;
  logTimeConstant: number;
  offset: number;
  driftSlope: number;
};

type Candidate = {
  name: 'constant' | 'linear' | 'exponential-constant-decay' | 'exponential-constant-growth' | 'exponential-drift-decay' | 'exponential-drift-growth';
  kind: ExponentialKind | null;
  k: number;
  intercept: number;
  slope: number;
  weights: number[];
  scale: number;
  residuals: number[];
  rSquared: number;
  support: number;
  effectiveN: number;
  aicC: number;
  logLikelihood: number;
  exponential?: {
    sign: 1 | -1;
    amplitude: number;
    timeConstant: number;
    offset: number;
    driftSlope: number;
    standardErrorLogTau: number;
  };
};

const OUTLIER_STRATEGY: DecayOutlierStrategy = 'tukey-bisquare';
const TUKEY_C = 4.685;
const STUDENT_DF = 5;
const Z_95 = 1.959963984540054;
const MIN_POINTS = 3;
const MIN_DISTINCT_TIMES = 3;

function makeResult(
  status: DecayFitStatus,
  statusReason: string,
  startNs: number | undefined,
  endNs: number | undefined,
  usedPoints: number,
  rejectedPoints: number,
  partial?: Partial<DecayFit>,
): DecayFit {
  return {
    model: 'exponential',
    status,
    statusReason,
    fitStartNs: startNs ?? null,
    fitEndNs: endNs ?? null,
    amplitude0: null,
    timeConstantNs: null,
    offset: partial?.offset ?? 0,
    driftSlopePerNs: null,
    baselineStrategy: null,
    outlierStrategy: OUTLIER_STRATEGY,
    rSquared: null,
    usedPoints,
    rejectedPoints,
    confidence: { level: 0.95, lowerNs: null, upperNs: null, score: null, ...partial?.confidence },
    modelEvidence: { exponential: 0, constantBaseline: 0, linearBaseline: 0, ...partial?.modelEvidence },
    fitQuality: {
      modelEvidence: 0,
      inlierFraction: 0,
      identifiability: null,
      intervalPrecision: null,
      ...partial?.fitQuality,
    },
    warnings: partial?.warnings ?? [],
    ...partial,
  };
}

function weightedMean(values: number[], weights: number[]): number {
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  return totalWeight === 0 ? NaN : values.reduce((sum, value, index) => sum + value * weights[index], 0) / totalWeight;
}

function weightedMedian(values: number[], weights: number[]): number {
  const ordered = values
    .map((value, index) => ({ value, weight: Math.max(0, weights[index] ?? 0) }))
    .filter((item) => Number.isFinite(item.value) && Number.isFinite(item.weight) && item.weight > 0)
    .sort((a, b) => a.value - b.value);
  const total = ordered.reduce((sum, item) => sum + item.weight, 0);
  if (total === 0) return NaN;
  let cumulative = 0;
  for (const item of ordered) {
    cumulative += item.weight;
    if (cumulative >= total / 2) return item.value;
  }
  return ordered.at(-1)!.value;
}

function robustScale(residuals: number[], weights: number[]): number {
  return weightedMedian(residuals.map((residual) => Math.abs(residual)), weights) / 0.6744897501960817;
}

function effectiveN(weights: number[]): number {
  const sum = weights.reduce((total, weight) => total + weight, 0);
  const sumSquares = weights.reduce((total, weight) => total + weight * weight, 0);
  return sumSquares === 0 ? 0 : (sum * sum) / sumSquares;
}

function solve2x2(matrix: number[][], vector: number[]): number[] | null {
  const [[a, b], [c, d]] = matrix;
  const determinant = a * d - b * c;
  const scale = Math.max(1, Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d), Math.abs(vector[0]), Math.abs(vector[1]));
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= scale * scale * 1e-14) return null;
  return [(vector[0] * d - b * vector[1]) / determinant, (a * vector[1] - vector[0] * c) / determinant];
}

function weightedLinear(points: Point[], weights: number[]): LinearSolution | null {
  let sw = 0;
  let swx = 0;
  let swx2 = 0;
  let swy = 0;
  let swxy = 0;
  points.forEach((point, index) => {
    const w = weights[index];
    sw += w;
    swx += w * point.x;
    swx2 += w * point.x * point.x;
    swy += w * point.y;
    swxy += w * point.x * point.y;
  });
  const solution = solve2x2([[sw, swx], [swx, swx2]], [swy, swxy]);
  if (!solution || !solution.every(Number.isFinite)) return null;
  return { intercept: solution[0], slope: solution[1] };
}

function tukeyWeights(residuals: number[], points: Point[], scale: number): number[] {
  return residuals.map((residual, index) => {
    const sigma = Math.hypot(points[index].sigma, scale);
    if (!(sigma > 0) || !Number.isFinite(sigma)) return 0;
    const standardized = Math.abs(residual) / sigma;
    if (standardized > TUKEY_C) return 0;
    const factor = 1 - (standardized / TUKEY_C) ** 2;
    return factor * factor;
  });
}

type RobustLinearResult = LinearSolution & { weights: number[]; scale: number; residuals: number[] };

function fitRobustLinear(points: Point[]): RobustLinearResult | null {
  let weights = points.map((point) => point.qualityWeight);
  const initialRegression = weightedLinear(points, weights);
  if (!initialRegression) return null;
  let regression: LinearSolution = initialRegression;
  let residuals = points.map((point) => point.y - (regression.intercept + regression.slope * point.x));
  let scale = robustScale(residuals, weights);

  for (let iteration = 0; iteration < 30; iteration += 1) {
    const nextWeights = tukeyWeights(residuals, points, scale);
    const nextRegression = weightedLinear(points, nextWeights);
    if (!nextRegression) return null;
    const nextResiduals = points.map((point) => point.y - (nextRegression.intercept + nextRegression.slope * point.x));
    const nextScale = robustScale(nextResiduals, nextWeights);
    const weightChange = Math.max(...nextWeights.map((weight, index) => Math.abs(weight - weights[index])));
    weights = nextWeights;
    regression = nextRegression;
    residuals = nextResiduals;
    scale = nextScale;
    if (weightChange <= 1e-6) break;
  }

  return { ...regression, weights, scale, residuals };
}

function fitRobustConstant(points: Point[]): RobustLinearResult | null {
  let weights = points.map((point) => point.qualityWeight);
  let intercept = weightedMean(points.map((point) => point.y), weights);
  if (!Number.isFinite(intercept)) return null;
  let residuals = points.map((point) => point.y - intercept);
  let scale = robustScale(residuals, weights);

  for (let iteration = 0; iteration < 30; iteration += 1) {
    const nextWeights = tukeyWeights(residuals, points, scale);
    const nextIntercept = weightedMean(points.map((point) => point.y), nextWeights);
    if (!Number.isFinite(nextIntercept)) return null;
    const nextResiduals = points.map((point) => point.y - nextIntercept);
    const nextScale = robustScale(nextResiduals, nextWeights);
    const weightChange = Math.max(...nextWeights.map((weight, index) => Math.abs(weight - weights[index])));
    weights = nextWeights;
    intercept = nextIntercept;
    residuals = nextResiduals;
    scale = nextScale;
    if (weightChange <= 1e-6) break;
  }

  return { intercept, slope: 0, weights, scale, residuals };
}

function logGamma(value: number): number {
  const coefficients = [
    76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
    0.001208650973866179, -0.000005395239384953,
  ];
  let x = value;
  let y = value;
  let term = x + 5.5;
  term -= (x + 0.5) * Math.log(term);
  let sum = 1.000000000190015;
  for (const coefficient of coefficients) {
    y += 1;
    sum += coefficient / y;
  }
  x += 1;
  return -term + Math.log((2.5066282746310002 * sum) / x);
}

function studentLogLikelihood(residuals: number[], points: Point[], scale: number): number {
  const constant = logGamma((STUDENT_DF + 1) / 2) - logGamma(STUDENT_DF / 2) - 0.5 * Math.log(STUDENT_DF * Math.PI);
  return residuals.reduce((sum, residual, index) => {
    const sigma = Math.hypot(points[index].sigma, scale);
    const z = residual / sigma;
    return sum + constant - Math.log(sigma) - ((STUDENT_DF + 1) / 2) * Math.log1p((z * z) / STUDENT_DF);
  }, 0);
}

function weightedRSquared(residuals: number[], points: Point[], weights: number[]): number {
  const mean = weightedMean(points.map((point) => point.y), weights);
  const sse = residuals.reduce((sum, residual, index) => sum + weights[index] * residual * residual, 0);
  const sst = points.reduce((sum, point, index) => sum + weights[index] * (point.y - mean) ** 2, 0);
  if (sst === 0) return sse === 0 ? 1 : 0;
  return Math.max(0, 1 - sse / sst);
}

function inverseMatrix(matrix: number[][]): number[][] | null {
  const n = matrix.length;
  const augmented = matrix.map((row, rowIndex) => [
    ...row,
    ...Array.from({ length: n }, (_, columnIndex) => (rowIndex === columnIndex ? 1 : 0)),
  ]);

  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    const pivotValue = Math.abs(augmented[pivot][column]);
    const scale = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1);
    if (pivotValue <= scale * 1e-13) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    augmented[column] = augmented[column].map((value) => value / divisor);
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      augmented[row] = augmented[row].map((value, index) => value - factor * augmented[column][index]);
    }
  }

  return augmented.map((row) => row.slice(n));
}

function exponentialPrediction(point: Point, parameters: ExponentialParameters): number {
  const amplitude = Math.exp(parameters.logAmplitude);
  const timeConstant = Math.exp(parameters.logTimeConstant);
  return parameters.offset
    + parameters.driftSlope * point.x
    + parameters.sign * amplitude * Math.exp(-point.x / timeConstant);
}

function exponentialJacobian(point: Point, parameters: ExponentialParameters, kind: ExponentialKind): number[] {
  const amplitude = Math.exp(parameters.logAmplitude);
  const timeConstant = Math.exp(parameters.logTimeConstant);
  const exponential = Math.exp(-point.x / timeConstant);
  const derivatives = [
    parameters.sign * amplitude * exponential,
    parameters.sign * amplitude * exponential * (point.x / timeConstant),
    1,
  ];
  if (kind === 'drift') derivatives.push(point.x);
  return derivatives;
}

function solveLmStep(points: Point[], parameters: ExponentialParameters, weights: number[], scale: number, lambda: number, kind: ExponentialKind): number[] | null {
  const dimensions = kind === 'drift' ? 4 : 3;
  const hessian = Array.from({ length: dimensions }, () => Array(dimensions).fill(0));
  const gradient = Array(dimensions).fill(0);

  points.forEach((point, index) => {
    const residual = point.y - exponentialPrediction(point, parameters);
    const sigma = Math.hypot(point.sigma, scale);
    const w = weights[index] / (sigma * sigma);
    const jacobian = exponentialJacobian(point, parameters, kind).map((value) => value / sigma);
    jacobian.forEach((value, row) => {
      gradient[row] += w * value * residual;
      jacobian.forEach((other, column) => {
        hessian[row][column] += w * value * other;
      });
    });
  });

  const regularized = hessian.map((row, rowIndex) => row.map((value, columnIndex) => value + (rowIndex === columnIndex ? lambda * Math.max(value, 1e-12) : 0)));
  const inverse = inverseMatrix(regularized);
  if (!inverse) return null;
  return inverse.map((row) => row.reduce((sum, value, columnIndex) => sum + value * gradient[columnIndex], 0));
}

function solveWeightedLinearSystem(matrix: number[][], vector: number[]): number[] | null {
  const n = vector.length;
  const augmented = matrix.map((row, rowIndex) => [...row, vector[rowIndex]]);
  const scale = Math.max(1, ...matrix.flat().map((value) => Math.abs(value)));

  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) <= scale * 1e-13) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let index = column; index <= n; index += 1) augmented[column][index] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let index = column; index <= n; index += 1) augmented[row][index] -= factor * augmented[column][index];
    }
  }

  return augmented.map((row) => row[n]);
}

function fitGridExponential(points: Point[], kind: ExponentialKind, sign: 1 | -1, minLogTau: number, maxLogTau: number): ExponentialParameters | null {
  let bestParameters: ExponentialParameters | null = null;
  let bestCost = Number.POSITIVE_INFINITY;
  const dimensions = kind === 'drift' ? 3 : 2;

  for (let gridIndex = 0; gridIndex < 160; gridIndex += 1) {
    const logTau = minLogTau + ((maxLogTau - minLogTau) * gridIndex) / 159;
    const tau = Math.exp(logTau);
    const basis = points.map((point) => [sign * Math.exp(-point.x / tau), 1, ...(kind === 'drift' ? [point.x] : [])]);
    let weights = points.map((point) => point.qualityWeight);
    let scale = robustScale(points.map((point) => point.y), weights);
    let parameters: ExponentialParameters | null = null;

    for (let iteration = 0; iteration < 10; iteration += 1) {
      const matrix = Array.from({ length: dimensions }, () => Array(dimensions).fill(0));
      const vector = Array(dimensions).fill(0);
      points.forEach((point, pointIndex) => {
        const sigma = Math.hypot(point.sigma, scale);
        const w = weights[pointIndex] / (sigma * sigma);
        basis[pointIndex].forEach((basisValue, row) => {
          vector[row] += w * basisValue * point.y;
          basis[pointIndex].forEach((other, column) => {
            matrix[row][column] += w * basisValue * other;
          });
        });
      });
      const solution = solveWeightedLinearSystem(matrix, vector);
      if (!solution) return null;
      const [amplitude, offset, driftSlope = 0] = solution;
      if (!(amplitude > 0) || !solution.every(Number.isFinite)) {
        parameters = null;
        break;
      }
      parameters = {
        sign,
        logAmplitude: Math.log(amplitude),
        logTimeConstant: logTau,
        offset,
        driftSlope: kind === 'drift' ? driftSlope : 0,
      };
      const residuals = points.map((point) => point.y - exponentialPrediction(point, parameters!));
      const nextScale = robustScale(residuals, weights);
      const nextWeights = tukeyWeights(residuals, points, nextScale);
      const weightChange = Math.max(...nextWeights.map((weight, index) => Math.abs(weight - weights[index])));
      weights = nextWeights;
      scale = nextScale;
      if (weightChange <= 1e-6) break;
    }

    if (parameters) {
      const residuals = points.map((point) => point.y - exponentialPrediction(point, parameters!));
      const finalScale = robustScale(residuals, weights);
      const finalWeights = tukeyWeights(residuals, points, finalScale);
      const cost = residuals.reduce((sum, residual, index) => {
        const sigma = Math.hypot(points[index].sigma, finalScale);
        return sum + finalWeights[index] * (residual / sigma) ** 2;
      }, 0);
      if (cost < bestCost) {
        bestCost = cost;
        bestParameters = parameters;
      }
    }
  }

  return bestParameters;
}

function fitRobustExponential(
  points: Point[],
  kind: ExponentialKind,
  sign: 1 | -1,
  minLogTau: number,
  maxLogTau: number,
): Candidate | null {
  const gridFit = fitGridExponential(points, kind, sign, minLogTau, maxLogTau);
  if (!gridFit) return null;

  let parameters = gridFit;
  let residuals = points.map((point) => point.y - exponentialPrediction(point, parameters));
  let scale = robustScale(residuals, points.map((point) => point.qualityWeight));
  let weights = tukeyWeights(residuals, points, scale);
  let lambda = 1e-3;
  let best = { parameters, weights, scale };

  let bestCost = residuals.reduce((sum, residual, index) => {
    const sigma = Math.hypot(points[index].sigma, scale);
    return sum + weights[index] * (residual / sigma) ** 2;
  }, 0);

  for (let iteration = 0; iteration < 80; iteration += 1) {
    const step = solveLmStep(points, parameters, weights, scale, lambda, kind);
    if (!step) {
      lambda *= 3;
      continue;
    }
    const candidateParameters: ExponentialParameters = {
      ...parameters,
      logAmplitude: parameters.logAmplitude + step[0],
      logTimeConstant: parameters.logTimeConstant + step[1],
      offset: parameters.offset + step[2],
      driftSlope: kind === 'drift' ? parameters.driftSlope + step[3] : 0,
    };
    if (!Number.isFinite(candidateParameters.logAmplitude)
      || candidateParameters.logTimeConstant < minLogTau || candidateParameters.logTimeConstant > maxLogTau) {
      lambda *= 3;
      continue;
    }
    const candidateResiduals = points.map((point) => point.y - exponentialPrediction(point, candidateParameters));
    const candidateScale = robustScale(candidateResiduals, weights);
    const candidateWeights = tukeyWeights(candidateResiduals, points, candidateScale);
    const candidateCost = candidateResiduals.reduce((sum, residual, index) => {
      const sigma = Math.hypot(points[index].sigma, candidateScale);
      return sum + candidateWeights[index] * (residual / sigma) ** 2;
    }, 0);
    if (!(candidateCost < bestCost)) {
      lambda *= 3;
      continue;
    }

    const weightChange = Math.max(...candidateWeights.map((weight, index) => Math.abs(weight - weights[index])));
    parameters = candidateParameters;
    residuals = candidateResiduals;
    scale = candidateScale;
    weights = candidateWeights;
    bestCost = candidateCost;
    best = { parameters, weights, scale };
    lambda = Math.max(1e-8, lambda * 0.3);
    if (weightChange <= 1e-6 && Math.max(...step.map(Math.abs)) < 1e-7) break;
  }

  const k = kind === 'drift' ? 4 : 3;
  const support = best.weights.filter((weight) => weight > 0).length;
  const effN = effectiveN(best.weights);
  if (support < k + 1 || effN <= k + 1) return null;

  const dimensions = kind === 'drift' ? 4 : 3;
  const bread = Array.from({ length: dimensions }, () => Array(dimensions).fill(0));
  const meat = Array.from({ length: dimensions }, () => Array(dimensions).fill(0));
  points.forEach((point, index) => {
    const sigma = Math.hypot(point.sigma, best.scale);
    const weight = best.weights[index] / (sigma * sigma);
    const jacobian = exponentialJacobian(point, best.parameters, kind);
    jacobian.forEach((rowValue, row) => {
      jacobian.forEach((columnValue, column) => {
        bread[row][column] += weight * rowValue * columnValue;
        meat[row][column] += rowValue * columnValue * weight * weight * residuals[index] * residuals[index];
      });
    });
  });
  const breadInverse = inverseMatrix(bread);
  if (!breadInverse) return null;
  const robustCovariance = Array.from({ length: dimensions }, () => Array(dimensions).fill(0));
  for (let row = 0; row < dimensions; row += 1) {
    for (let column = 0; column < dimensions; column += 1) {
      robustCovariance[row][column] = breadInverse[row].reduce(
        (sum, left, middleRow) => sum + left * meat[middleRow].reduce(
          (inner, meatValue, middleColumn) => inner + meatValue * breadInverse[column][middleColumn],
          0,
        ),
        0,
      );
    }
  }
  const variance = robustCovariance[1][1];
  if (!Number.isFinite(variance) || variance <= 0) return null;
  const standardErrorLogTau = Math.sqrt(variance);
  if (!Number.isFinite(standardErrorLogTau)) return null;

  return finalizeCandidate(
    kind === 'drift'
      ? sign === 1 ? 'exponential-drift-decay' : 'exponential-drift-growth'
      : sign === 1 ? 'exponential-constant-decay' : 'exponential-constant-growth',
    kind,
    k,
    points,
    best.weights,
    best.scale,
    residuals,
    {
      intercept: best.parameters.offset,
      slope: best.parameters.driftSlope,
      exponential: {
        sign: best.parameters.sign,
        amplitude: Math.exp(best.parameters.logAmplitude),
        timeConstant: Math.exp(best.parameters.logTimeConstant),
        offset: best.parameters.offset,
        driftSlope: best.parameters.driftSlope,
        standardErrorLogTau,
      },
    },
  );
}

function finalizeCandidate(
  name: Candidate['name'],
  kind: ExponentialKind | null,
  k: number,
  points: Point[],
  weights: number[],
  scale: number,
  residuals: number[],
  options: { intercept: number; slope: number; exponential?: Candidate['exponential'] },
): Candidate {
  const n = points.length;
  const logLikelihood = studentLogLikelihood(residuals, points, scale);
  const aic = 2 * k - 2 * logLikelihood;
  const aicC = n > k + 1 ? aic + (2 * k * (k + 1)) / (n - k - 1) : Number.POSITIVE_INFINITY;
  return {
    name,
    kind,
    k,
    intercept: options.intercept,
    slope: options.slope,
    weights,
    scale,
    residuals,
    rSquared: weightedRSquared(residuals, points, weights),
    support: weights.filter((weight) => weight > 0).length,
    effectiveN: effectiveN(weights),
    aicC,
    logLikelihood,
    exponential: options.exponential,
  };
}

function baselineCandidate(points: Point[], name: 'constant' | 'linear'): Candidate | null {
  const fit = name === 'constant' ? fitRobustConstant(points) : fitRobustLinear(points);
  if (!fit) return null;
  const residuals = points.map((point) => point.y - (fit.intercept + fit.slope * point.x));
  const k = name === 'constant' ? 1 : 2;
  const support = fit.weights.filter((weight) => weight > 0).length;
  if (support < k + 1 || effectiveN(fit.weights) <= k + 1) return null;
  return finalizeCandidate(name, null, k, points, fit.weights, fit.scale, residuals, { intercept: fit.intercept, slope: fit.slope });
}

function modelWeights(candidates: Candidate[]): { candidate: Candidate; weight: number }[] {
  const finite = candidates.filter((candidate) => Number.isFinite(candidate.aicC));
  if (!finite.length) return [];
  const minimum = Math.min(...finite.map((candidate) => candidate.aicC));
  const weighted = finite.map((candidate) => ({ candidate, weight: Math.exp(-0.5 * (candidate.aicC - minimum)) }));
  const total = weighted.reduce((sum, item) => sum + item.weight, 0);
  return weighted.map((item) => ({ candidate: item.candidate, weight: item.weight / total }))
    .sort((a, b) => b.weight - a.weight);
}

function inlierFraction(points: Point[], weights: number[]): number {
  const denominator = points.reduce((sum, point) => sum + point.qualityWeight, 0);
  const numerator = weights.reduce((sum, weight, index) => sum + weight * points[index].qualityWeight, 0);
  return denominator === 0 ? 0 : numerator / denominator;
}

export function fitExponential(samples: SynchronizedSample[], startNs?: number, endNs?: number): DecayFit {
  const selected = samples
    .filter((sample) => (startNs === undefined || sample.timestampNs >= startNs) && (endNs === undefined || sample.timestampNs < endNs))
    .sort((a, b) => a.timestampNs - b.timestampNs || a.syncIndex - b.syncIndex);

  if (!selected.length) {
    return makeResult('insufficient-points', 'No samples are inside the fit interval.', startNs, endNs, 0, 0);
  }

  const points: Point[] = [];
  let invalidPoints = 0;
  for (const sample of selected) {
    const qualityWeight = sample.quality === 'suspect' ? 0.5 : sample.quality === 'invalid' ? 0 : 1;
    if (!Number.isFinite(sample.timestampNs) || !Number.isFinite(sample.amplitude)
      || !Number.isFinite(sample.uncertainty) || sample.uncertainty <= 0 || qualityWeight === 0) {
      invalidPoints += 1;
      continue;
    }
    points.push({ x: sample.timestampNs, y: sample.amplitude, sigma: sample.uncertainty, qualityWeight });
  }

  if (!points.length) {
    return makeResult('invalid-data', 'Every selected sample is invalid or has non-positive measurement uncertainty.', startNs, endNs, 0, selected.length);
  }

  const fallbackOffset = weightedMedian(points.map((point) => point.y), points.map((point) => point.qualityWeight));
  const distinctTimes = new Set(points.map((point) => point.x)).size;
  if (points.length < MIN_POINTS || distinctTimes < MIN_DISTINCT_TIMES) {
    return makeResult(
      'insufficient-points',
      'At least three valid samples spanning three distinct times are required to describe a trend or decay.',
      startNs,
      endNs,
      points.length,
      invalidPoints,
      { offset: fallbackOffset },
    );
  }

  const firstTime = points[0].x;
  points.forEach((point) => { point.x -= firstTime; });
  const span = points.at(-1)!.x;
  if (!(span > 0) || !Number.isFinite(span)) {
    return makeResult('invalid-data', 'The fit interval does not contain a finite positive time span.', startNs, endNs, 0, selected.length, { offset: fallbackOffset });
  }

  const spacings = points.map((point, index) => index > 0 ? point.x - points[index - 1].x : Number.POSITIVE_INFINITY)
    .filter((spacing) => Number.isFinite(spacing) && spacing > 0)
    .sort((a, b) => a - b);
  const minLogTau = Math.log(Math.max(spacings[0] / 2, Number.MIN_VALUE));
  const maxLogTau = Math.log(Math.max(span * 100, Math.exp(minLogTau) * 10));

  const candidates = [
    baselineCandidate(points, 'constant'),
    baselineCandidate(points, 'linear'),
    fitRobustExponential(points, 'constant', 1, minLogTau, maxLogTau),
    fitRobustExponential(points, 'constant', -1, minLogTau, maxLogTau),
    fitRobustExponential(points, 'drift', 1, minLogTau, maxLogTau),
    fitRobustExponential(points, 'drift', -1, minLogTau, maxLogTau),
  ].filter((candidate): candidate is Candidate => candidate !== null);

  if (!candidates.length) {
    return makeResult('invalid-data', 'No baseline or exponential model could be fitted to the selected data.', startNs, endNs, 0, selected.length, { offset: fallbackOffset });
  }

  const weightedCandidates = modelWeights(candidates);
  const probability = new Map(weightedCandidates.map((item) => [item.candidate.name, item.weight]));
  const exponentialProbability = weightedCandidates
    .filter((item) => item.candidate.name.startsWith('exponential-'))
    .reduce((sum, item) => sum + item.weight, 0);
  const constantProbability = probability.get('constant') ?? 0;
  const linearProbability = probability.get('linear') ?? 0;
  const winner = weightedCandidates[0].candidate;
  const winnerProbability = weightedCandidates[0].weight;
  const rejectedOutliers = winner.weights.filter((weight) => weight === 0).length;
  const usedPoints = winner.support;
  const rejectedPoints = invalidPoints + rejectedOutliers;
  const inliers = inlierFraction(points, winner.weights);

  if (!winner.exponential || winner.exponential.sign === -1 || exponentialProbability < 0.5) {
    const constant = candidates.find((candidate) => candidate.name === 'constant');
    const linear = candidates.find((candidate) => candidate.name === 'linear');
    const preferLinear = winner.exponential ? winner.kind === 'drift' : winner.name === 'linear';
    const selectedBaseline = (preferLinear ? linear ?? constant : constant ?? linear) ?? winner;
    const isLinear = selectedBaseline.name === 'linear';
    const reason = winner.exponential?.sign === -1
      ? 'The best exponential trend is increasing rather than decaying.'
      : exponentialProbability < 0.5
        ? 'A constant or drifting baseline is supported as well as, or better than, an exponential decay.'
        : 'The selected data are described by a baseline, not an exponential decay.';
    return makeResult(
      'non-decaying',
      reason,
      startNs,
      endNs,
      usedPoints,
      rejectedPoints,
      {
        offset: selectedBaseline.intercept,
        driftSlopePerNs: isLinear ? selectedBaseline.slope : null,
        baselineStrategy: isLinear ? 'linear-drift' : 'constant',
        rSquared: selectedBaseline?.rSquared ?? null,
        modelEvidence: { exponential: exponentialProbability, constantBaseline: constantProbability, linearBaseline: linearProbability },
        fitQuality: { modelEvidence: winnerProbability, inlierFraction: inliers, identifiability: null, intervalPrecision: null },
        warnings: rejectedOutliers > 0 ? ['robust-outliers-rejected'] : [],
      },
    );
  }

  const exponential = winner.exponential;
  const logTau = Math.log(exponential.timeConstant);
  const lowerNs = Math.exp(logTau - Z_95 * exponential.standardErrorLogTau);
  const upperNs = Math.exp(logTau + Z_95 * exponential.standardErrorLogTau);
  if (![lowerNs, upperNs, exponential.amplitude, exponential.offset].every(Number.isFinite)) {
    return makeResult(
      'invalid-data',
      'The exponential fit converged, but its time-constant interval is not numerically finite.',
      startNs,
      endNs,
      usedPoints,
      rejectedPoints,
      {
        offset: exponential.offset,
        driftSlopePerNs: winner.kind === 'drift' ? exponential.driftSlope : null,
        baselineStrategy: winner.kind === 'drift' ? 'linear-drift' : 'constant',
        rSquared: winner.rSquared,
        modelEvidence: { exponential: exponentialProbability, constantBaseline: constantProbability, linearBaseline: linearProbability },
        fitQuality: { modelEvidence: exponentialProbability, inlierFraction: inliers, identifiability: null, intervalPrecision: null },
        warnings: ['time-constant-interval-not-finite'],
      },
    );
  }

  const identifiability = 1 - Math.exp(-span / exponential.timeConstant);
  const intervalPrecision = Math.exp(-2 * Z_95 * exponential.standardErrorLogTau);
  const confidenceScore = Math.pow(
    Math.max(Number.MIN_VALUE, exponentialProbability)
      * Math.max(Number.MIN_VALUE, inliers)
      * Math.max(Number.MIN_VALUE, identifiability)
      * Math.max(Number.MIN_VALUE, intervalPrecision),
    0.25,
  );
  const status = winner.kind === 'drift' ? 'fitted-with-drift' : 'fitted';
  const reason = winner.kind === 'drift'
    ? 'Exponential decay with a jointly estimated linear baseline drift was the best-supported model.'
    : 'Exponential decay with a jointly estimated constant baseline was the best-supported model.';
  const warnings: string[] = [];
  if (rejectedOutliers > 0) warnings.push('robust-outliers-rejected');
  if (exponentialProbability < 0.5) warnings.push('model-evidence-ambiguous');
  if (confidenceScore < 0.5) warnings.push('low-fit-confidence');

  return makeResult(status, reason, startNs, endNs, usedPoints, rejectedPoints, {
    amplitude0: exponential.amplitude,
    timeConstantNs: exponential.timeConstant,
    offset: exponential.offset,
    driftSlopePerNs: winner.kind === 'drift' ? exponential.driftSlope : null,
    baselineStrategy: winner.kind === 'drift' ? 'linear-drift' : 'constant',
    rSquared: winner.rSquared,
    confidence: { level: 0.95, lowerNs, upperNs, score: confidenceScore },
    modelEvidence: { exponential: exponentialProbability, constantBaseline: constantProbability, linearBaseline: linearProbability },
    fitQuality: { modelEvidence: exponentialProbability, inlierFraction: inliers, identifiability, intervalPrecision },
    warnings,
  });
}
