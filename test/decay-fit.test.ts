import test from 'node:test';
import assert from 'node:assert/strict';
import { fitExponential } from '../src/domain/fit/decay.ts';
import type { SynchronizedSample } from '../src/domain/types.ts';

function makeSamples(values: Array<number | { t: number; y: number; quality?: 'good' | 'suspect' }>, dtNs = 1_000_000, uncertainty = 0.01): SynchronizedSample[] {
  return values.map((entry, index) => {
    const item = typeof entry === 'number' ? { t: index * dtNs, y: entry } : entry;
    return {
      captureId: `c-${index}`, experimentId: 'exp-1', channelId: 'ch-a', role: 'readout-i' as const,
      triggerId: 't-1', timestampNs: item.t, receivedAtNs: item.t + 100, sequence: index,
      sampleRateHz: 1_000, i: item.y, q: 0, quality: item.quality ?? 'good',
      calibratedI: item.y, calibratedQ: 0, amplitude: item.y, phaseRad: 0,
      uncertainty, calibrationProfileId: 'cal-1', syncIndex: index,
    };
  });
}

// Deterministic pseudo-noise so thresholds cannot be tuned against a lucky draw.
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

test('clean exponential decay recovers the time constant with a finite interval and high confidence', () => {
  const tau = 3_000_000;
  const values = Array.from({ length: 40 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau));
  const fit = fitExponential(makeSamples(values));
  assert.equal(fit.status, 'fitted');
  assert.equal(fit.confidence, 'high');
  assert.notEqual(fit.timeConstantNs, null);
  assert.notEqual(fit.confidenceInterval.lowerNs, null);
  assert.notEqual(fit.confidenceInterval.upperNs, null);
  const fittedTau = fit.timeConstantNs as number;
  const lower = fit.confidenceInterval.lowerNs as number;
  const upper = fit.confidenceInterval.upperNs as number;
  assert.ok(Math.abs(fittedTau - tau) / tau < 0.05, `tau=${fittedTau}`);
  assert.ok(fit.amplitude0 && Math.abs(fit.amplitude0 - 9) / 9 < 0.02);
  assert.ok(lower !== null && upper !== null);
  assert.ok(lower < fittedTau && upper > fittedTau);
  assert.equal(fit.usedPoints, 40);
  assert.equal(fit.rejectedPoints, 0);
  assert.ok(fit.rSquared !== null && fit.rSquared > 0.999);
});

test('preparation-stage residue before the fit window is excluded through fitStartNs and does not bias tau', () => {
  const tau = 2_500_000;
  const values = [50, 30, 12] // preparation residue
    .concat(Array.from({ length: 36 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau)));
  const biased = fitExponential(makeSamples(values));
  const windowed = fitExponential(makeSamples(values), 3_000_000);
  assert.equal(windowed.usedPoints, 36);
  assert.equal(windowed.rejectedPoints, 0);
  assert.ok(windowed.status === 'fitted' && windowed.timeConstantNs !== null);
  const windowedTau = windowed.timeConstantNs as number;
  assert.ok(Math.abs(windowedTau - tau) / tau < 0.08);
  assert.ok(biased.timeConstantNs === null || windowedTau === null
    || Math.abs(biased.timeConstantNs - tau) / tau > Math.abs(windowedTau - tau) / tau || biased.confidence === 'low');
});

test('sparse gross outliers inside the window are rejected and repeated runs agree on tau', () => {
  const tau = 4_000_000;
  const rand = lcg(42);
  const values = Array.from({ length: 60 }, (_, index) => {
    const clean = 1 + 9 * Math.exp(-(index * 1_000_000) / tau) + (rand() - 0.5) * 0.04;
    if (index % 13 === 4) return clean + 25; // glitch spikes
    return clean;
  });
  const first = fitExponential(makeSamples(values));
  const second = fitExponential(makeSamples(values));
  assert.equal(first.status, 'fitted');
  assert.equal(first.timeConstantNs, second.timeConstantNs);
  assert.ok(Math.abs(first.timeConstantNs! - tau) / tau < 0.12, `tau=${first.timeConstantNs}`);
  assert.equal(first.rejectedPoints, values.filter((_, index) => index % 13 === 4).length);
  assert.equal(first.usedPoints, 60 - first.rejectedPoints);
});

test('a single downward prep dip does not pull the fitted baseline away', () => {
  const tau = 5_000_000;
  const values = Array.from({ length: 40 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau));
  values[2] = -20;
  const fit = fitExponential(makeSamples(values));
  assert.equal(fit.status, 'fitted');
  assert.equal(fit.rejectedPoints, 1);
  assert.ok(Math.abs(fit.offset - 1) < 0.05, `offset=${fit.offset}`);
  assert.ok(Math.abs(fit.timeConstantNs! - tau) / tau < 0.1);
});

test('a drifting baseline with embedded decay is reported as baseline-drift, not as a confident lifetime', () => {
  const tau = 3_000_000;
  const values = Array.from({ length: 40 }, (_, index) => 1 + 0.12 * index + 4 * Math.exp(-(index * 1_000_000) / tau));
  const fit = fitExponential(makeSamples(values));
  assert.ok(fit.status === 'baseline-drift' || fit.confidence === 'low', `status=${fit.status}`);
  assert.ok(fit.baselineSlopePerNs > 0);
  // The result must carry an explainable reason instead of looking like a clean success.
  assert.ok(fit.reasons.length > 0);
});

test('fewer than three usable points returns insufficient-points and never a tau', () => {
  const two = fitExponential(makeSamples([1, 0.9]));
  assert.equal(two.status, 'insufficient-points');
  assert.equal(two.timeConstantNs, null);
  assert.equal(two.usedPoints, 2);
  const empty = fitExponential(makeSamples(Array.from({ length: 10 }, (_, i) => i)), 500_000_000);
  assert.equal(empty.status, 'invalid');
  assert.ok(empty.reasons.includes('no-samples-in-window'));
});

test('a flat noisy trace is classified non-decaying with null time constant', () => {
  const rand = lcg(7);
  const values = Array.from({ length: 50 }, () => 2 + (rand() - 0.5) * 0.02);
  const fit = fitExponential(makeSamples(values));
  assert.equal(fit.status, 'non-decaying');
  assert.equal(fit.confidence, 'none');
  assert.equal(fit.timeConstantNs, null);
  assert.equal(fit.amplitude0, null);
  assert.ok(fit.reasons.includes('no-decay-detected'));
  assert.ok(Math.abs(fit.offset - 2) < 0.05);
});

test('a purely rising non-decaying curve is baseline-drift rather than an exponential lifetime', () => {
  const values = Array.from({ length: 40 }, (_, index) => 1 + 0.2 * index);
  const fit = fitExponential(makeSamples(values));
  assert.equal(fit.status, 'baseline-drift');
  assert.equal(fit.timeConstantNs, null);
  assert.ok(fit.baselineSlopePerNs > 0);
});

test('a decay observed over much less than one time constant is flagged instead of praised', () => {
  const tau = 200_000_000; // 50x the observed window
  const values = Array.from({ length: 30 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau));
  const fit = fitExponential(makeSamples(values));
  assert.notEqual(fit.confidence, 'high');
  assert.ok(fit.timeConstantNs === null
    || fit.reasons.some((reason) => reason === 'wide-confidence-interval' || reason === 'time-constant-at-window-edge' || reason === 'time-constant-unconstrained'),
    `reasons=${fit.reasons.join(',')}`);
  if (fit.timeConstantNs !== null) assert.ok(fit.confidenceInterval.upperNs === null || fit.confidenceInterval.upperNs >= tau);
});

test('non-finite samples count as rejected and do not crash the fit', () => {
  const tau = 3_000_000;
  const values = Array.from({ length: 20 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau));
  const samples = makeSamples(values);
  samples[5] = { ...samples[5], amplitude: Number.NaN };
  samples[10] = { ...samples[10], amplitude: Number.POSITIVE_INFINITY };
  const fit = fitExponential(samples);
  assert.equal(fit.rejectedPoints, 2);
  assert.ok(fit.reasons.includes('nonfinite-points-removed'));
  assert.equal(fit.status, 'fitted');
});

test('suspect-quality points contribute but cannot dominate a clean fit', () => {
  const tau = 3_000_000;
  const values = Array.from({ length: 30 }, (_, index) => ({ t: index * 1_000_000, y: 1 + 9 * Math.exp(-(index * 1_000_000) / tau), quality: 'good' as 'good' | 'suspect' }));
  values[7].y = 40;
  values[7].quality = 'suspect';
  const fit = fitExponential(makeSamples(values));
  assert.equal(fit.rejectedPoints, 1);
  assert.ok(Math.abs(fit.timeConstantNs! - tau) / tau < 0.1);
});

test('repeated noisy measurements of one device agree on tau within their intervals', () => {
  const tau = 3_500_000;
  const noiseSigma = 0.15;
  const estimates: Array<{ tau: number; lower: number; upper: number }> = [];
  for (let run = 0; run < 8; run += 1) {
    const rand = lcg(1000 + run);
    const values = Array.from({ length: 48 }, (_, index) => 1 + 9 * Math.exp(-(index * 1_000_000) / tau) + (rand() - 0.5) * 2 * noiseSigma);
    // Every run also carries a random preparation spike near the start.
    values[run * 3 % 10] += 15 + rand() * 10;
    const fit = fitExponential(makeSamples(values, 1_000_000, noiseSigma));
    assert.equal(fit.status, 'fitted', `run ${run}: ${fit.status} ${fit.reasons.join(',')}`);
    assert.equal(fit.rejectedPoints, 1, `run ${run} rejected ${fit.rejectedPoints}`);
    assert.ok(fit.confidenceInterval.lowerNs !== null && fit.confidenceInterval.upperNs !== null);
    estimates.push({ tau: fit.timeConstantNs!, lower: fit.confidenceInterval.lowerNs!, upper: fit.confidenceInterval.upperNs! });
  }
  const taus = estimates.map((item) => item.tau);
  const spread = (Math.max(...taus) - Math.min(...taus)) / tau;
  assert.ok(spread < 0.15, `tau spread across runs: ${(spread * 100).toFixed(1)}%`);
  for (const estimate of estimates) {
    assert.ok(estimate.lower <= tau && tau <= estimate.upper, `true tau outside [${estimate.lower}, ${estimate.upper}]`);
  }
});

test('a slow drift plus noise is not mistaken for a confident time constant', () => {
  const rand = lcg(31337);
  const values = Array.from({ length: 44 }, (_, index) => -0.05 * index + (rand() - 0.5) * 0.03);
  const fit = fitExponential(makeSamples(values));
  assert.ok(fit.status === 'baseline-drift' || fit.status === 'non-decaying', fit.status);
  assert.equal(fit.timeConstantNs, null);
  if (fit.status === 'baseline-drift') {
    assert.ok(fit.baselineSlopePerNs < 0);
    assert.ok(fit.reasons.includes('drift-without-decay'));
  }
});
