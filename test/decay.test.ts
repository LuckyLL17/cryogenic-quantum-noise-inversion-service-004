import test from 'node:test';
import assert from 'node:assert/strict';
import { fitExponential } from '../src/domain/fit/decay.ts';
import { analyzeExperiment } from '../src/application/analysis-service.ts';
import type { CalibrationProfile, RawSample, SynchronizedSample } from '../src/domain/types.ts';

function synchronizedSample(index: number, timestampNs: number, amplitude: number, uncertainty = 0.01): SynchronizedSample {
  return {
    captureId: `capture-${index}`,
    experimentId: 'exp-decay',
    channelId: 'ch-a',
    role: 'readout-i',
    triggerId: `trigger-${index}`,
    timestampNs,
    receivedAtNs: timestampNs,
    sequence: index,
    sampleRateHz: 10_000_000,
    i: amplitude,
    q: 0,
    quality: 'good',
    calibratedI: amplitude,
    calibratedQ: 0,
    amplitude,
    phaseRad: 0,
    uncertainty,
    calibrationProfileId: 'cal-1',
    syncIndex: index,
  };
}

test('fits a clean exponential with a jointly estimated constant baseline', () => {
  const samples = Array.from({ length: 30 }, (_, index) => {
    const t = index * 100;
    return synchronizedSample(index, t, 0.2 + 2 * Math.exp(-t / 1_000) + 0.002 * ((index % 3) - 1));
  });
  const fit = fitExponential(samples);

  assert.equal(fit.status, 'fitted');
  assert.equal(fit.baselineStrategy, 'constant');
  assert.ok(fit.timeConstantNs !== null);
  assert.ok(Math.abs(fit.timeConstantNs! - 1_000) < 120);
  assert.ok(fit.amplitude0 !== null && Math.abs(fit.amplitude0 - 2) < 0.1);
  assert.ok(Math.abs(fit.offset - 0.2) < 0.03);
  assert.equal(fit.usedPoints + fit.rejectedPoints, samples.length);
  assert.ok(fit.confidence.lowerNs !== null && fit.confidence.upperNs !== null);
  assert.ok(fit.confidence.lowerNs! < fit.timeConstantNs! && fit.confidence.upperNs! > fit.timeConstantNs!);
});

test('rejects preparation spikes robustly without changing the physical fit interval', () => {
  const samples = Array.from({ length: 30 }, (_, index) => {
    const t = index * 100;
    const spike = index === 2 || index === 7 || index === 20 ? 8 : 0;
    return synchronizedSample(index, t, 0.2 + 2 * Math.exp(-t / 1_000) + spike);
  });
  const fit = fitExponential(samples, 0, 3_000);

  assert.equal(fit.status, 'fitted');
  assert.equal(fit.fitStartNs, 0);
  assert.equal(fit.fitEndNs, 3_000);
  assert.equal(fit.outlierStrategy, 'tukey-bisquare');
  assert.equal(fit.rejectedPoints, 3);
  assert.ok(fit.timeConstantNs !== null && Math.abs(fit.timeConstantNs - 1_000) < 180);
  assert.ok(fit.warnings.includes('robust-outliers-rejected'));
});

test('identifies exponential decay on top of a linear baseline drift', () => {
  const samples = Array.from({ length: 30 }, (_, index) => {
    const t = index * 100;
    return synchronizedSample(index, t, 0.1 + 0.0004 * t + 2 * Math.exp(-t / 1_000) + 0.001 * ((index % 3) - 1));
  });
  const fit = fitExponential(samples);

  assert.equal(fit.status, 'fitted-with-drift');
  assert.equal(fit.baselineStrategy, 'linear-drift');
  assert.ok(fit.timeConstantNs !== null && Math.abs(fit.timeConstantNs - 1_000) < 180);
  assert.ok(fit.driftSlopePerNs !== null && Math.abs(fit.driftSlopePerNs - 0.0004) < 5e-5);
  assert.ok(fit.amplitude0 !== null && Math.abs(fit.amplitude0 - 2) < 0.15);
});

test('reports insufficient points instead of returning a time constant', () => {
  const fit = fitExponential([
    synchronizedSample(0, 0, 1),
    synchronizedSample(1, 100, 0.8),
  ]);

  assert.equal(fit.status, 'insufficient-points');
  assert.equal(fit.timeConstantNs, null);
  assert.equal(fit.confidence.lowerNs, null);
  assert.equal(fit.confidence.upperNs, null);
  assert.equal(fit.usedPoints, 2);
  assert.equal(fit.rejectedPoints, 0);
});

test('classifies flat and monotonic non-decaying data without fabricating a lifetime', () => {
  const flat = Array.from({ length: 20 }, (_, index) => synchronizedSample(index, index * 100, 0.5 + 0.001 * (index % 2)));
  const flatFit = fitExponential(flat);
  assert.equal(flatFit.status, 'non-decaying');
  assert.equal(flatFit.timeConstantNs, null);
  assert.equal(flatFit.baselineStrategy, 'constant');
  assert.ok(flatFit.modelEvidence.constantBaseline > flatFit.modelEvidence.exponential);

  const rising = Array.from({ length: 20 }, (_, index) => synchronizedSample(index, index * 100, 0.1 + 0.002 * index * 100));
  const risingFit = fitExponential(rising);
  assert.equal(risingFit.status, 'non-decaying');
  assert.equal(risingFit.timeConstantNs, null);
  assert.equal(risingFit.baselineStrategy, 'linear-drift');
  assert.ok(risingFit.driftSlopePerNs !== null && risingFit.driftSlopePerNs > 0);
});

test('continues to omit decay fitting for noise-only analyses', () => {
  const calibration: CalibrationProfile = {
    profileId: 'cal-1',
    channelId: 'ch-a',
    validFromNs: 0,
    adcScale: 1,
    iOffset: 0,
    qOffset: 0,
    gainCorrection: 1,
    phaseCorrectionRad: 0,
    temperatureCoefficientPerK: 0,
    referenceTemperatureK: 0.02,
    systematicUncertainty: 0,
    sampleUncertainty: 0.01,
  };
  const samples: RawSample[] = Array.from({ length: 32 }, (_, index) => ({
    captureId: `capture-${index}`,
    experimentId: 'exp-noise-only',
    channelId: 'ch-a',
    role: 'readout-i',
    triggerId: `trigger-${Math.floor(index / 4)}`,
    timestampNs: index * 1_000_000,
    receivedAtNs: index * 1_000_000,
    sequence: index,
    sampleRateHz: 1_000,
    i: Math.cos(index / 4),
    q: Math.sin(index / 4),
    quality: 'good',
  }));
  const result = analyzeExperiment({
    experimentId: 'exp-noise-only',
    kind: 'noise-only',
    samples,
    calibrations: [calibration],
    analysis: { windowSize: 16 },
  });
  assert.equal(result.metrics.decay, null);
});
