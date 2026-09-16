import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { BrainBodyBridge, createZeroDecoder, encodeObservation } from '../src/bridge/controller.mjs';

const body = JSON.parse(fs.readFileSync(new URL('../config/body-map.json', import.meta.url)));
// Fixture identifiers only; not a real circuit or a trained controller.
const binding = {
  id: 'test-fixture-v1', dataset: 'test-fixture',
  neurons: [{ id: '720575940000000001' }, { id: '720575940000000002' }],
};
const frame = (rates = {}, encoding = 'sparse-zero') => ({
  bindingId: binding.id, dataset: binding.dataset, encoding, ratesHz: rates,
});
const decoderWith = (id, weight) => {
  const decoder = createZeroDecoder(binding, body);
  decoder.weights[decoder.jointIds.indexOf(id)][0] = weight;
  decoder.provenance = 'synthetic-test';
  return decoder;
};

test('zero decoder stays neutral, even with active neurons', () => {
  const bridge = new BrainBodyBridge(binding, body);
  const result = bridge.step(frame({ [binding.neurons[0].id]: 100 }), 0.02);
  assert.deepEqual(result.angles, body.joints.map(joint => joint.neutral));
});

test('ID strings distinguish adjacent 64-bit IDs and reject numeric IDs', () => {
  assert.equal(Number(binding.neurons[0].id), Number(binding.neurons[1].id));
  assert.throws(() => new BrainBodyBridge({
    ...binding, neurons: [{ id: Number(binding.neurons[0].id) }],
  }, body), /strings/);
  const bridge = new BrainBodyBridge(binding, body, decoderWith('left_hip_pitch', 1));
  const result = bridge.step(frame({ [binding.neurons[1].id]: 100 }), 0.02);
  assert.equal(result.jointAngles.left_hip_pitch, 0);
});

test('one input changes only its configured output; low-pass follows elapsed time', () => {
  const bridge = new BrainBodyBridge(binding, body, decoderWith('left_hip_pitch', 1));
  const result = bridge.step(frame({ [binding.neurons[0].id]: 100 }), 0.02);
  assert.ok(Math.abs(result.filteredHz[0] - 100 * (1 - Math.exp(-0.4))) < 1e-10);
  assert.ok(result.jointAngles.left_hip_pitch > 0);
  assert.equal(result.jointAngles.right_hip_pitch, 0);
  for (const joint of body.joints) {
    if (joint.id !== 'left_hip_pitch') assert.equal(result.jointAngles[joint.id], joint.neutral);
  }
});

test('negative weights produce opposite actions around the neutral pose', () => {
  const bridge = new BrainBodyBridge(binding, body, decoderWith('left_hip_pitch', -1));
  assert.ok(bridge.step(frame({ [binding.neurons[0].id]: 100 }), 0.02).jointAngles.left_hip_pitch < 0);
});

test('large firing rates cannot exceed angular or speed limits', () => {
  const decoder = createZeroDecoder(binding, body);
  decoder.weights.forEach(row => { row[0] = 10000; });
  const bridge = new BrainBodyBridge(binding, body, decoder);
  let previous = body.joints.map(joint => joint.neutral);
  for (let i = 0; i < 150; i++) {
    const result = bridge.step(frame({ [binding.neurons[0].id]: 1e8 }), 0.02);
    result.angles.forEach((angle, j) => {
      assert.ok(angle >= body.joints[j].min && angle <= body.joints[j].max);
      assert.ok(Math.abs(angle - previous[j]) <= body.joints[j].maxSpeed * 0.02 + 1e-12);
    });
    assert.ok(result.filteredHz[0] <= body.filter.maxRateHz);
    previous = result.angles;
  }
});

test('dense frames require every ID; sparse frames must explicitly declare zero omissions', () => {
  const bridge = new BrainBodyBridge(binding, body);
  assert.throws(() => bridge.step(frame({}, 'dense'), 0.02), /Missing/);
  const invalid = frame();
  delete invalid.encoding;
  assert.throws(() => bridge.step(invalid, 0.02), /Declare/);
  assert.doesNotThrow(() => bridge.step(frame(), 0.02));
});

test('bad input is rejected before state advances', () => {
  const bridge = new BrainBodyBridge(binding, body);
  for (const bad of [
    { ...frame(), dataset: 'malecns' }, { ...frame(), bindingId: 'different' },
    frame({ '1': 4 }), frame({ [binding.neurons[0].id]: NaN }),
    frame({ [binding.neurons[0].id]: -1 }),
  ]) assert.throws(() => bridge.step(bad, 0.02));
  for (const dt of [0, -1, 1, NaN]) assert.throws(() => bridge.step(frame(), dt));
  assert.equal(bridge.elapsedSeconds, 0);
  assert.deepEqual(bridge.filteredHz, [0, 0]);
});

test('decoder rejects incompatible IDs, ordering, dimensions, and nonfinite weights', () => {
  const bridge = new BrainBodyBridge(binding, body);
  for (const mutate of [
    d => { d.bindingId = 'other'; },
    d => { d.bodyMapId = 'other'; },
    d => { d.neuronIds.reverse(); },
    d => { d.jointIds.reverse(); },
    d => { d.weights.pop(); },
    d => { d.weights[0].pop(); },
    d => { d.weights[0][0] = Infinity; },
  ]) {
    const decoder = createZeroDecoder(binding, body);
    mutate(decoder);
    assert.throws(() => bridge.setDecoder(decoder));
  }
});

test('reset produces repeatable state', () => {
  const bridge = new BrainBodyBridge(binding, body, decoderWith('left_hip_pitch', 1));
  const input = frame({ [binding.neurons[0].id]: 100 });
  const first = bridge.step(input, 0.02);
  bridge.step(input, 0.02);
  bridge.reset();
  assert.deepEqual(bridge.step(input, 0.02), first);
});

test('feedback contract has 50 values and never fabricates contact or gravity', () => {
  const snapshot = Object.fromEntries(body.feedback.map(field => [field.id, Array(field.size).fill(0)]));
  snapshot.projectedGravity = [0, -1, 0];
  assert.equal(encodeObservation(snapshot, body).length, 50);
  assert.throws(() => encodeObservation({ ...snapshot, footContacts: [0, 0.5] }, body), /contacts/);
  assert.throws(() => encodeObservation({ ...snapshot, projectedGravity: [0, 0, 0] }, body), /gravity/);
  delete snapshot.linearVelocity;
  assert.throws(() => encodeObservation(snapshot, body), /linearVelocity/);
});
