const fail = message => { throw new Error(message); };
const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${name} must be finite`);
};
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function validateBinding(binding) {
  if (!binding?.id || !binding.dataset || !Array.isArray(binding.neurons) || !binding.neurons.length) {
    fail('A versioned neuron binding is required');
  }
  const ids = binding.neurons.map(neuron => neuron.id);
  if (ids.some(id => typeof id !== 'string' || !/^\d+$/.test(id))) {
    fail('Neuron IDs must be decimal strings; JavaScript numbers lose root-ID precision');
  }
  if (new Set(ids).size !== ids.length) fail('Duplicate neuron IDs');
}

export function validateBodyMap(body) {
  if (!body?.id || body.units !== 'radians' || !body.joints?.length) fail('Invalid body map');
  const seen = new Set();
  const axes = new Set();
  for (const joint of body.joints) {
    if (!joint.id || seen.has(joint.id)) fail('Duplicate or empty joint ID');
    seen.add(joint.id);
    if (!joint.bone || !['x', 'y', 'z'].includes(joint.axis) || ![-1, 1].includes(joint.sign)) {
      fail(`Invalid rig mapping: ${joint.id}`);
    }
    const key = `${joint.bone}:${joint.axis}`;
    if (axes.has(key)) fail(`Duplicate bone axis: ${key}`);
    axes.add(key);
    for (const key of ['min', 'neutral', 'max', 'maxSpeed']) finite(joint[key], `${joint.id}.${key}`);
    if (!(joint.min <= joint.neutral && joint.neutral <= joint.max && joint.min < joint.max && joint.maxSpeed > 0)) {
      fail(`Invalid limits: ${joint.id}`);
    }
  }
  for (const key of ['rateTimeConstantSeconds', 'rateScaleHz', 'maxRateHz']) {
    finite(body.filter?.[key], key);
    if (body.filter[key] <= 0) fail(`${key} must be positive`);
  }
}

export function createZeroDecoder(binding, body) {
  validateBinding(binding);
  validateBodyMap(body);
  return {
    bindingId: binding.id, bodyMapId: body.id, provenance: 'untrained-zero',
    neuronIds: binding.neurons.map(neuron => neuron.id),
    jointIds: body.joints.map(joint => joint.id),
    weights: body.joints.map(() => binding.neurons.map(() => 0)),
    bias: body.joints.map(() => 0),
  };
}

export class BrainBodyBridge {
  constructor(binding, body, decoder = createZeroDecoder(binding, body)) {
    validateBinding(binding);
    validateBodyMap(body);
    this.binding = structuredClone(binding);
    this.body = structuredClone(body);
    this.ids = binding.neurons.map(neuron => neuron.id);
    this.idSet = new Set(this.ids);
    this.setDecoder(decoder);
    this.reset();
  }

  setDecoder(decoder) {
    if (decoder?.bindingId !== this.binding.id || decoder.bodyMapId !== this.body.id) {
      fail('Decoder binding/body version mismatch');
    }
    const same = (actual, expected) => Array.isArray(actual)
      && actual.length === expected.length && actual.every((item, i) => item === expected[i]);
    if (!same(decoder.neuronIds, this.ids)
      || !same(decoder.jointIds, this.body.joints.map(joint => joint.id))) {
      fail('Decoder neuron/joint order mismatch');
    }
    if (!Array.isArray(decoder.weights) || decoder.weights.length !== this.body.joints.length
      || !Array.isArray(decoder.bias) || decoder.bias.length !== this.body.joints.length) {
      fail('Decoder dimensions mismatch');
    }
    decoder.weights.forEach((row, j) => {
      if (!Array.isArray(row) || row.length !== this.ids.length) fail('Decoder row dimensions mismatch');
      row.forEach(value => finite(value, 'weight'));
      finite(decoder.bias[j], 'bias');
    });
    if (decoder.referenceAngles !== undefined) {
      if (!Array.isArray(decoder.referenceAngles) || decoder.referenceAngles.length !== this.body.joints.length) fail('Invalid reference pose');
      decoder.referenceAngles.forEach((angle, i) => {
        finite(angle, 'reference angle');
        if (angle < this.body.joints[i].min || angle > this.body.joints[i].max) fail('Reference angle outside limits');
      });
    }
    this.decoder = structuredClone(decoder);
  }

  reset(initialAngles = this.decoder.referenceAngles ?? this.body.joints.map(joint => joint.neutral)) {
    if (!Array.isArray(initialAngles) || initialAngles.length !== this.body.joints.length) fail('Invalid initial angles');
    initialAngles.forEach((angle, i) => {
      finite(angle, 'initial angle');
      const joint = this.body.joints[i];
      if (angle < joint.min || angle > joint.max) fail('Initial angle outside limits');
    });
    this.filteredHz = this.ids.map(() => 0);
    this.angles = [...initialAngles];
    this.elapsedSeconds = 0;
  }

  step(frame, dt) {
    finite(dt, 'dt');
    if (dt <= 0 || dt > 0.1) fail('dt must be in (0, 0.1] seconds');
    if (frame?.bindingId !== this.binding.id || frame.dataset !== this.binding.dataset) {
      fail('Input binding/dataset mismatch');
    }
    if (!frame.ratesHz || typeof frame.ratesHz !== 'object' || Array.isArray(frame.ratesHz)) {
      fail('ratesHz must map neuron ID strings to firing rates');
    }
    if (!['dense', 'sparse-zero'].includes(frame.encoding)) fail('Declare dense or sparse-zero encoding');
    for (const [id, value] of Object.entries(frame.ratesHz)) {
      if (!this.idSet.has(id)) fail(`Unknown output neuron: ${id}`);
      finite(value, `rate ${id}`);
      if (value < 0) fail('Firing rates must be nonnegative');
    }
    if (frame.encoding === 'dense' && this.ids.some(id => !Object.hasOwn(frame.ratesHz, id))) {
      fail('Missing firing rates in dense frame');
    }
    const { rateTimeConstantSeconds: tau, rateScaleHz: scale, maxRateHz } = this.body.filter;
    const alpha = -Math.expm1(-dt / tau);
    const filtered = this.filteredHz.map((old, i) => old
      + alpha * (Math.min(frame.ratesHz[this.ids[i]] ?? 0, maxRateHz) - old));
    const features = filtered.map(value => value / scale);
    const logits = this.decoder.weights.map((row, j) =>
      row.reduce((sum, weight, i) => sum + weight * features[i], this.decoder.bias[j]));
    logits.forEach(value => finite(value, 'decoder output'));
    const actions = logits.map(Math.tanh);
    const targets = actions.map((a, i) => {
      const joint = this.body.joints[i];
      const reference = this.decoder.referenceAngles?.[i] ?? joint.neutral;
      return reference + a * (a >= 0 ? joint.max - reference : reference - joint.min);
    });
    const angles = targets.map((target, i) => {
      const joint = this.body.joints[i];
      return clamp(
        this.angles[i] + clamp(target - this.angles[i], -joint.maxSpeed * dt, joint.maxSpeed * dt),
        joint.min, joint.max,
      );
    });
    const velocities = angles.map((value, i) => (value - this.angles[i]) / dt);
    this.filteredHz = filtered;
    this.angles = angles;
    this.elapsedSeconds += dt;
    return {
      elapsedSeconds: this.elapsedSeconds, provenance: this.decoder.provenance,
      filteredHz: [...filtered], actions, targetAngles: targets, angles: [...angles],
      commandVelocities: velocities,
      // These are commands; actual physical joint states must come from the simulator.
      jointAngles: Object.fromEntries(this.body.joints.map((joint, i) => [joint.id, angles[i]])),
    };
  }
}

export function encodeObservation(snapshot, body) {
  const output = [];
  for (const field of body.feedback) {
    const values = snapshot[field.id];
    if (!Array.isArray(values) || values.length !== field.size) fail(`Invalid feedback: ${field.id}`);
    values.forEach(value => finite(value, field.id));
    if (field.id === 'footContacts' && values.some(value => value !== 0 && value !== 1)) {
      fail('Foot contacts must be 0 or 1');
    }
    if (field.id === 'projectedGravity' && Math.abs(Math.hypot(...values) - 1) > 0.01) {
      fail('Projected gravity must be a unit vector');
    }
    output.push(...values);
  }
  return output;
}
