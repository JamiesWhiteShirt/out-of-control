import './style.css';

import defaultVertexShaderSource from './vertex.vs';
import defaultFragmentShaderSource from './fragment.fs';
import trailVertexShaderSource from './trail.vs';
import trailFragmentShaderSource from './trail.fs';
import sheep from './sheep.png';
import dirt from './dirt.png';
import smoke from './smoke.png';
import planet from './planet.png';

import { mat4, vec3, vec2, ReadonlyVec2 } from 'gl-matrix';

class SheepEntity {
  prevPosition: vec2;
  position: vec2;
  velocity: vec2;
  prevRotation: number;
  rotation: number;
  angularVelocity: number;
  panicUntil: number;
  boostIntensity: number;
}

class Particle {
  prevPosition: vec2;
  position: vec2;
  velocity: vec2;
  angularVelocity: number;
  removeOnTick: number;
}


const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = 720;

const WORLD_WIDTH = CANVAS_WIDTH * 4;
const WORLD_HEIGHT = CANVAS_HEIGHT * 4;

const canvas = document.createElement('canvas');
canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
document.body.appendChild(canvas);

const gl = canvas.getContext('webgl2');

const LOG_TRAIL_LENGTH = 6;
const TRAIL_LENGTH = 1 << LOG_TRAIL_LENGTH;
const MAX_TRAILS = 256;

function loadShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw Error('Error compiling shader: ' + infoLog);
  }

  return shader;
}

function loadProgram(gl: WebGL2RenderingContext, shaders: Array<WebGLShader>): WebGLProgram {
  const program = gl.createProgram();
  for (const shader of shaders) {
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw Error('Error linking shader program: ' + infoLog);
  }

  return program;
}

const defaultVertexShader = loadShader(gl, gl.VERTEX_SHADER, defaultVertexShaderSource);
const defaultFragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, defaultFragmentShaderSource);
const trailVertexShader = loadShader(gl, gl.VERTEX_SHADER, trailVertexShaderSource);
const trailFragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, trailFragmentShaderSource);

const defaultProgram = loadProgram(gl, [defaultVertexShader, defaultFragmentShader]);
const trailProgram = loadProgram(gl, [trailVertexShader, trailFragmentShader]);

const defaultProgramInfo = {
  program: defaultProgram,
  attribLocations: {
    position: gl.getAttribLocation(defaultProgram, 'aPosition'),
    textureCoord: gl.getAttribLocation(defaultProgram, 'aTextureCoord'),
  },
  uniformLocations: {
    projection: gl.getUniformLocation(defaultProgram, 'uProjection'),
    modelView: gl.getUniformLocation(defaultProgram, 'uModelView'),
    sampler: gl.getUniformLocation(defaultProgram, 'uSampler'),
    colorMultiplier: gl.getUniformLocation(defaultProgram, 'uColorMultiplier'),
  }
};

const trailProgramInfo = {
  program: trailProgram,
  attribLocations: {
    nodeIndex: gl.getAttribLocation(trailProgram, 'aNodeIndex'),
    scale: gl.getAttribLocation(trailProgram, 'aScale'),
    alpha: gl.getAttribLocation(trailProgram, 'aAlpha'),
  },
  uniformLocations: {
    projection: gl.getUniformLocation(trailProgram, 'uProjection'),
    modelView: gl.getUniformLocation(trailProgram, 'uModelView'),
    intensitySampler: gl.getUniformLocation(trailProgram, 'uIntensitySampler'),
    positionVelocitySampler: gl.getUniformLocation(trailProgram, 'uPositionVelocitySampler'),
    colorMultiplier: gl.getUniformLocation(trailProgram, 'uColorMultiplier'),
    firstNode: gl.getUniformLocation(trailProgram, 'uFirstNode'),
    tickDelta: gl.getUniformLocation(trailProgram, 'uTickDelta'),
  }
};

const squareVao = gl.createVertexArray();
gl.bindVertexArray(squareVao);

const squareBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, squareBuffer);
const squareData = new Float32Array([
  -0.5,  0.5, 0.0, 0.0,
   0.5,  0.5, 1.0, 0.0,
  -0.5, -0.5, 0.0, 1.0,
   0.5, -0.5, 1.0, 1.0,
]);
gl.bufferData(gl.ARRAY_BUFFER, squareData, gl.STATIC_DRAW);
gl.vertexAttribPointer(defaultProgramInfo.attribLocations.position, 2, gl.FLOAT, false, 4 * 4, 0);
gl.vertexAttribPointer(defaultProgramInfo.attribLocations.textureCoord, 2, gl.FLOAT, false, 4 * 4, 2 * 4);
gl.enableVertexAttribArray(defaultProgramInfo.attribLocations.position);
gl.enableVertexAttribArray(defaultProgramInfo.attribLocations.textureCoord);

const trailVao = gl.createVertexArray();
gl.bindVertexArray(trailVao);

const trailBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, trailBuffer);
{
  const STRIDE = 8;
  const NODE_INDEX_OFFSET = 0;
  const ALPHA_OFFSET = 1;
  const SCALE_OFFSET = 4;

  const trailData = new ArrayBuffer(TRAIL_LENGTH * STRIDE * 2);
  const trailDataView = new DataView(trailData);
  for (let i = 0; i < TRAIL_LENGTH; ++i) {
    const progress = i / TRAIL_LENGTH;
    const alpha = 64 * progress * progress;
    const scaleMagnitude = 32 * (1 - progress);

    trailDataView.setInt8(i * STRIDE * 2 + NODE_INDEX_OFFSET, i);
    trailDataView.setUint8(i * STRIDE * 2 + ALPHA_OFFSET, alpha);
    trailDataView.setFloat32(i * STRIDE * 2 + SCALE_OFFSET, -scaleMagnitude, true);
    trailDataView.setInt8(i * STRIDE * 2 + STRIDE + NODE_INDEX_OFFSET, i);
    trailDataView.setUint8(i * STRIDE * 2 + STRIDE + ALPHA_OFFSET, alpha);
    trailDataView.setFloat32(i * STRIDE * 2 + STRIDE + SCALE_OFFSET, scaleMagnitude, true);
  }

  gl.bufferData(gl.ARRAY_BUFFER, trailData, gl.STATIC_DRAW);
  gl.vertexAttribIPointer(trailProgramInfo.attribLocations.nodeIndex, 1, gl.BYTE, STRIDE, NODE_INDEX_OFFSET);
  gl.vertexAttribPointer(trailProgramInfo.attribLocations.alpha, 1, gl.UNSIGNED_BYTE, true, STRIDE, ALPHA_OFFSET);
  gl.vertexAttribPointer(trailProgramInfo.attribLocations.scale, 1, gl.FLOAT, false, STRIDE, SCALE_OFFSET);
}
gl.enableVertexAttribArray(trailProgramInfo.attribLocations.nodeIndex);
gl.enableVertexAttribArray(trailProgramInfo.attribLocations.alpha);
gl.enableVertexAttribArray(trailProgramInfo.attribLocations.scale);

const trailPositionVelocityTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, trailPositionVelocityTexture)
gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TRAIL_LENGTH, MAX_TRAILS);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

const trailIntensityTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, trailIntensityTexture)
gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, TRAIL_LENGTH, MAX_TRAILS);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>(resolve => {
    const image = new Image();
    image.onload = () => {
      resolve(image);
    }
    image.src = src;
  });
}

function modulo(o1: number, o2: number): number {
  return o1 >= 0 ? o1 % o2 : (o2 + o1 % o2);
}

function interpolateVec2(o: vec2, a: ReadonlyVec2, b: ReadonlyVec2, scalar: number): vec2 {
  vec2.scale(o, a, 1.0 - scalar);
  return vec2.scaleAndAdd(o, o, b, scalar);
}

const projectionMat = mat4.create();
mat4.ortho(projectionMat, -WORLD_WIDTH / 2, WORLD_WIDTH / 2, -WORLD_HEIGHT / 2, WORLD_HEIGHT / 2, -1, 1);
const invProjectionMat = mat4.create();
mat4.invert(invProjectionMat, projectionMat);

const allSheep: Array<SheepEntity> = [];
// let particles: Array<Particle> = [];

const allGrass: Array<vec2> = [
  vec2.fromValues(1024, 512),
  vec2.fromValues(1024, -512),
  vec2.fromValues(-1024, 512),
  vec2.fromValues(-1024, -512),
  vec2.fromValues(0, 1024),
  vec2.fromValues(0, -1024),
];

const allPlanets: Array<vec2> = [
 vec2.fromValues(0, 0),
 vec2.fromValues(2048, 0),
 vec2.fromValues(-2048, 0),
];

function findNearestGrass(pos: ReadonlyVec2): vec2 {
  let bestSquaredDistance = Number.POSITIVE_INFINITY;
  let bestGrass: vec2;
  for (let grass of allGrass) {
    const squaredDistance = vec2.squaredDistance(pos, grass);
    if (squaredDistance < bestSquaredDistance) {
      bestSquaredDistance = squaredDistance;
      bestGrass = grass;
    }
  }
  return bestGrass;
}

function findFurthestGrass(pos: ReadonlyVec2): vec2 {
  let bestSquaredDistance = Number.NEGATIVE_INFINITY;
  let bestGrass: vec2;
  for (let grass of allGrass) {
    const squaredDistance = vec2.squaredDistance(pos, grass);
    if (squaredDistance > bestSquaredDistance) {
      bestSquaredDistance = squaredDistance;
      bestGrass = grass;
    }
  }
  return bestGrass;
}

const SHEEP_RADIUS = 64;

canvas.onclick = e => {
  const position = vec2.fromValues(2.0 * e.x / CANVAS_WIDTH - 1.0, 2.0 * (CANVAS_HEIGHT - e.y) / CANVAS_HEIGHT - 1.0);
  vec2.transformMat4(position, position, invProjectionMat);
  const prevPosition = vec2.fromValues(position[0], position[1]);

  const sheep = new SheepEntity();
  sheep.prevPosition = prevPosition;
  sheep.position = position;
  sheep.velocity = vec2.fromValues(0, 0);
  sheep.angularVelocity = 0;
  // sheep.rotation = Math.atan2(GRASS_POS[1] - sheep.position[1], GRASS_POS[0] - sheep.position[0]);
  sheep.rotation = Math.random() * Math.PI * 2.0;
  sheep.prevRotation = sheep.rotation;
  sheep.panicUntil = 0;
  sheep.boostIntensity = 0;

  allSheep.push(sheep);
};

let panic = false;

function tick(currentTick: number) {
  for (const sheep of allSheep) {
    vec2.copy(sheep.prevPosition, sheep.position);
    sheep.prevRotation = sheep.rotation;
  }

  /* particles = particles.filter(particle => particle.removeOnTick > currentTick);

  for (const particle of particles) {
    vec2.copy(particle.prevPosition, particle.position);
    vec2.add(particle.position, particle.position, particle.velocity);
  } */

  for (let i = 0; i < allSheep.length; ++i) {
    const sheepA = allSheep[i]
    for (let j = i + 1; j < allSheep.length; ++j) {
      const sheepB = allSheep[j]

      const positionDelta = vec2.create();
      vec2.copy(positionDelta, sheepB.position);
      vec2.subtract(positionDelta, positionDelta, sheepA.position);

      if (vec2.squaredLength(positionDelta) < 4 * SHEEP_RADIUS * SHEEP_RADIUS) {
        vec2.normalize(positionDelta, positionDelta);
        const prevVelocityA = vec2.create();
        const prevVelocityB = vec2.create();
        const velocitySum = vec2.create();
        const velocityDelta = vec2.create();

        vec2.copy(prevVelocityA, sheepA.velocity);
        vec2.copy(prevVelocityB, sheepB.velocity);
        vec2.add(velocitySum, prevVelocityB, prevVelocityA);
        vec2.subtract(velocityDelta, prevVelocityB, prevVelocityA);

        const orthogonal = vec2.fromValues(-positionDelta[1], positionDelta[0]);

        const impactCoefficient = vec2.dot(velocityDelta, positionDelta);
        const otherCoefficient = vec2.dot(velocityDelta, orthogonal);

        const surfaceVelocityDelta = otherCoefficient * Math.PI / -SHEEP_RADIUS + sheepA.angularVelocity + sheepB.angularVelocity;
        const frictionAngularAcc = 0.1 * surfaceVelocityDelta;
        const absorption = 0.2;

        const acceleration = vec2.create();
        vec2.scale(acceleration, positionDelta, (1.0 - absorption) * impactCoefficient - 0.3);
        vec2.scaleAndAdd(acceleration, acceleration, orthogonal, -frictionAngularAcc * SHEEP_RADIUS / Math.PI);

        vec2.add(sheepA.velocity, sheepA.velocity, acceleration);
        vec2.subtract(sheepB.velocity, sheepB.velocity, acceleration);
        sheepA.angularVelocity -= frictionAngularAcc;
        sheepB.angularVelocity -= frictionAngularAcc;

        if (vec2.squaredLength(velocityDelta) > 32.0 * 32.0) {
          sheepA.panicUntil = currentTick + 300;
          sheepB.panicUntil = currentTick + 300;
        }
      }
    }
  }

  const trailIntensityData = new Uint8Array(allSheep.length * 4);
  const trailPositionVelocityData = new Float32Array(allSheep.length * 4);
  for (let i = 0; i < allSheep.length; ++i) {
    const sheep = allSheep[i];
    // sheep.angularVelocity += (Math.random() - Math.random()) * 0.001;
    // vec2.add(sheep.velocity, sheep.velocity, vec2.fromValues((Math.random() - Math.random()) * 0.05, (Math.random() - Math.random()) * 0.05));

    /* sheep.boostIntensity -= 32;
    if (sheep.boostIntensity < 0) {
      sheep.boostIntensity = 0;
    } */
    sheep.boostIntensity = 0;

    const panicking = panic || sheep.panicUntil > currentTick;

    const targetDelta = vec2.create();
    vec2.subtract(targetDelta, panicking ? findFurthestGrass(sheep.position) : findNearestGrass(sheep.position), sheep.position);
    vec2.scaleAndAdd(targetDelta, targetDelta, sheep.velocity, -100);

    const originAngle = Math.atan2(targetDelta[1], targetDelta[0]);
    const angularOffset = modulo(originAngle - sheep.rotation + Math.PI, Math.PI * 2) - Math.PI;
    const angularCorrection = -sheep.angularVelocity * 50 + angularOffset;

    if (panicking || vec2.squaredLength(targetDelta) > 256 * 256)
    {
      if (Math.abs(angularCorrection) > 0.25) {
        sheep.angularVelocity += Math.sign(angularCorrection) * 0.001;
      }

      const mOrientation = mat4.create();
      mat4.rotateZ(mOrientation, mOrientation, sheep.rotation);

      const estimatePos = targetDelta;
      const acc = vec2.fromValues(1.0, 0.0);
      vec2.transformMat4(acc, acc, mOrientation);

      const velocityCorrection = vec2.dot(estimatePos, acc) / vec2.length(estimatePos);

      if (panicking || Math.abs(angularOffset) < Math.PI / 8 && velocityCorrection > 0.8)
      {
        const boost = 0.25;
        const acceleration = vec2.fromValues(boost, 0);
        vec2.transformMat4(acceleration, acceleration, mOrientation);
        vec2.add(sheep.velocity, sheep.velocity, acceleration);

        /* const particle = new Particle();
        particle.position = vec2.create();
        particle.prevPosition = vec2.create();
        particle.velocity = vec2.create();
        particle.angularVelocity = (Math.random() - Math.random()) * 0.1;
        particle.removeOnTick = currentTick + 80 + Math.random() * 40;
        vec2.copy(particle.position, sheep.position);
        vec2.copy(particle.prevPosition, sheep.position);
        vec2.scale(particle.velocity, acceleration, -10.0);
        vec2.add(particle.velocity, particle.velocity, sheep.velocity);

        particles.push(particle); */

        sheep.boostIntensity = 255;
      }
    }

    const GRAVITATION = 10000;

    for (const planet of allPlanets) {
      const planetDelta = vec2.create();
      vec2.subtract(planetDelta, planet, sheep.position);
      const squaredDistance = vec2.squaredLength(planetDelta);
      vec2.scaleAndAdd(sheep.velocity, sheep.velocity, planetDelta, GRAVITATION / (squaredDistance * Math.sqrt(squaredDistance)));
    }

    vec2.add(sheep.position, sheep.position, sheep.velocity);
    sheep.rotation += sheep.angularVelocity;

    // write trail data
    trailIntensityData[i * 4] = sheep.boostIntensity;
    trailPositionVelocityData[i * 4 + 0] = sheep.position[0];
    trailPositionVelocityData[i * 4 + 1] = sheep.position[1];
    trailPositionVelocityData[i * 4 + 2] = sheep.velocity[0] - Math.cos(sheep.rotation) * 10;
    trailPositionVelocityData[i * 4 + 3] = sheep.velocity[1] - Math.sin(sheep.rotation) * 10;
  }

  if (allSheep.length > 0) {
    const nodeIndex = currentTick & (TRAIL_LENGTH - 1);
    gl.bindTexture(gl.TEXTURE_2D, trailIntensityTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, nodeIndex, 0, 1, allSheep.length, gl.RED, gl.UNSIGNED_BYTE, trailIntensityData);
    gl.bindTexture(gl.TEXTURE_2D, trailPositionVelocityTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, nodeIndex, 0, 1, allSheep.length, gl.RGBA, gl.FLOAT, trailPositionVelocityData);
  }
}

function createTexture(gl: WebGL2RenderingContext, image: HTMLImageElement): WebGLTexture {
  const texture = gl.createTexture();

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, image.width, image.height);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, image);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

  return texture;
}

gl.useProgram(defaultProgram);
gl.uniform1i(defaultProgramInfo.uniformLocations.sampler, 0);
gl.uniformMatrix4fv(defaultProgramInfo.uniformLocations.projection, false, projectionMat);

gl.useProgram(trailProgram);
gl.uniform1i(trailProgramInfo.uniformLocations.intensitySampler, 0);
gl.uniform1i(trailProgramInfo.uniformLocations.positionVelocitySampler, 1);
gl.uniformMatrix4fv(trailProgramInfo.uniformLocations.projection, false, projectionMat);
gl.uniform4f(trailProgramInfo.uniformLocations.colorMultiplier, 1.0, 1.0, 1.0, 1.0);

Promise.all([
  loadImage(sheep),
  loadImage(dirt),
  loadImage(smoke),
  loadImage(planet),
]).then(([
  sheepImage,
  dirtImage,
  smokeImage,
  planetImage,
]) => {
  const sheepTexture = createTexture(gl, sheepImage);
  const dirtTexture = createTexture(gl, dirtImage);
  const smokeTexture = createTexture(gl, smokeImage);
  const planetTexture = createTexture(gl, planetImage);

  gl.clearColor(30 / 255, 42 / 255, 69 / 255, 1.0);

  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.BLEND);

  const TICKS_PER_SECOND = 60;
  let lastTick = 0;

  document.onkeyup = e => {
    if (e.keyCode == 32) {
      allSheep[0].panicUntil = lastTick + 300;
      // panic = !panic;
    }
  };

  function render(time: number) {
    const realTickTime = 0.001 * time * TICKS_PER_SECOND;
    const nextTick = Math.floor(realTickTime);
    while (nextTick > lastTick) {
      tick(lastTick);
      ++lastTick;
    }
    const tickDelta = lastTick - realTickTime;

    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(defaultProgram);
    gl.bindVertexArray(squareVao);
    gl.uniform4f(defaultProgramInfo.uniformLocations.colorMultiplier, 1.0, 1.0, 1.0, 1.0);

    gl.bindTexture(gl.TEXTURE_2D, dirtTexture);
    for (const grass of allGrass) {
      const modelView = mat4.create();
      mat4.translate(modelView, modelView, vec3.fromValues(grass[0], grass[1], 0.0));
      mat4.scale(modelView, modelView, vec3.fromValues(256, 256, 1));
      gl.uniformMatrix4fv(defaultProgramInfo.uniformLocations.modelView, false, modelView);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    gl.bindTexture(gl.TEXTURE_2D, planetTexture);
    for (const planet of allPlanets) {
      const modelView = mat4.create();
      mat4.translate(modelView, modelView, vec3.fromValues(planet[0], planet[1], 0.0));
      mat4.scale(modelView, modelView, vec3.fromValues(256, 256, 1));
      gl.uniformMatrix4fv(defaultProgramInfo.uniformLocations.modelView, false, modelView);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    gl.bindTexture(gl.TEXTURE_2D, sheepTexture);
    for (const sheep of allSheep) {
      const rotation = sheep.rotation * (1 - tickDelta) + sheep.rotation * tickDelta;
      const position = vec2.create();
      interpolateVec2(position, sheep.prevPosition, sheep.position, tickDelta);

      const modelView = mat4.create();
      mat4.translate(modelView, modelView, vec3.fromValues(position[0], position[1], 0.0));
      mat4.rotateZ(modelView, modelView, rotation);
      mat4.scale(modelView, modelView, vec3.fromValues(256, 256, 1));
      gl.uniformMatrix4fv(defaultProgramInfo.uniformLocations.modelView, false, modelView);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    /* gl.bindTexture(gl.TEXTURE_2D, smokeTexture);
    for (const particle of particles) {
      const rotation = particle.angularVelocity * realTickTime;
      const position = vec2.create();
      const scale = 128 - 120 * (particle.removeOnTick - realTickTime) / 120;
      interpolateVec2(position, particle.prevPosition, particle.position, tickDelta);

      gl.uniform4f(defaultProgramInfo.uniformLocations.colorMultiplier, 1.0, 1.0, 1.0, 0.1 * (128 - scale) / 128);
      const modelView = mat4.create();
      mat4.translate(modelView, modelView, vec3.fromValues(position[0], position[1], 0.0));
      mat4.rotateZ(modelView, modelView, rotation);
      mat4.scale(modelView, modelView, vec3.fromValues(scale, scale, 1));
      gl.uniformMatrix4fv(defaultProgramInfo.uniformLocations.modelView, false, modelView);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    } */

    gl.bindVertexArray(trailVao);
    gl.useProgram(trailProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, trailPositionVelocityTexture);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, trailIntensityTexture);
    {
      const modelView = mat4.create();
      gl.uniformMatrix4fv(trailProgramInfo.uniformLocations.modelView, false, modelView);
      gl.uniform1i(trailProgramInfo.uniformLocations.firstNode, lastTick & (TRAIL_LENGTH - 1));
      gl.uniform1f(trailProgramInfo.uniformLocations.tickDelta, tickDelta);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, TRAIL_LENGTH * 2, allSheep.length);
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
});
