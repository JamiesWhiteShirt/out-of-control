#version 300 es
const int LOG_TRAIL_LENGTH = 6;
const int TRAIL_LENGTH = 1 << LOG_TRAIL_LENGTH;
const int MAX_TRAILS = 256;

in int aNodeIndex;
in float aScale;
in float aAlpha;

uniform mat4 uProjection;
uniform mat4 uModelView;
uniform sampler2D uIntensitySampler;
uniform sampler2D uPositionVelocitySampler;
uniform int uFirstNode;
uniform float uTickDelta;

out highp float vAlpha;

float getNodeIntensity(int nodeIndex, int trailIndex) {
  // vec2 uv = vec2(float(nodeIndex + uFirstNode) / float(TRAIL_LENGTH), float(trailIndex) / float(MAX_TRAILS));
  // return texture(uIntensitySampler, uv).r;
  ivec2 uv = ivec2((nodeIndex + uFirstNode) & (TRAIL_LENGTH - 1), trailIndex);
  return texelFetch(uIntensitySampler, uv, 0).r;
  // return texelFetch(uIntensitySampler, ivec2((nodeIndex + uFirstNode) & (TRAIL_LENGTH - 1), trailIndex), 0).r;
}

vec2 getNodePosition(int nodeIndex, int trailIndex) {
  // vec2 uv = vec2(float(nodeIndex + uFirstNode) / float(TRAIL_LENGTH), float(trailIndex) / float(MAX_TRAILS));
  // vec4 positionVelocity = texture(uPositionVelocitySampler, uv);
  ivec2 uv = ivec2((nodeIndex + uFirstNode) & (TRAIL_LENGTH - 1), trailIndex);
  vec4 positionVelocity = texelFetch(uPositionVelocitySampler, uv, 0);
  return positionVelocity.xy + (float(TRAIL_LENGTH - nodeIndex) + uTickDelta) * positionVelocity.zw;
}

void main() {
  vec2 prevNode = getNodePosition(aNodeIndex - 1, gl_InstanceID);
  vec2 thisNode = getNodePosition(aNodeIndex, gl_InstanceID);
  vec2 nextNode = getNodePosition(aNodeIndex + 1, gl_InstanceID);

  float intensity = getNodeIntensity(aNodeIndex, gl_InstanceID);
  vec2 surroundingDelta = nextNode - prevNode;
  vec2 forwardNormal = normalize(surroundingDelta);
  vec2 sideNormal = vec2(-forwardNormal.y, forwardNormal.x);

  gl_Position = uProjection * uModelView * vec4(thisNode + sideNormal * aScale, 0.0, 1.0);
  vAlpha = aAlpha * intensity;
}
