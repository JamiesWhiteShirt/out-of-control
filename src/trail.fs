#version 300 es
in highp float vAlpha;

uniform highp vec4 uColorMultiplier;

out highp vec4 diffuseColor;

void main() {
  diffuseColor = vec4(1.0, 1.0, 1.0, vAlpha) * uColorMultiplier;
}
