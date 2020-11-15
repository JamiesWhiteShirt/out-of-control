varying highp vec2 vTextureCoord;

uniform sampler2D uSampler;
uniform highp vec4 uColorMultiplier;

void main() {
  gl_FragColor = texture2D(uSampler, vTextureCoord) * uColorMultiplier;
}
