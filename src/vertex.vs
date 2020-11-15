attribute vec4 aPosition;
attribute vec2 aTextureCoord;

uniform mat4 uProjection;
uniform mat4 uModelView;

varying highp vec2 vTextureCoord;

void main() {
  gl_Position = uProjection * uModelView * aPosition;
  vTextureCoord = aTextureCoord;
}
