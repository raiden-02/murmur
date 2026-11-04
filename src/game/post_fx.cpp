#include "game/post_fx.hpp"

namespace murmur::game {

namespace {

// Positions come from gl_FragCoord (bottom-up) and the scene is sampled at
// explicit coordinates, so the render texture's upside-down storage needs
// no special care. Blur comes from the scene's mipmaps: each level averages
// the one below over twice the distance, so a few lookups make a wide glow.
constexpr const char* kShader = R"(#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec2 resolution;
uniform vec2 street;   // top and bottom of the wet street, in pixels from the top
uniform float time;
out vec4 finalColor;

vec3 scene(vec2 p, float lod) {
    return textureLod(texture0, vec2(p.x / resolution.x, 1.0 - p.y / resolution.y), lod).rgb;
}

void main() {
    vec2 p = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    vec3 color = scene(p, 0.0);

    // Wet street: the storefronts mirrored at each curb, rippling with the
    // rain and fading with the distance from the curb.
    if (p.y > street.x && p.y < street.y) {
        float ripple = sin(p.y * 0.21 + time * 2.7) * 2.5 + sin(p.x * 0.033 + p.y * 0.05 + time * 1.3) * 1.5;
        float top = p.y - street.x;
        float bottom = street.y - p.y;
        vec3 reflection = scene(vec2(p.x + ripple, street.x - top), 1.5) * exp(-top / 70.0)
                        + scene(vec2(p.x + ripple, street.y + bottom), 1.5) * exp(-bottom / 70.0);
        color += reflection * 0.45;
    }

    // Bloom: what stays bright after blurring glows.
    vec3 glow = max(scene(p, 2.0) - 0.30, 0.0) * 0.6
              + max(scene(p, 3.5) - 0.20, 0.0) * 0.8
              + max(scene(p, 5.0) - 0.10, 0.0) * 1.0;
    finalColor = vec4(color + glow, 1.0);
}
)";

}  // namespace

PostFx::PostFx(int width, int height, float street_top, float street_bottom)
    : scene_(LoadRenderTexture(width, height)),
      shader_(LoadShaderFromMemory(nullptr, kShader)),
      street_top_(street_top),
      street_bottom_(street_bottom) {
    GenTextureMipmaps(&scene_.texture);
    SetTextureFilter(scene_.texture, TEXTURE_FILTER_TRILINEAR);
    time_location_ = GetShaderLocation(shader_, "time");
    const float resolution[2] = { static_cast<float>(width), static_cast<float>(height) };
    SetShaderValue(shader_, GetShaderLocation(shader_, "resolution"), resolution, SHADER_UNIFORM_VEC2);
    const float street[2] = { street_top_, street_bottom_ };
    SetShaderValue(shader_, GetShaderLocation(shader_, "street"), street, SHADER_UNIFORM_VEC2);
}

PostFx::~PostFx() {
    UnloadShader(shader_);
    UnloadRenderTexture(scene_);
}

void PostFx::BeginScene() const {
    BeginTextureMode(scene_);
}

// Draws the scene to the screen through the shader.
void PostFx::EndScene(float time) {
    EndTextureMode();
    GenTextureMipmaps(&scene_.texture);
    SetShaderValue(shader_, time_location_, &time, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(shader_);
    const Rectangle source{ 0.0f, 0.0f, static_cast<float>(scene_.texture.width), -static_cast<float>(scene_.texture.height) };
    DrawTextureRec(scene_.texture, source, { 0.0f, 0.0f }, WHITE);
    EndShaderMode();
}

}  // namespace murmur::game
