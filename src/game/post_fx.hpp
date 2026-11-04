#pragma once

// The street's light: everything drawn between BeginScene and EndScene goes
// into an offscreen texture, which is then drawn to the screen with bloom
// (bright neon bleeds into its surroundings) and wet-street reflections
// (the storefronts mirrored at each curb, rippling). What's drawn after
// EndScene (NPCs, speech, the HUD) stays crisp.

#include "raylib.h"

namespace murmur::game {

class PostFx {
public:
    PostFx(int width, int height, float street_top, float street_bottom);
    ~PostFx();

    PostFx(const PostFx&) = delete;
    PostFx& operator=(const PostFx&) = delete;

    void BeginScene() const;
    void EndScene(float time);

private:
    RenderTexture2D scene_{};
    Shader shader_{};
    int time_location_ = -1;
    float street_top_;
    float street_bottom_;
};

}  // namespace murmur::game
