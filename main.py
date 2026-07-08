#!/usr/bin/env python3
"""
PaintVolley Starter Bot
========================

Edit this file to implement your bot. You must define ONE function:

    def get_action(obs: dict) -> str

Return one of:
    "LEFT" "RIGHT" "JUMP" "JUMP_LEFT" "JUMP_RIGHT" "NONE"

`obs` is the full, deterministic game state each tick -- including every physics
constant in obs["rules"] -- so you can forward-simulate the world yourself to
predict where balls will go and plan your bumps. See README.md for the schema.

This starter bot:
  1. Targets the ball that will reach its helmet height soonest, with a bonus for
     balls that aren't its color (bumping one steals the tile stream from an opponent).
  2. Forward-simulates that ball's flight (wall bounces, no gravity) to find where
     it will next cross helmet height, and walks there.
  3. Jumps to meet a descending ball a little higher when it's close.
  4. Aims the bump toward the side of the field it owns less of.

Beat it by intercepting more reliably, holding onto your own ball while stealing the
opponent's, and aiming to overwrite their territory. Note that characters are solid --
you cannot pass through the other player.
"""

from __future__ import annotations


def get_action(obs: dict) -> str:
    me = obs["you"]["id"]
    rules = obs["rules"]
    W = obs["field"]["width"]
    R = rules["ball_radius"]
    hw = rules["player_half_width"]
    my = next(p for p in obs["players"] if p["id"] == me)
    helmet_y = my["y"]

    balls = obs["balls"]
    if not balls:
        return "NONE"

    # Target a ball to bump. A stationary ball (the resting ball at the start, or one
    # you've cornered) is bumped by standing under it and jumping, so its target is just
    # its own x. A moving ball is intercepted where it will next descend to helmet
    # height. Prefer the ball whose landing is NEAREST -- so we secure and keep re-hitting
    # the one we can actually reach (usually our own) instead of abandoning it to chase a
    # far one. A small discount pulls us toward stealing a reachable enemy/neutral ball.
    pspeed = max(rules["player_speed"], 1e-6)
    best = None
    for b in balls:
        if abs(b["vx"]) + abs(b["vy"]) < 0.01:
            cx = b["x"]
            tk = abs(cx - my["x"]) / pspeed + 5
        else:
            hit = _predict_cross(b, helmet_y, W, R)
            if hit is None:
                continue
            cx, tk = hit
        dist = abs(cx - my["x"])
        reachable = dist <= pspeed * tk + hw
        priority = dist + 0.05 * tk - (2.0 if (b["color"] != me and reachable) else 0.0)
        if best is None or priority < best[0]:
            best = (priority, cx, tk, b)
    if best is None:
        return _toward(my["x"], W / 2)
    _, cross_x, ticks, ball = best

    # Aim: stand a little to one side of where the ball will land so we strike it
    # off-center and deflect it toward whichever half of the field we own less of.
    # Commit to this offset all the way through the catch -- if we re-centered as the
    # ball approached we'd hit it dead-center every time and it would just bounce
    # straight up and down in one column, painting nothing new.
    left_owned, right_owned = _side_ownership(obs, me)
    aim = 1.0 if right_owned < left_owned else -1.0   # +1 => deflect the ball rightward
    target_x = max(hw, min(W - hw, cross_x - aim * hw * 0.6))

    move = _toward(my["x"], target_x)

    # Jump up into the ball when we're basically underneath it and it's within reach
    # above us and not already flying upward (so this handles both a resting ball we
    # want to start and a ball descending toward us). can_jump is False while another
    # player is standing on our head.
    if (
        my.get("can_jump", my["on_ground"])
        and abs(ball["x"] - my["x"]) < hw + 1.0
        and ball["y"] < helmet_y
        and ball["vy"] >= -0.01
        and (helmet_y - ball["y"]) < 6
    ):
        return {"LEFT": "JUMP_LEFT", "RIGHT": "JUMP_RIGHT", "NONE": "JUMP"}[move]
    return move


def _toward(x: float, target: float) -> str:
    if target < x - 0.25:
        return "LEFT"
    if target > x + 0.25:
        return "RIGHT"
    return "NONE"


def _predict_cross(ball: dict, helmet_y: float, W: float, R: float):
    """Simulate a ball forward (wall bounces, no gravity) and return (x, ticks) of the
    next time it descends across helmet height. None if it doesn't within the horizon."""
    x, y = ball["x"], ball["y"]
    vx, vy = ball["vx"], ball["vy"]
    if vx == 0 and vy == 0:
        return None
    for t in range(1, 600):
        py = y
        x += vx
        y += vy
        if x < R:
            x = R
            vx = abs(vx)
        elif x > W - R:
            x = W - R
            vx = -abs(vx)
        if y < R:
            y = R
            vy = abs(vy)
        if py < helmet_y <= y and vy > 0:
            return x, t
    return None


def _side_ownership(obs: dict, me: int):
    """Count tiles we own on the left vs right half of the field."""
    tiles = obs["tiles"]
    cols = obs["field"]["cols"]
    mid = cols // 2
    left = right = 0
    for row in tiles:
        for c in range(cols):
            if row[c] == me:
                if c < mid:
                    left += 1
                else:
                    right += 1
    return left, right
