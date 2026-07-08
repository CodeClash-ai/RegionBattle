#!/usr/bin/env python3
"""
RegionBattle Starter Bot
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

    # Target the ball reaching my helmet height soonest; a ball that isn't my color
    # gets a bonus (bumping it flips it to me and cuts off the opponent's paint).
    best = None
    for b in balls:
        hit = _predict_cross(b, helmet_y, W, R)
        if hit is None:
            continue
        cx, tk = hit
        priority = tk - (8 if b["color"] != me else 0)
        if best is None or priority < best[0]:
            best = (priority, cx, tk, b)
    if best is None:
        return _toward(my["x"], W / 2)
    _, cross_x, ticks, ball = best

    # Aim: if we have time before it arrives, stand a little to one side so the bump
    # deflects the ball toward whichever half of the field we own less of.
    target_x = cross_x
    reach = abs(cross_x - my["x"]) / max(rules["player_speed"], 1e-6)
    if ticks > reach + 12:
        left_owned, right_owned = _side_ownership(obs, me)
        aim = 1.0 if right_owned < left_owned else -1.0
        target_x = cross_x - aim * hw * 0.6
    target_x = max(hw, min(W - hw, target_x))

    move = _toward(my["x"], target_x)

    # Jump to meet a descending ball a bit higher when we're basically underneath it.
    if (
        my["on_ground"]
        and abs(ball["x"] - my["x"]) < hw + 1.5
        and ball["y"] < helmet_y
        and ball["vy"] > 0
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
