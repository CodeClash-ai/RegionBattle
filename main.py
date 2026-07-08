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
  1. Picks the most valuable ball to intercept (prefers balls that aren't
     already its color, and balls that are heading toward the floor).
  2. Forward-simulates that ball's flight (wall bounces, no gravity) to find
     where it will next cross helmet height.
  3. Walks to that spot; jumps if the ball can be met higher up.
  4. Aims the bump toward the side of the field it owns less of, so the ball
     paints more territory for it.
"""

from __future__ import annotations


def get_action(obs: dict) -> str:
    me = obs["you"]["id"]
    rules = obs["rules"]
    field = obs["field"]
    W = field["width"]
    R = rules["ball_radius"]
    speed = rules["ball_speed"]
    my = next(p for p in obs["players"] if p["id"] == me)
    helmet_y = my["y"]  # world-y of my helmet surface right now

    balls = obs["balls"]
    if not balls:
        return "NONE"

    # --- pick a target ball -----------------------------------------------------------
    # Go for the ball that will reach my helmet height soonest so I reliably catch
    # something every time one comes down. Balls that aren't my color get a small
    # bonus (bumping them steals the tile stream from an opponent).
    best = None
    for b in balls:
        hit = _predict_cross(b, helmet_y, W, R, speed)
        if hit is None:
            continue
        cross_x, ticks = hit
        priority = ticks - (8 if b["color"] != me else 0)
        if best is None or priority < best[0]:
            best = (priority, cross_x, ticks, b)

    if best is None:
        # No ball is coming down soon; drift toward the middle to stay flexible.
        return _toward(my["x"], W / 2)

    _, cross_x, ticks, ball = best
    hw = rules["player_half_width"]

    # --- aim: only when we have time to spare, nudge our stand point so the bump
    # deflects toward the side we own less of. When the ball is arriving imminently,
    # forget aiming and just get squarely underneath it so we don't whiff the catch.
    target_x = cross_x
    reach_ticks = abs(cross_x - my["x"]) / max(rules["player_speed"], 1e-6)
    if ticks > reach_ticks + 12:
        left_owned, right_owned = _side_ownership(obs, me)
        aim = 1.0 if right_owned < left_owned else -1.0
        # ball right of center -> positive offset -> veers right, so to aim right
        # we stand a little left of the ball's landing point.
        target_x = cross_x - aim * hw * 0.6
    target_x = max(hw, min(W - hw, target_x))

    # --- decide movement + whether to jump --------------------------------------------
    move = _toward(my["x"], target_x)

    # Jump if the ball is descending and close enough that meeting it higher helps,
    # and we're roughly underneath it.
    ball_close_x = abs(ball["x"] - my["x"]) < rules["player_half_width"] + 1.5
    ball_above = ball["y"] < helmet_y and ball["vy"] > 0
    if my["on_ground"] and ball_close_x and ball_above and (helmet_y - ball["y"]) < 6:
        if move == "LEFT":
            return "JUMP_LEFT"
        if move == "RIGHT":
            return "JUMP_RIGHT"
        return "JUMP"
    return move


def _toward(x: float, target: float) -> str:
    if target < x - 0.25:
        return "LEFT"
    if target > x + 0.25:
        return "RIGHT"
    return "NONE"


def _predict_cross(ball: dict, helmet_y: float, W: float, R: float, speed: float):
    """Simulate a ball forward (wall bounces, no gravity) and return (x, ticks) of the
    next time it descends across helmet height. None if it doesn't within the horizon."""
    x, y = ball["x"], ball["y"]
    vx, vy = ball["vx"], ball["vy"]
    if vx == 0 and vy == 0:
        return None
    for t in range(1, 400):
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
        # (bottom wall handled implicitly; we only care about the downward crossing)
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
