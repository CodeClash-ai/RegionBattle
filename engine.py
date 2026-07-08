#!/usr/bin/env python3
"""
RegionBattle Game Engine
========================

A deterministic, tick-based territory-painting arena for CodeClash.

Each player controls a helmet-wearing character that patrols the bottom of a
rectangular field. Balls (one per player) fly around the field, bouncing off
all four walls with no gravity. Every tick, each ball paints the tile it
currently overlaps in its own color. When a ball lands on a player's helmet it
is *recolored* to that player's color and reflected upward -- the exit angle
depends on where on the helmet it struck (center = straight up, edges = steep).

Whoever owns the most tiles when the tick budget runs out wins the game.

The simulation is fully deterministic given (seed, bot code) and every physics
constant is exposed to bots, so a strong bot can forward-simulate the world and
plan interceptions / aim its bumps. Strategy lives in the code, not in reflexes.

Usage:
    python engine.py /path/to/p1/main.py /path/to/p2/main.py -r NUM_GAMES -o OUTPUT_DIR

The bot interface (see README.md and main.py):

    def get_action(obs: dict) -> str:
        # return one of: "LEFT" "RIGHT" "JUMP" "JUMP_LEFT" "JUMP_RIGHT" "NONE"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import signal
import sys
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------------------
# Game constants  (all distances/speeds are in *tile units*; 1 tile == 1.0)
# --------------------------------------------------------------------------------------

COLS = 32                    # tile grid width
ROWS = 24                    # tile grid height
WIDTH = float(COLS)          # field width  (world units)
HEIGHT = float(ROWS)         # field height (world units)

BALL_RADIUS = 0.6
BALL_SPEED = 0.45            # constant ball speed magnitude (tiles / tick)

PLAYER_HALF_WIDTH = 2.0      # helmet spans [x - hw, x + hw]
PLAYER_HEIGHT = 2.5          # from feet (bottom wall) up to helmet surface
PLAYER_SPEED = 0.5           # horizontal move per tick (tiles / tick)
JUMP_SPEED = 0.62            # initial upward velocity of a jump
GRAVITY = 0.032              # downward accel applied to a jumping player per tick
MAX_BOUNCE_ANGLE = math.radians(60)  # helmet-edge hit deflects up to this from vertical

MAX_TICKS = 1500             # tick budget per game
TURN_TIMEOUT = 0.10          # seconds a single get_action call may take before -> NONE
REPLAY_EVERY = 5             # record a replay frame every N ticks

NEUTRAL = -1                 # unpainted tile / unclaimed ball color

# Base36 alphabet for compactly encoding a tile's owner id in replay frames
# ('.' == neutral). Supports up to 36 players, far more than any real match.
_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

VALID_ACTIONS = {"LEFT", "RIGHT", "JUMP", "JUMP_LEFT", "JUMP_RIGHT", "NONE"}


# --------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    color: int  # owning player id, or NEUTRAL


@dataclass
class PlayerState:
    """Physical state of a player character (distinct from its bot function)."""

    pid: int
    x: float                 # helmet center x
    y_off: float = 0.0       # jump height above resting position (>= 0)
    vy: float = 0.0          # vertical velocity while airborne (up = negative)
    on_ground: bool = True

    @property
    def helmet_y(self) -> float:
        """World-y of the helmet surface (top of the character). Smaller y = higher."""
        return HEIGHT - PLAYER_HEIGHT - self.y_off


# --------------------------------------------------------------------------------------
# Bot loading + sandboxed invocation
# --------------------------------------------------------------------------------------


class TimeoutError_(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ARG001
    raise TimeoutError_()


_HAS_ALARM = hasattr(signal, "SIGALRM")
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _on_alarm)

_module_counter = 0


def load_bot(path: str) -> Callable:
    """Import a bot module and return its get_action function."""
    global _module_counter
    _module_counter += 1
    module_name = f"bot_module_{_module_counter}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_action"):
        raise ValueError(f"Bot module {path} must define a get_action(obs) function")
    return module.get_action


def call_bot(fn: Callable, obs: dict) -> str:
    """Call a bot's get_action with a crash/timeout guard. Any failure -> 'NONE'."""
    if _HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, TURN_TIMEOUT)
    try:
        action = fn(obs)
    except Exception:
        return "NONE"
    finally:
        if _HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)

    if isinstance(action, str):
        action = action.strip().upper()
        if action in VALID_ACTIONS:
            return action
    return "NONE"


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------


class Game:
    def __init__(self, num_players: int, seed: int):
        self.n = num_players
        self.rng = random.Random(seed)
        self.tick = 0
        # grid[row][col] = owning player id or NEUTRAL
        self.grid: list[list[int]] = [[NEUTRAL] * COLS for _ in range(ROWS)]

        # Players evenly spaced across the bottom.
        self.players: list[PlayerState] = []
        for i in range(num_players):
            frac = (i + 0.5) / num_players
            x = frac * (WIDTH - 2 * PLAYER_HALF_WIDTH) + PLAYER_HALF_WIDTH
            self.players.append(PlayerState(pid=i, x=x))

        # One ball per player, starting above that player's head. Balls start
        # NEUTRAL (uncolored) and paint nothing until a player bumps one to claim
        # it -- so a passive bot paints nothing, and you must keep bumping to score.
        # Launched upward in a deterministic-but-varied direction.
        self.balls: list[Ball] = []
        for i, p in enumerate(self.players):
            angle = self.rng.uniform(-MAX_BOUNCE_ANGLE, MAX_BOUNCE_ANGLE)
            self.balls.append(
                Ball(
                    x=p.x,
                    y=p.helmet_y - 2.0,
                    vx=BALL_SPEED * math.sin(angle),
                    vy=-BALL_SPEED * math.cos(angle),
                    color=NEUTRAL,
                )
            )

    # -- observation -------------------------------------------------------------------

    def _rules(self) -> dict:
        return {
            "ball_radius": BALL_RADIUS,
            "ball_speed": BALL_SPEED,
            "player_half_width": PLAYER_HALF_WIDTH,
            "player_height": PLAYER_HEIGHT,
            "player_speed": PLAYER_SPEED,
            "jump_speed": JUMP_SPEED,
            "gravity": GRAVITY,
            "max_bounce_angle": MAX_BOUNCE_ANGLE,
        }

    def observation(self, me: int) -> dict:
        return {
            "tick": self.tick,
            "max_ticks": MAX_TICKS,
            "field": {"width": WIDTH, "height": HEIGHT, "cols": COLS, "rows": ROWS},
            "rules": self._rules(),
            "you": {"id": me, "color": me},
            "players": [
                {
                    "id": p.pid,
                    "color": p.pid,
                    "x": p.x,
                    "y": p.helmet_y,
                    "on_ground": p.on_ground,
                }
                for p in self.players
            ],
            "balls": [
                {"x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy, "color": b.color}
                for b in self.balls
            ],
            "tiles": [row[:] for row in self.grid],
            "scores": self.scores(),
        }

    # -- stepping ----------------------------------------------------------------------

    def apply_action(self, p: PlayerState, action: str) -> None:
        if action in ("LEFT", "JUMP_LEFT"):
            p.x -= PLAYER_SPEED
        elif action in ("RIGHT", "JUMP_RIGHT"):
            p.x += PLAYER_SPEED
        p.x = max(PLAYER_HALF_WIDTH, min(WIDTH - PLAYER_HALF_WIDTH, p.x))

        if action in ("JUMP", "JUMP_LEFT", "JUMP_RIGHT") and p.on_ground:
            p.vy = -JUMP_SPEED
            p.on_ground = False

    def step_players(self) -> None:
        for p in self.players:
            if not p.on_ground:
                # y_off is height above rest; vy negative = moving up (y_off increasing)
                p.y_off -= p.vy
                p.vy += GRAVITY
                if p.y_off <= 0:
                    p.y_off = 0.0
                    p.vy = 0.0
                    p.on_ground = True

    def _paint(self, x: float, y: float, color: int) -> None:
        if color == NEUTRAL:
            return
        col = int(x)
        row = int(y)
        if 0 <= col < COLS and 0 <= row < ROWS:
            self.grid[row][col] = color

    def step_balls(self) -> None:
        for b in self.balls:
            x0, y0 = b.x, b.y
            nx, ny = b.x + b.vx, b.y + b.vy

            # Paint along the movement segment so fast balls leave no gaps.
            dist = math.hypot(nx - x0, ny - y0)
            samples = max(1, int(dist / 0.34) + 1)
            for s in range(1, samples + 1):
                t = s / samples
                self._paint(x0 + (nx - x0) * t, y0 + (ny - y0) * t, b.color)

            b.x, b.y = nx, ny

            # Wall reflections (all four walls -- a ball is never lost).
            if b.x < BALL_RADIUS:
                b.x = BALL_RADIUS
                b.vx = abs(b.vx)
            elif b.x > WIDTH - BALL_RADIUS:
                b.x = WIDTH - BALL_RADIUS
                b.vx = -abs(b.vx)
            if b.y < BALL_RADIUS:
                b.y = BALL_RADIUS
                b.vy = abs(b.vy)
            elif b.y > HEIGHT - BALL_RADIUS:
                b.y = HEIGHT - BALL_RADIUS
                b.vy = -abs(b.vy)

            self._resolve_helmet_bump(b, y0)

    def _resolve_helmet_bump(self, b: Ball, y_prev: float) -> None:
        """Recolor + reflect a ball that lands on a helmet (Breakout-paddle style)."""
        for p in self.players:
            top = p.helmet_y
            near_x = (p.x - PLAYER_HALF_WIDTH - BALL_RADIUS) <= b.x <= (
                p.x + PLAYER_HALF_WIDTH + BALL_RADIUS
            )
            crossed_top = (y_prev + BALL_RADIUS) <= top and (b.y + BALL_RADIUS) >= top
            if near_x and crossed_top and b.vy > 0:
                # offset in [-1, 1]: where on the helmet it struck
                offset = (b.x - p.x) / PLAYER_HALF_WIDTH
                offset = max(-1.0, min(1.0, offset))
                theta = offset * MAX_BOUNCE_ANGLE
                b.vx = BALL_SPEED * math.sin(theta)
                b.vy = -BALL_SPEED * math.cos(theta)
                b.y = top - BALL_RADIUS
                b.color = p.pid
                return

    def scores(self) -> dict[int, int]:
        counts = {i: 0 for i in range(self.n)}
        for row in self.grid:
            for c in row:
                if c != NEUTRAL:
                    counts[c] += 1
        return counts

    def frame(self) -> dict:
        """A compact replay frame. The grid is encoded as one string per row, each
        char a base36 owner id or '.' for a neutral tile -- far smaller than an
        int-list per tile across hundreds of frames."""
        return {
            "tick": self.tick,
            "grid": ["".join("." if c < 0 else _B36[c] for c in row) for row in self.grid],
            "balls": [
                {"x": round(b.x, 3), "y": round(b.y, 3), "c": b.color} for b in self.balls
            ],
            "players": [
                {"x": round(p.x, 3), "y": round(p.helmet_y, 3), "id": p.pid}
                for p in self.players
            ],
            "scores": self.scores(),
        }


def run_game(bot_paths: list[str], seed: int) -> dict:
    """Run a single game. Returns result dict with per-player tiles, winner, replay."""
    n = len(bot_paths)

    # Load bots; a bot that fails to import simply never acts (all NONE).
    bots: list[Callable | None] = []
    load_errors: dict[int, str] = {}
    for i, path in enumerate(bot_paths):
        try:
            bots.append(load_bot(path))
        except Exception as e:
            bots.append(None)
            load_errors[i] = str(e)

    game = Game(num_players=n, seed=seed)
    frames = [game.frame()]

    for t in range(1, MAX_TICKS + 1):
        game.tick = t
        for i, p in enumerate(game.players):
            fn = bots[i]
            action = "NONE" if fn is None else call_bot(fn, game.observation(i))
            game.apply_action(p, action)
        game.step_players()
        game.step_balls()
        if t % REPLAY_EVERY == 0 or t == MAX_TICKS:
            frames.append(game.frame())

    scores = game.scores()
    total = COLS * ROWS
    best = max(scores.values()) if scores else 0
    leaders = [i for i, v in scores.items() if v == best]
    winner = leaders[0] if len(leaders) == 1 else None  # None == draw

    return {
        "winner": winner,
        "scores": scores,
        "total_tiles": total,
        "load_errors": load_errors,
        "replay": {
            "cols": COLS,
            "rows": ROWS,
            "num_players": n,
            "max_ticks": MAX_TICKS,
            "frames": frames,
        },
    }


# --------------------------------------------------------------------------------------
# CLI / tournament driver
# --------------------------------------------------------------------------------------


def write_replay(result: dict, game_num: int, bot_paths: list[str], output_dir: str) -> None:
    rp = dict(result["replay"])
    rp["names"] = [
        os.path.basename(os.path.dirname(p)) or f"player{i + 1}"
        for i, p in enumerate(bot_paths)
    ]
    rp["winner"] = result["winner"]
    rp["final_scores"] = result["scores"]
    path = os.path.join(output_dir, f"sim_{game_num}.json")
    with open(path, "w") as f:
        json.dump(rp, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="RegionBattle Game Engine")
    parser.add_argument("bots", nargs="+", help="Paths to bot files (main.py)")
    parser.add_argument("-r", "--rounds", type=int, default=10, help="Number of games")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Replay output dir")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Base RNG seed")
    args = parser.parse_args()

    bot_paths = args.bots
    n = len(bot_paths)
    names = [
        os.path.basename(os.path.dirname(p)) or f"player{i + 1}"
        for i, p in enumerate(bot_paths)
    ]

    wins = {i: 0 for i in range(n)}
    tile_totals = {i: 0 for i in range(n)}
    draws = 0

    print(f"Running {args.rounds} games between:")
    for i, p in enumerate(bot_paths):
        print(f"  Player {i + 1}: {p}  ({names[i]})")
    print()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for g in range(args.rounds):
        result = run_game(bot_paths, seed=args.seed + g)
        for i, v in result["scores"].items():
            tile_totals[i] += v
        if result["winner"] is None:
            draws += 1
            wtxt = "draw"
        else:
            wins[result["winner"]] += 1
            wtxt = f"Player {result['winner'] + 1}"
        share = "  ".join(f"P{i + 1}={result['scores'][i]}" for i in range(n))
        print(f"Game {g + 1}: {wtxt} wins   [{share}]")
        if result["load_errors"]:
            for i, err in result["load_errors"].items():
                print(f"    (Player {i + 1} failed to load: {err})")
        if args.output_dir:
            write_replay(result, g, bot_paths, args.output_dir)

    print()
    print("FINAL_RESULTS")
    for i in range(n):
        avg = tile_totals[i] / args.rounds if args.rounds else 0
        print(f"Bot_{i + 1}: {wins[i]} games won, {avg:.1f} avg_tiles ({names[i]})")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    main()
