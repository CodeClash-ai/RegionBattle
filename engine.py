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
BALL_SPEED = 0.45            # constant ball speed magnitude once a ball is in motion
BALL_REST_GAP = 2.0          # how far above the helmet a ball floats at rest (pre-bump)

PLAYER_HALF_WIDTH = 0.9      # solid character half-width: the whole body is [x-hw, x+hw]
                             # and is used for player-player collision AND ball collision
PLAYER_HEIGHT = 2.5          # from feet up to helmet surface
PLAYER_SPEED = 0.5           # horizontal move per tick (tiles / tick)
JUMP_SPEED = 0.39            # initial upward velocity of a jump (~2.5-tile peak height)
GRAVITY = 0.032              # downward accel applied to an airborne player per tick
STACK_CLEAR = 0.6            # vertical separation above which players can overlap in x
                             # (so one can rise over another and land on its head)
MAX_BOUNCE_ANGLE = math.radians(60)  # helmet-edge hit deflects up to this from vertical

MAX_TICKS = 1500             # tick budget per game
TURN_TIMEOUT = 0.10          # seconds a single get_action call may take before -> NONE
REPLAY_EVERY = 1             # record a replay frame every N ticks (1 = smoothest playback)

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
    y_off: float = 0.0       # feet height above the floor (0 = standing on the floor)
    vy: float = 0.0          # vertical velocity of y_off (up = positive)
    resting: bool = True     # feet are on a surface (the floor or another player's head)
    carrying: bool = False   # another player is standing on this one's head

    @property
    def helmet_y(self) -> float:
        """World-y of the helmet surface (top of the character). Smaller y = higher."""
        return HEIGHT - PLAYER_HEIGHT - self.y_off

    @property
    def can_jump(self) -> bool:
        """A player may jump only when resting on a surface and NOT being stood on."""
        return self.resting and not self.carrying


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

        # Players evenly spaced across the bottom, with a small per-game random jitter
        # so that the sims within a round aren't identical clones (balls start at rest,
        # so the starting layout is the only source of per-game variation).
        self.players: list[PlayerState] = []
        for i in range(num_players):
            frac = (i + 0.5) / num_players
            x = frac * (WIDTH - 2 * PLAYER_HALF_WIDTH) + PLAYER_HALF_WIDTH
            x += self.rng.uniform(-2.0, 2.0)
            x = max(PLAYER_HALF_WIDTH, min(WIDTH - PLAYER_HALF_WIDTH, x))
            self.players.append(PlayerState(pid=i, x=x))

        # One ball per player, resting motionless just above that player's head.
        # Balls start NEUTRAL (uncolored) and paint nothing; they stay put until a
        # player bumps one -- by jumping up into it -- which claims it (recolors to
        # that player) and launches it. A passive bot never starts its ball, so it
        # paints nothing and you must keep bumping to score.
        self.balls: list[Ball] = []
        for p in self.players:
            self.balls.append(Ball(x=p.x, y=p.helmet_y - BALL_REST_GAP, vx=0.0, vy=0.0, color=NEUTRAL))

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
                    "on_ground": p.resting,     # resting on a surface (floor or a head)
                    "can_jump": p.can_jump,     # False while another player stands on you
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
        dx = 0.0
        if action in ("LEFT", "JUMP_LEFT"):
            dx = -PLAYER_SPEED
        elif action in ("RIGHT", "JUMP_RIGHT"):
            dx = PLAYER_SPEED
        if dx:
            # Solid bodies: you can't move INTO a player at your height -- you're blocked
            # (no pushing them). Players more than STACK_CLEAR apart in height don't block,
            # so you can jump up and over one, or move out from under one standing on you.
            target = p.x + dx
            for q in self.players:
                if q is p or abs(p.y_off - q.y_off) >= STACK_CLEAR:
                    continue
                if dx > 0 and p.x <= q.x:
                    target = min(target, q.x - 2 * PLAYER_HALF_WIDTH)
                elif dx < 0 and p.x >= q.x:
                    target = max(target, q.x + 2 * PLAYER_HALF_WIDTH)
            p.x = max(PLAYER_HALF_WIDTH, min(WIDTH - PLAYER_HALF_WIDTH, target))

        if action in ("JUMP", "JUMP_LEFT", "JUMP_RIGHT") and p.can_jump:
            p.vy = JUMP_SPEED
            p.resting = False

    def step_players(self) -> None:
        # Integrate airborne players under gravity.
        for p in self.players:
            if not p.resting:
                p.y_off += p.vy
                p.vy -= GRAVITY

        # Resolve what each player is standing on: the floor, or the head of a lower
        # player it overlaps in x. Settle lowest players first so stacks resolve, and
        # flag the supporting player as "carrying" (which blocks its own jump).
        for p in self.players:
            p.carrying = False
        for p in sorted(self.players, key=lambda q: q.y_off):
            support = 0.0
            carrier = None
            for q in self.players:
                if q is p or abs(p.x - q.x) >= 2 * PLAYER_HALF_WIDTH:
                    continue
                head_top = q.y_off + PLAYER_HEIGHT  # height of q's head above the floor
                if q.y_off < p.y_off and head_top > support:
                    support = head_top
                    carrier = q
            if p.vy <= 0 and p.y_off <= support + 0.05:
                p.y_off = support
                p.vy = 0.0
                p.resting = True
                if carrier is not None:
                    carrier.carrying = True
            else:
                p.resting = False

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

            self._resolve_player_ball(b)

    def _resolve_player_ball(self, b: Ball) -> None:
        """Ball vs solid characters. The character is a solid box [x-hw, x+hw] from its
        head (helmet_y) down to its feet. The TOP of the head *catches* the ball --
        recoloring it to that player and launching it upward at an angle set by where it
        struck (center = straight up, edge = steep). The sides and underside are solid
        and simply *bounce* the ball (reflect, no recolor), so balls ricochet off bodies
        instead of passing through them.

        A top-catch fires whether the ball descends onto the head or a jumping head rises
        into it. Contested catches (players stacked) are resolved fairly: highest head,
        then nearest center, RNG only for an exact tie -- never by player id."""
        hw = PLAYER_HALF_WIDTH

        # 1) Top-of-head catch. Collect every player the ball is resting on the top of.
        catchers = []
        for p in self.players:
            top = p.helmet_y
            on_top = (p.x - hw) <= b.x <= (p.x + hw) and (top - BALL_RADIUS) <= b.y <= top and b.vy > -0.05
            if on_top:
                catchers.append(p)
        if catchers:
            def key(p: PlayerState) -> tuple[float, float]:
                return (round(p.helmet_y, 6), round(abs(b.x - p.x), 6))

            best = min(key(p) for p in catchers)
            finalists = [p for p in catchers if key(p) == best]
            w = finalists[0] if len(finalists) == 1 else finalists[self.rng.randrange(len(finalists))]
            offset = max(-1.0, min(1.0, (b.x - w.x) / hw))
            theta = offset * MAX_BOUNCE_ANGLE
            b.vx = BALL_SPEED * math.sin(theta)
            b.vy = -BALL_SPEED * math.cos(theta)
            b.y = w.helmet_y - BALL_RADIUS
            b.color = w.pid
            return

        # 2) Solid-body bounce off the sides / underside of any character the ball is
        # penetrating. Reflect along the axis of shallower penetration and push clear.
        for p in self.players:
            top = p.helmet_y
            feet = HEIGHT - p.y_off
            overlap_x = (hw + BALL_RADIUS) - abs(b.x - p.x)
            overlap_y = min((b.y + BALL_RADIUS) - top, (feet + BALL_RADIUS) - b.y)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            if overlap_x <= overlap_y:
                if b.x < p.x:
                    b.x = p.x - hw - BALL_RADIUS
                    b.vx = -abs(b.vx)
                else:
                    b.x = p.x + hw + BALL_RADIUS
                    b.vx = abs(b.vx)
            else:
                if b.y < (top + feet) / 2:
                    b.y = top - BALL_RADIUS
                    b.vy = -abs(b.vy)
                else:
                    b.y = feet + BALL_RADIUS
                    b.vy = abs(b.vy)

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
