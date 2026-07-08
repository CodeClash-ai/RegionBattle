# RegionBattle Game Engine for CodeClash

This repository contains the game engine and starter bot for the **RegionBattle**
arena in [CodeClash](https://github.com/CodeClash-ai/CodeClash).

## Overview

RegionBattle is a competitive **territory-painting** game. Each player controls a
helmet-wearing character that patrols the bottom of a rectangular field. Balls fly
around the field bouncing off all four walls (no gravity). Every tick, each ball
paints the tile it is passing over **in its own color**.

- Balls start **neutral** (uncolored) and paint nothing.
- When a ball lands on your **helmet**, it is recolored to *your* color and bounces
  back up — so from then on it paints for you (until an opponent steals it back).
- The exit angle depends on **where** on the helmet the ball struck: center → straight
  up, edges → steep angle. This is how you aim where paint goes.
- Overwriting is allowed: your ball repaints whatever tile it crosses.
- After a fixed tick budget, **whoever owns the most tiles wins.**

Because a neutral ball paints nothing, doing nothing scores nothing — you have to keep
bumping balls to paint, and steal the opponent's balls to deny them.

## Repository Structure

```
RegionBattle/
├── engine.py    # Game engine — runs games between bots (do not edit for the contest)
├── main.py      # Starter bot implementation (EDIT THIS)
└── README.md
```

## Quick Start

1. Edit `main.py` to implement your bot logic.
2. Test locally (bot vs itself):
   ```bash
   python engine.py main.py main.py -r 10 -o /tmp/rb_out
   ```
3. Submit `main.py` to CodeClash.

## Bot Interface

Your bot must implement one function in `main.py`:

### `get_action(obs: dict) -> str`

Return exactly one of these strings each tick:

| Action        | Effect                                        |
|---------------|-----------------------------------------------|
| `"LEFT"`      | move left                                     |
| `"RIGHT"`     | move right                                    |
| `"JUMP"`      | jump straight up (only if on the ground)      |
| `"JUMP_LEFT"` | jump while moving left                        |
| `"JUMP_RIGHT"`| jump while moving right                       |
| `"NONE"`      | do nothing                                    |

Anything else — an invalid string, an exception, or taking longer than the per-tick
time limit — is treated as `"NONE"` for that tick (your character just stays put).

### The observation

`obs` is a plain dict giving you the **complete, deterministic** game state. Every
physics constant is included in `obs["rules"]`, so you can forward-simulate the world
yourself to predict ball trajectories and plan where to stand.

```python
obs = {
    "tick": 42,                 # current tick (starts at 1)
    "max_ticks": 1500,          # game ends after this many ticks

    "field": {
        "width": 32.0,          # world units (== number of tile columns)
        "height": 24.0,         # world units (== number of tile rows)
        "cols": 32,             # tile grid width
        "rows": 24,             # tile grid height
    },

    "rules": {                  # all constants needed to simulate the world forward
        "ball_radius": 0.6,
        "ball_speed": 0.45,     # constant speed magnitude of every ball
        "player_half_width": 2.0,   # helmet spans [x - hw, x + hw]
        "player_height": 2.5,
        "player_speed": 0.5,        # horizontal move distance per tick
        "jump_speed": 0.62,         # initial upward speed of a jump
        "gravity": 0.032,           # downward accel on a jumping player per tick
        "max_bounce_angle": 1.047,  # radians; helmet-edge deflection from vertical
    },

    "you":   {"id": 0, "color": 0},   # your player id; your color == your id

    "players": [                       # every player, including you
        {"id": 0, "color": 0, "x": 9.0,  "y": 21.5, "on_ground": True},
        {"id": 1, "color": 1, "x": 23.0, "y": 21.5, "on_ground": True},
    ],

    "balls": [                         # every ball
        {"x": 9.0, "y": 19.5, "vx": 0.1, "vy": -0.44, "color": -1},
        # color == owning player id, or -1 for a neutral (unclaimed) ball
    ],

    "tiles": [[-1, -1, 0, 1, ...], ...],  # tiles[row][col] = owner id, or -1 neutral
    "scores": {0: 210, 1: 188},           # current tile count per player id
}
```

### Coordinate system

- Origin is the **top-left**. `x` increases to the right, `y` increases **downward**.
- "Up" (toward the top wall) is therefore **decreasing `y`** and a **negative `vy`**.
- Players sit at the bottom; a player's `y` is the world-y of its **helmet surface**
  (smaller `y` = jumped higher). Tile `[row][col]` covers world region
  `x ∈ [col, col+1)`, `y ∈ [row, row+1)`.

## Physics, exactly

Each tick, in order: every player's action is applied (move + maybe start a jump);
**characters are solid** — two players at the same height can't pass through each other:
you're simply *blocked* if you try to walk into another player (you can't push them). But
they only block at similar heights, so you can **jump up and over** a player, and land
**on their head** to stand there. A player being stood on **cannot jump** (its
`can_jump` is `false`) though it can still move. Airborne players integrate under
gravity. Then every ball moves by `(vx, vy)`, painting each tile along its path in its
color, and finally collides with the solid characters. **Touching any part of a
character claims the ball for that player** (it recolors). The **top of the head**
launches it upward at an angle set by where it struck; the **sides and underside**
simply bounce it (reflect) — either way it turns that player's color. A ball then reflects off any wall it hits
(speed preserved, so `|v|` is always `ball_speed`). Finally, if a ball crossed a
helmet's top surface while descending and was within `[x - hw, x + hw]` horizontally,
it is **recolored** to that player and launched upward with

```
offset = clamp((ball.x - player.x) / player_half_width, -1, 1)
theta  = offset * max_bounce_angle
vx, vy = ball_speed * sin(theta),  -ball_speed * cos(theta)
```

The simulation is fully deterministic given the initial seed and the bots' actions.

## Running Games Locally

```bash
# 10 games, write replay files to a directory
python engine.py path/to/bot1.py path/to/bot2.py -r 10 -o /tmp/rb_out

# reproduce a specific game with a fixed base seed
python engine.py bot1.py bot2.py -r 1 -s 123
```

**Options:**
- `-r, --rounds`: number of games to play (default: 10)
- `-o, --output-dir`: directory for `sim_*.json` replay files (optional)
- `-s, --seed`: base RNG seed; game *g* uses `seed + g` (default: 0)

The engine prints a `FINAL_RESULTS` block the tournament parses:

```
FINAL_RESULTS
Bot_1: 7 games won, 355.1 avg_tiles (player1)
Bot_2: 3 games won, 233.2 avg_tiles (player2)
Draws: 0
```

## Strategy Tips

- A neutral ball paints nothing — claim your ball early and keep it in play.
- Steal the opponent's balls: bumping any ball flips it to your color, cutting off
  their tile stream and starting yours.
- Use `obs["rules"]` to forward-simulate where a ball will land, then be there.
- Aim with the helmet offset: hit a ball off-center to send it into regions you don't
  own yet (or into the opponent's territory to overwrite it).
- Jump to intercept balls higher up so you control them sooner.

## License

MIT License — see the CodeClash repository for details.
