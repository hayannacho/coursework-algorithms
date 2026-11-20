import random


def seedGrid(n, rng):
    assert type(n) == int
    grid = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = rng.choice([-1, 1])
    return grid


def updateGrid(grid, killing_rate=0.05, reproducing_rate=0.05, rng=None):
    rng = rng or random.Random()
    assert type(grid) == list
    assert type(killing_rate) == float
    assert type(reproducing_rate) == float
    n = len(grid)
    num_activated = int(killing_rate * n * n)
    assert type(num_activated) == int
    num_killed = 0
    for i in range(num_activated):
        x = rng.choice(range(n))
        y = rng.choice(range(n))
        if grid[x][y] == 0:
            continue
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if j == 0 and k == 0:
                    continue
                nx, ny = (x + j, y + k)
                if nx < 0 or nx >= n or ny < 0 or (ny >= n):
                    continue
                if grid[nx][ny] == 0:
                    continue
                if grid[nx][ny] != grid[x][y]:
                    grid[nx][ny] = 0
                    num_killed += 1
    num_reproducing = int(reproducing_rate * n * n)
    assert type(num_reproducing) == int
    num_new = 0
    for i in range(num_reproducing):
        x = rng.choice(range(n))
        y = rng.choice(range(n))
        if grid[x][y] == 0:
            continue
        if grid[x][y] == 0:
            continue
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if j == 0 and k == 0:
                    continue
                nx, ny = (x + j, y + k)
                if nx < 0 or nx >= n or ny < 0 or (ny >= n):
                    continue
                if grid[nx][ny] == 0:
                    grid[nx][ny] = grid[x][y]
                    num_new += 1
    return (num_killed, num_new)


def simulate(size=20, steps=50, seed=42, killing=0.05, reproduction=0.05):
    if (
        not isinstance(size, int)
        or size < 1
        or size > 500
        or not isinstance(steps, int)
        or not 0 <= steps <= 10000
    ):
        raise ValueError("Invalid grid size or steps")
    if not 0 <= killing <= 1 or not 0 <= reproduction <= 1:
        raise ValueError("Rates must be in [0,1]")
    rng = random.Random(seed)
    grid = seedGrid(size, rng)
    history = []
    for step in range(steps + 1):
        cells = [c for row in grid for c in row]
        history.append(
            {"step": step, "red": cells.count(-1), "blue": cells.count(1), "empty": cells.count(0)}
        )
        if step < steps:
            updateGrid(grid, float(killing), float(reproduction), rng)
    return history


if __name__ == "__main__":
    import json

    h = simulate()
    print(json.dumps({"initial": h[0], "final": h[-1], "seed": 42}, indent=2))
