from __future__ import annotations

from geometry import Point


def occupancy_matrix(n: int, selected_points: set[Point]) -> list[list[int]]:
    """Return a dense ``n x n`` occupancy matrix in ``matrix[y][x]`` order.

    ``n`` is the board size and ``selected_points`` is the set of occupied
    lattice points. The function returns a nested Python list of ``0`` and
    ``1`` values, does not mutate the input set, and assumes every point lies
    within the ``0 <= x, y < n`` bounds.
    """
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for x, y in selected_points:
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"point out of bounds for board size {n}: {(x, y)}")
        matrix[y][x] = 1
    return matrix


def ascii_board(n: int, selected_points: set[Point]) -> str:
    """Return a terminal-friendly ASCII board with coordinate headers.

    ``n`` and ``selected_points`` describe the current board state. The
    function returns a string with rows printed from high ``y`` to low ``y``,
    does not mutate state, and assumes the points are valid board coordinates.
    """
    matrix = occupancy_matrix(n, selected_points)
    row_label_width = len(str(max(n - 1, 0)))
    header_prefix = " " * (row_label_width + 1)
    rows: list[str] = []

    if n >= 10:
        rows.append(header_prefix + "".join(str((x // 10) % 10) for x in range(n)))
    rows.append(header_prefix + "".join(str(x % 10) for x in range(n)))

    for y in range(n - 1, -1, -1):
        cells = "".join("X" if matrix[y][x] else "." for x in range(n))
        rows.append(f"{y:>{row_label_width}} {cells}")
    return "\n".join(rows)


def print_board(n: int, selected_points: set[Point]) -> None:
    """Print the ASCII board representation for the current point set.

    The inputs match :func:`ascii_board`. The function writes to standard
    output, returns ``None``, does not mutate Python data structures, and
    assumes the board description is valid.
    """
    print(ascii_board(n, selected_points))


def summarize_state(
    n: int,
    selected_points: set[Point],
    num_active_lines: int | None = None,
) -> str:
    """Return a short text summary of the current board state.

    ``n`` is the board size, ``selected_points`` is the occupied set, and
    ``num_active_lines`` optionally reports the backend line count. The
    function returns a compact string, does not mutate state, and assumes the
    point set belongs to an ``n x n`` grid.
    """
    summary = f"grid={n}x{n}, selected_points={len(selected_points)}"
    if num_active_lines is not None:
        summary += f", active_lines={num_active_lines}"
    return summary
