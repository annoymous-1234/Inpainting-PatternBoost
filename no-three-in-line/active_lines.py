from __future__ import annotations

from geometry import LineKey, Point, line_key


class ActiveLineBackend:
    """Maintain active two-point lines for the current selected point set.

    The backend stores selected points, the active lines induced by pairs of
    selected points, and the reverse mapping from each point to the lines that
    currently contain it. Methods on this class mutate the backend state in
    place and assume the environment keeps the persistent point set legal.
    """

    def __init__(self) -> None:
        """Initialize an empty backend with no selected points or active lines.

        The constructor takes no inputs, returns ``None``, mutates only the new
        instance, and assumes no prior state.
        """
        self.selected_points: set[Point] = set()
        self.line_to_points: dict[LineKey, set[Point]] = {}
        self.point_to_active_lines: dict[Point, set[LineKey]] = {}

    def build_from_points(self, points: list[Point]) -> None:
        """Rebuild the backend from a legal list of selected points.

        ``points`` is treated as the full selected set for the new state. The
        method clears the current backend contents, recomputes all pair-induced
        active lines, returns ``None``, and mutates this backend in place. It
        assumes the input point set is already legal and does not validate that
        no line contains three points.
        """
        self.selected_points = set(points)
        self.line_to_points = {}
        self.point_to_active_lines = {point: set() for point in self.selected_points}

        point_list = list(self.selected_points)
        for index, p in enumerate(point_list):
            for q in point_list[index + 1 :]:
                line = line_key(p, q)
                self.line_to_points.setdefault(line, set()).update((p, q))

        for line, line_points in self.line_to_points.items():
            for point in line_points:
                self.point_to_active_lines.setdefault(point, set()).add(line)

    def add_point(self, p: Point) -> None:
        """Insert one accepted point and update all affected active lines.

        ``p`` is a new selected lattice point. The method creates every active
        line through ``p`` and an existing selected point, returns ``None``,
        and mutates all three backend tables in place. It assumes ``p`` is not
        already selected and raises ``ValueError`` if that assumption is
        violated.
        """
        if p in self.selected_points:
            raise ValueError(f"point already selected: {p}")

        existing_points = list(self.selected_points)
        self.selected_points.add(p)
        self.point_to_active_lines[p] = set()

        for q in existing_points:
            line = line_key(p, q)
            self.line_to_points.setdefault(line, set()).update((p, q))
            self.point_to_active_lines[p].add(line)
            self.point_to_active_lines.setdefault(q, set()).add(line)

    def remove_point(self, p: Point) -> None:
        """Remove one selected point and all active lines that contain it.

        ``p`` must already be selected. The method deletes the point, removes
        any active line passing through it from the backend tables, returns
        ``None``, and mutates this backend in place. It assumes the caller is
        sequencing deletions consistently and raises ``KeyError`` if ``p`` is
        absent.
        """
        if p not in self.selected_points:
            raise KeyError(f"point not selected: {p}")

        for line in list(self.point_to_active_lines.get(p, set())):
            line_points = self.line_to_points.pop(line, set())
            for q in line_points:
                if q != p and q in self.point_to_active_lines:
                    self.point_to_active_lines[q].discard(line)

        self.point_to_active_lines.pop(p, None)
        self.selected_points.remove(p)

    def get_selected_points(self) -> set[Point]:
        """Return a shallow copy of the current selected point set.

        The method takes no inputs beyond ``self``, returns a new ``set`` of
        ``Point`` values, does not mutate backend state, and assumes the stored
        points are internally consistent with the line tables.
        """
        return set(self.selected_points)
