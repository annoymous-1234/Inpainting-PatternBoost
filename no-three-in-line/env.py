from __future__ import annotations

from dataclasses import dataclass

from active_lines import ActiveLineBackend
from geometry import Point
from proposal_eval import (
    CandidatePoint,
    ProposalSlot,
    ResolvedProposal,
    collect_existing_slot_line_conflict_counts,
    search_repaired_proposals,
)
from render import print_board, summarize_state


@dataclass
class SlotState:
    """Represent the persistent environment state for one slot.

    ``slot_id`` is the slot index, ``status`` is either ``"MASKED"`` or
    ``"FILLED"``, and ``point`` stores the owned lattice point when filled.
    The dataclass carries state only and is mutated by the environment during
    reset and apply.
    """
    slot_id: int
    status: str
    point: Point | None


class NoThreeInLineEnv:
    """Minimal no-three-in-line environment with an RL-style interface.

    The environment keeps geometry evaluation, repair, and persistent slot
    bookkeeping separate. Its public API mutates internal state on ``reset``
    and ``step`` and returns observations, rewards, and rendering output that
    are deterministic given the current backend and proposal inputs.
    """

    def __init__(
        self,
        n: int,
        num_slots: int,
        max_steps: int | None = None,
        reward_config: dict | None = None,
        proposal_search_config: dict | None = None,
    ) -> None:
        """Initialize an empty environment and reward configuration.

        ``n`` is the grid size, ``num_slots`` is the total slot count,
        ``max_steps`` optionally caps episode length, and ``reward_config`` can
        override the minimal default reward weights and repair costs.
        ``proposal_search_config`` configures the per-step proposal retry loop.
        The constructor returns ``None``, mutates only the new instance, and
        assumes slot ids will later be used consistently with the environment's
        column-pair conventions.
        """
        self.n = n
        self.num_slots = num_slots
        self.max_steps = max_steps
        self.reward_config = {
            "accepted_weight": 1.0,
            "deleted_existing_weight": 1.0,
            "existing_delete_cost": 3.0,
            "proposal_delete_cost": 1.0,
        }
        if reward_config is not None:
            self.reward_config.update(reward_config)
        self.proposal_search_config = {
            "num_search_attempts": 1,
            "allow_existing_deletions": False,
        }
        if proposal_search_config is not None:
            self.proposal_search_config.update(proposal_search_config)

        self.backend = ActiveLineBackend()
        self.slot_states: dict[int, SlotState] = {}
        self.point_to_slot: dict[Point, int] = {}
        self.timestep = 0

    def reset(
        self,
        init_points: list[Point] | None = None,
        init_slot_assignment: dict[Point, int] | None = None,
    ) -> dict[str, object]:
        """Reset backend, slot bookkeeping, and timestep to a new start state.

        ``init_points`` optionally seeds the environment with an already legal
        point set, and ``init_slot_assignment`` maps each seeded point to its
        owning slot. The method returns the initial observation, mutates the
        full environment state in place, and assumes the caller provides basic
        slot-assignment consistency when initializing from points.
        """
        self.backend = ActiveLineBackend()
        self.slot_states = {
            slot_id: SlotState(slot_id=slot_id, status="MASKED", point=None)
            for slot_id in range(self.num_slots)
        }
        self.point_to_slot = {}
        self.timestep = 0

        if init_points is not None:
            if init_slot_assignment is None:
                raise ValueError("init_slot_assignment is required when init_points are provided")

            self.backend.build_from_points(init_points)
            for point in init_points:
                if point not in init_slot_assignment:
                    raise ValueError(f"missing slot assignment for point {point}")
                slot_id = init_slot_assignment[point]
                if slot_id not in self.slot_states:
                    raise ValueError(f"invalid slot id {slot_id} for point {point}")
                if self.slot_states[slot_id].status == "FILLED":
                    raise ValueError(f"slot {slot_id} assigned to multiple points")

                self.slot_states[slot_id] = SlotState(slot_id=slot_id, status="FILLED", point=point)
                self.point_to_slot[point] = slot_id

        return self.observe()

    def observe(self) -> dict[str, object]:
        """Return the minimal current observation used by the first environment.

        The observation contains the sorted selected points and their count.
        The method takes no inputs beyond ``self``, returns a new dictionary,
        does not mutate state, and assumes the backend already reflects the
        persistent legal point set.
        """
        selected_points = sorted(self.backend.get_selected_points())
        return {
            "selected_points": selected_points,
            "num_selected": len(selected_points),
        }

    def step(self, proposal_slots: list[ProposalSlot]) -> tuple[dict[str, object], float, bool, dict]:
        """Apply one proposal step through resolve, repair, and state updates.

        ``proposal_slots`` is an ordered list of env-compatible proposal slots.
        The method returns ``(observation, reward, done, info)``, mutates the
        environment state in place, and assumes proposal duplicates are removed
        before geometric repair evaluates the surviving candidate set.
        """
        existing_points = self.backend.get_selected_points()
        search_result = search_repaired_proposals(
            proposal_slots,
            existing_points,
            existing_point_to_slot=dict(self.point_to_slot),
            num_search_attempts=int(self.proposal_search_config["num_search_attempts"]),
            existing_delete_cost=float(self.reward_config["existing_delete_cost"]),
            proposal_delete_cost=float(self.reward_config["proposal_delete_cost"]),
            allow_existing_deletions=bool(
                self.proposal_search_config["allow_existing_deletions"]
            ),
        )
        conflict_report = search_result["conflict_report"]
        resolve_result = search_result["resolve_result"]
        repair_result = search_result["repair_result"]
        last_attempt = conflict_report["attempts"][-1] if conflict_report["attempts"] else None
        num_conflict_lines = 0 if last_attempt is None else len(last_attempt["line_conflicts"])
        existing_slot_line_conflict_counts = collect_existing_slot_line_conflict_counts(conflict_report)

        apply_result = self._apply_step_result(resolve_result["resolved"], repair_result)
        reward = self._compute_reward(apply_result)

        self.timestep += 1
        done = self._check_done()

        info = {
            "num_resolved": len(resolve_result["resolved"]),
            "unresolved_slot_ids": list(conflict_report["unresolved_slot_ids"]),
            "used_fallback_slot_ids": list(conflict_report["used_fallback_slot_ids"]),
            "num_conflict_lines": num_conflict_lines,
            "accepted_proposal_points": apply_result["accepted_proposal_points"],
            "rejected_proposal_points": apply_result["rejected_proposal_points"],
            "deleted_existing_points": apply_result["deleted_existing_points"],
            "existing_slot_line_conflict_counts": existing_slot_line_conflict_counts,
            "conflict_report": conflict_report,
        }
        return self.observe(), reward, done, info

    def remask_slots(self, slot_ids: list[int]) -> dict[str, object]:
        """Remove filled points owned by ``slot_ids`` and restore those slots to MASKED.

        ``slot_ids`` is the ordered list of slots the caller wants to reopen.
        The method mutates backend and slot bookkeeping in place, does not
        advance ``timestep``, and returns removed slot ids, removed points, plus
        the post-remask observation. Already masked slots are ignored while
        out-of-range slot ids still raise ``ValueError``.
        """

        seen_slot_ids: set[int] = set()
        removed_slot_ids: list[int] = []
        removed_points: list[Point] = []
        for slot_id in slot_ids:
            if slot_id in seen_slot_ids:
                continue
            seen_slot_ids.add(slot_id)
            if not 0 <= slot_id < self.num_slots:
                raise ValueError(f"invalid slot id {slot_id}")

            slot_state = self.slot_states[slot_id]
            if slot_state.status != "FILLED" or slot_state.point is None:
                continue

            self.backend.remove_point(slot_state.point)
            self.point_to_slot.pop(slot_state.point, None)
            self.slot_states[slot_id] = SlotState(slot_id=slot_id, status="MASKED", point=None)
            removed_slot_ids.append(slot_id)
            removed_points.append(slot_state.point)

        return {
            "removed_slot_ids": removed_slot_ids,
            "removed_points": sorted(removed_points),
            "observation": self.observe(),
        }

    def _apply_step_result(
        self,
        resolved_proposals: list[ResolvedProposal],
        repair_result: dict,
    ) -> dict[str, list[Point]]:
        """Apply repaired proposal outcomes to backend and slot bookkeeping.

        ``resolved_proposals`` is the duplicate-filtered proposal list and
        ``repair_result`` names rejected proposal points plus existing points to
        delete. The method returns sorted accepted, rejected, and deleted point
        lists, mutates backend and slot state in place, and assumes repair was
        computed from the same candidate point set.
        """
        rejected_proposal_points = set(repair_result["rejected_proposal_points"])
        deleted_existing_points = sorted(set(repair_result["deleted_existing_points"]))

        accepted_resolved_proposals = [
            proposal
            for proposal in resolved_proposals
            if proposal.point not in rejected_proposal_points
        ]

        for point in deleted_existing_points:
            self.backend.remove_point(point)
            slot_id = self.point_to_slot.pop(point)
            self.slot_states[slot_id] = SlotState(slot_id=slot_id, status="MASKED", point=None)

        accepted_proposal_points: list[Point] = []
        for proposal in accepted_resolved_proposals:
            slot_state = self.slot_states[proposal.slot_id]
            if slot_state.status == "FILLED":
                raise ValueError(f"slot {proposal.slot_id} is already filled")

            self.backend.add_point(proposal.point)
            self.slot_states[proposal.slot_id] = SlotState(
                slot_id=proposal.slot_id,
                status="FILLED",
                point=proposal.point,
            )
            self.point_to_slot[proposal.point] = proposal.slot_id
            accepted_proposal_points.append(proposal.point)

        return {
            "accepted_proposal_points": sorted(accepted_proposal_points),
            "rejected_proposal_points": sorted(rejected_proposal_points),
            "deleted_existing_points": deleted_existing_points,
        }

    def _compute_reward(self, apply_result: dict) -> float:
        """Compute the minimal reward from the applied repair outcome.

        ``apply_result`` is the dictionary returned by ``_apply_step_result``.
        The method returns a scalar reward, does not mutate environment state,
        and assumes the configured reward weights are numeric.
        """
        return (
            self.reward_config["accepted_weight"] * len(apply_result["accepted_proposal_points"])
            - self.reward_config["deleted_existing_weight"] * len(apply_result["deleted_existing_points"])
        )

    def _check_done(self) -> bool:
        """Return whether the episode should terminate at the current timestep.

        The method currently checks only the optional ``max_steps`` bound,
        returns a boolean, does not mutate state, and assumes ``self.timestep``
        has already been advanced for the just-finished step.
        """
        if self.max_steps is not None and self.timestep >= self.max_steps:
            return True
        return False

    def render(self) -> None:
        """Print the current ASCII board and a short backend summary.

        The method takes no inputs beyond ``self``, writes to standard output,
        returns ``None``, and does not mutate environment state. It assumes the
        backend line table reflects the currently selected points.
        """
        selected_points = self.backend.get_selected_points()
        print_board(self.n, selected_points)
        print(
            summarize_state(
                self.n,
                selected_points,
                num_active_lines=len(self.backend.line_to_points),
            )
        )

    def saturate(
        self,
        rng,
        max_steps: int | None = None,
        slot_order: str = "mrv",
        y_order: str = "random",
    ) -> dict[str, object]:
        """Greedily fill legal masked slots through the shared saturation helper.

        ``rng`` supplies deterministic row/slot choices, ``max_steps``
        optionally caps the number of fill attempts, and the order arguments
        choose MRV/random slot ordering plus random/ascending row ordering. The
        method returns the saturation summary, mutates this environment through
        normal ``step`` calls, and assumes the helper owns all legal-candidate
        checks.
        """

        from saturation import greedy_saturate

        return greedy_saturate(
            self,
            rng,
            max_steps=max_steps,
            slot_order=slot_order,
            y_order=y_order,
        )
