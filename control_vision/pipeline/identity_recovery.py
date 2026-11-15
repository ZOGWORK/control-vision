"""Conservative re-identification of seeded robots after a tracker ID change.

BoT-SORT gives a robot a new track ID when it loses the old one for longer than
its buffer or when the reappearing box no longer overlaps the predicted one.
Without help, a seeded robot whose track ends is never recorded again. This
module re-links such a robot to a new track only when the evidence leaves a
single answer; any doubt leaves the robot lost. Unknown beats wrong.

A seeded robot whose assigned track has been missing for at least
``min_absence_s`` (and at most ``max_absence_s``) is *lost*. A new track can be
re-linked to a lost robot only when every rule below holds:

1. The track ID was never assigned to any robot before, and it was first seen
   after the robot was last seen (a track that coexisted with the robot is a
   different object).
2. The track's first footpoint, in field feet, lies within the robot's speed
   gate: ``max_speed_ft_s * (first_seen - last_seen) + margin_ft`` of the
   robot's last field position.
3. No other unassigned track has been inside the robot's gate at any frame
   since the robot vanished, so another track nearby, even one that has since
   disappeared or that failed the colour check, blocks re-linking.
4. The match is one-to-one: the robot has exactly one candidate and the track
   is a candidate of exactly one absent robot. Pairs come from an optimal
   (minimum total distance) assignment over lost robots and candidates, and a
   pair is kept only when both sides had no other option.
5. A robot whose assigned track is visible is tracked and is never re-linked.
6. After ``max_absence_s`` the robot's recovery window closes for the rest of
   the run and it stays lost.
7. Alliance colour, when the robot's alliance is known: inside the new box the
   other alliance's colour covers at most ``colour_other_max`` of the pixels
   and no more than the robot's own colour, and the own-colour fraction is
   within ``colour_ratio_max`` times the robot's reference (the median over its
   first sightings after seeding, never updated after a re-link). A robot
   without a reference never passes. This rejects robots of the other alliance
   and robots of a very different colour level, but it cannot tell apart two
   same-alliance robots that show similar amounts of their alliance colour.

Robots that were never seeded are never assigned; this only restores the
identity of a robot that the user seeded and that was matched in the opening
window.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

# A robot moving faster than this between two sightings is treated as a
# different object. FTC robots rarely exceed about 10 ft/s; 15 ft/s leaves room.
MAX_ROBOT_SPEED_FT_S = 15.0
# Footpoints jitter by roughly half a robot width between boxes, and corner
# calibration is approximate, so the gate is never smaller than this.
GATE_MARGIN_FT = 1.0
# Short gaps are BoT-SORT's to bridge (it re-finds lost tracks under the same
# ID). Only a robot missing for at least this long is treated as lost.
MIN_ABSENCE_S = 0.5
# The speed gate covers the 17 ft field diagonal after about 1.1 s, so beyond
# that only the exclusivity rules protect the identity. Unseeded robots that
# are undetected at the time can still be mistaken for the lost one, and the
# longer the gap, the more likely that becomes. Stop trying after this long.
MAX_ABSENCE_S = 10.0
# Alliance colour check (rule 7). Field tape, game elements and shadows put a
# little of either colour in most boxes, hence the small tolerances.
COLOUR_OTHER_MAX = 0.02
COLOUR_OWN_MIN = 0.01
COLOUR_RATIO_MAX = 3.0
COLOUR_UPPER_FLOOR = 0.05
COLOUR_REFERENCE_SAMPLES = 8


@dataclass(frozen=True)
class Sighting:
    t_s: float
    x_ft: float
    y_ft: float


@dataclass(frozen=True)
class Recovery:
    robot_id: int
    old_track_id: int
    new_track_id: int
    t_s: float
    absence_s: float
    distance_ft: float
    gate_ft: float


@dataclass
class RecoveryRules:
    max_speed_ft_s: float = MAX_ROBOT_SPEED_FT_S
    margin_ft: float = GATE_MARGIN_FT
    min_absence_s: float = MIN_ABSENCE_S
    max_absence_s: float = MAX_ABSENCE_S
    colour_other_max: float = COLOUR_OTHER_MAX
    colour_own_min: float = COLOUR_OWN_MIN
    colour_ratio_max: float = COLOUR_RATIO_MAX
    colour_upper_floor: float = COLOUR_UPPER_FLOOR
    colour_reference_samples: int = COLOUR_REFERENCE_SAMPLES

    def gate_ft(self, elapsed_s: float) -> float:
        return self.max_speed_ft_s * max(0.0, elapsed_s) + self.margin_ft


def _distance(a: Sighting, x_ft: float, y_ft: float) -> float:
    return math.hypot(a.x_ft - x_ft, a.y_ft - y_ft)


def _optimal_pairs(options: dict[int, dict[int, float]]) -> dict[int, int]:
    """Minimum-total-distance one-to-one assignment, most pairs first.

    The number of seeded robots is at most four, so exhaustive search is exact
    and cheap.
    """
    robots = sorted(options)
    best: tuple[int, float, dict[int, int]] = (0, 0.0, {})

    def search(index: int, used: set[int], mapping: dict[int, int], total: float) -> None:
        nonlocal best
        if index == len(robots):
            if (len(mapping), -total) > (best[0], -best[1]):
                best = (len(mapping), total, dict(mapping))
            return
        robot_id = robots[index]
        search(index + 1, used, mapping, total)
        for track_id, distance in options[robot_id].items():
            if track_id in used:
                continue
            mapping[robot_id] = track_id
            search(index + 1, used | {track_id}, mapping, total + distance)
            del mapping[robot_id]

    search(0, set(), {}, 0.0)
    return best[2]


@dataclass
class IdentityRecovery:
    """Per-run state for re-linking seeded robots to new BoT-SORT track IDs."""

    rules: RecoveryRules = field(default_factory=RecoveryRules)
    # Robot id -> "red" or "blue". None turns the colour check (rule 7) off,
    # which only unit tests of the other rules do.
    alliances: dict[int, str] | None = None
    colour_samples: dict[int, list[float]] = field(default_factory=dict)
    # Unassigned tracks seen inside each missing robot's gate since it vanished.
    rivals_seen: dict[int, set[int]] = field(default_factory=dict)
    reference_closed: set[int] = field(default_factory=set)
    last_seen: dict[int, Sighting] = field(default_factory=dict)
    first_seen: dict[int, Sighting | None] = field(default_factory=dict)
    ever_assigned: set[int] = field(default_factory=set)
    expired: set[int] = field(default_factory=set)
    _reported_ambiguity: set[tuple[int, int, str]] = field(default_factory=set)

    def note_assignment(self, track_id: int) -> None:
        self.ever_assigned.add(track_id)

    def apply(self, recovery: Recovery) -> None:
        """Record a re-link. The robot's colour reference is frozen from now on."""
        self.ever_assigned.add(recovery.new_track_id)
        self.reference_closed.add(recovery.robot_id)
        self.rivals_seen.pop(recovery.robot_id, None)

    def observe_tracks(self, t_s: float, visible: dict[int, tuple[float, float] | None]) -> None:
        """Record each track's first sighting; None marks an out-of-field footpoint."""
        for track_id, position in visible.items():
            if track_id not in self.first_seen:
                self.first_seen[track_id] = None if position is None else Sighting(t_s, *position)

    def observe_robot(self, robot_id: int, t_s: float, x_ft: float, y_ft: float,
                      colour: tuple[float, float] | None = None) -> None:
        self.last_seen[robot_id] = Sighting(t_s, x_ft, y_ft)
        samples = self.colour_samples.setdefault(robot_id, [])
        if (colour is not None and robot_id not in self.reference_closed
                and len(samples) < self.rules.colour_reference_samples):
            samples.append(self._own_other(robot_id, colour)[0])

    def wants_colour(self, robot_id: int) -> bool:
        """True while the robot's colour reference is still being collected."""
        return (self.alliances is not None and robot_id not in self.reference_closed
                and len(self.colour_samples.get(robot_id, [])) < self.rules.colour_reference_samples)

    def _own_other(self, robot_id: int, colour: tuple[float, float]) -> tuple[float, float]:
        red, blue = colour
        alliance = (self.alliances or {}).get(robot_id)
        return (red, blue) if alliance == "red" else (blue, red)

    def colour_reference(self, robot_id: int) -> float | None:
        samples = self.colour_samples.get(robot_id)
        return statistics.median(samples) if samples else None

    def colour_matches(self, robot_id: int, colour: tuple[float, float] | None) -> bool:
        """Rule 7. Without an alliance, a reference or a measurement, it fails."""
        if self.alliances is None:
            return True
        reference = self.colour_reference(robot_id)
        if self.alliances.get(robot_id) not in ("red", "blue") or reference is None or colour is None:
            return False
        rules = self.rules
        own, other = self._own_other(robot_id, colour)
        lower = max(rules.colour_own_min, reference / rules.colour_ratio_max)
        upper = max(rules.colour_upper_floor, reference * rules.colour_ratio_max)
        return other <= rules.colour_other_max and other <= own and lower <= own <= upper

    def recover(self, t_s: float, visible: dict[int, tuple[float, float] | None],
                seed_to_track: dict[int, int], logger=None,
                colours: dict[int, tuple[float, float]] | None = None) -> list[Recovery]:
        """Return the re-links that the rules allow at ``t_s``; mutates nothing in seed_to_track.

        ``colours`` maps visible track IDs to the (red, blue) pixel fractions
        inside their boxes in the current frame.
        """
        rules = self.rules
        assigned_now = set(seed_to_track.values())
        absent: dict[int, Sighting] = {}
        for robot_id, track_id in seed_to_track.items():
            # Rule 5: a robot whose track is visible is tracked, never re-linked.
            if track_id in visible or robot_id in self.expired or robot_id not in self.last_seen:
                self.rivals_seen.pop(robot_id, None)
                continue
            last = self.last_seen[robot_id]
            if t_s - last.t_s > rules.max_absence_s:
                # Rule 6: recovery window closed for good.
                self.expired.add(robot_id)
                if logger:
                    logger.warning("Robot %d has been missing for %.1fs since %.2fs; identity recovery stopped and "
                                   "the robot stays lost for the rest of the run", robot_id, t_s - last.t_s, last.t_s)
                continue
            absent[robot_id] = last
        if not absent:
            return []
        unassigned = {track_id: position for track_id, position in visible.items()
                      if track_id not in assigned_now and position is not None}
        candidates: dict[int, dict[int, float]] = {}
        gates: dict[int, float] = {}
        blocked: set[int] = set()
        for robot_id, last in absent.items():
            options: dict[int, float] = {}
            for track_id in unassigned:
                first = self.first_seen.get(track_id)
                # Rule 1: never-assigned track, born after the robot vanished.
                if track_id in self.ever_assigned or first is None or first.t_s <= last.t_s:
                    continue
                # A track born more than max_absence_s later cannot get here:
                # the robot has expired by then (rule 6).
                elapsed = first.t_s - last.t_s
                distance = _distance(last, first.x_ft, first.y_ft)
                # Rule 2: first footpoint inside the speed gate.
                if distance > rules.gate_ft(elapsed):
                    continue
                # Rule 7: alliance colour. A failing track still counts as a
                # rival under rule 3.
                colour = (colours or {}).get(track_id)
                if not self.colour_matches(robot_id, colour):
                    key = (robot_id, track_id, "colour")
                    if logger and key not in self._reported_ambiguity:
                        self._reported_ambiguity.add(key)
                        reference = self.colour_reference(robot_id)
                        logger.info("Robot %d not re-linked to track %d at %.2fs: alliance colour does not match "
                                    "(red/blue %s; own-colour reference %s)", robot_id, track_id, t_s,
                                    "n/a" if colour is None else "%.3f/%.3f" % colour,
                                    "n/a" if reference is None else "%.3f" % reference)
                    continue
                options[track_id] = distance
            gate_now = rules.gate_ft(t_s - last.t_s)
            gates[robot_id] = gate_now
            # Rule 3: any other unassigned track inside the gate, now or at any
            # frame since the robot vanished, blocks.
            seen = self.rivals_seen.setdefault(robot_id, set())
            seen.update(track_id for track_id, (x_ft, y_ft) in unassigned.items()
                        if _distance(last, x_ft, y_ft) <= gate_now)
            if options and (seen - set(options) or len(options) > 1):
                blocked.add(robot_id)
            candidates[robot_id] = options
        claimants: dict[int, set[int]] = {}
        for robot_id, options in candidates.items():
            for track_id in options:
                claimants.setdefault(track_id, set()).add(robot_id)
        pairs = _optimal_pairs({robot_id: options for robot_id, options in candidates.items() if options})
        recoveries = []
        for robot_id, track_id in sorted(pairs.items()):
            last = absent[robot_id]
            reason = None
            if robot_id in blocked:
                reason = "another unassigned track is inside the gate"
            elif len(claimants[track_id]) != 1:
                reason = "more than one missing robot could claim it"
            elif t_s - last.t_s < rules.min_absence_s:
                # Too soon to call the robot lost; keep it as a claimant only.
                continue
            if reason:
                key = (robot_id, track_id, reason)
                if logger and key not in self._reported_ambiguity:
                    self._reported_ambiguity.add(key)
                    logger.info("Robot %d not re-linked to track %d at %.2fs: %s", robot_id, track_id, t_s, reason)
                continue
            recoveries.append(Recovery(robot_id, seed_to_track[robot_id], track_id, t_s,
                                       t_s - last.t_s, candidates[robot_id][track_id], gates[robot_id]))
        return recoveries
