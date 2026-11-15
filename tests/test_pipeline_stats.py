import csv
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from control_vision.pipeline.cycles import count_zone_cycles
from control_vision.pipeline.stats import robot_statistics
from control_vision.pipeline.zones import field_position, point_in_polygon, zone_at
from control_vision.scout.exports import export_scout_json, export_stats_csv
from control_vision.storage.database import ProjectDatabase

FPS = 8
DT = 1 / FPS
ZONES = {
    "load": {"type": "loading_zone", "points": [[0, 0], [2, 0], [2, 2], [0, 2]]},
    "goal": {"type": "garden", "points": [[8, 8], [10, 8], [10, 10], [8, 10]]},
}
LOAD, TRANSIT, GOAL = (1.0, 1.0), (5.0, 5.0), (9.0, 9.0)


def track(*legs, robot_id=1, conf=0.9, start_frame=0):
    """Rows at 8 fps for legs of ((x, y), frame_count)."""
    rows, frame = [], start_frame
    for (x, y), frames in legs:
        for _ in range(frames):
            rows.append((robot_id, frame, frame * DT, x, y, conf))
            frame += 1
    return rows


class CycleDebounceTests(unittest.TestCase):
    def test_single_sample_spike_into_scoring_does_not_create_cycle(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 1), (TRANSIT, 8))
        cycles, events = count_zone_cycles(rows, ZONES)
        self.assertEqual(cycles, [])
        self.assertEqual(events, [])

    def test_short_visit_never_counts_but_half_second_dwell_does(self):
        # 4 samples span 0.375 s (< 0.5 s): never a zone entry.
        short = track((LOAD, 8), (TRANSIT, 8), (GOAL, 4), (TRANSIT, 8))
        self.assertEqual(count_zone_cycles(short, ZONES)[0], [])
        # 5 samples span exactly 0.5 s: counts.
        dwell = track((LOAD, 8), (TRANSIT, 8), (GOAL, 5), (TRANSIT, 8))
        self.assertEqual(len(count_zone_cycles(dwell, ZONES)[0]), 1)

    def test_entry_time_is_first_seen_regardless_of_history(self):
        for transit_frames in (4, 5, 7, 16, 30):
            with self.subTest(transit_frames=transit_frames):
                rows = track((LOAD, 8), (TRANSIT, transit_frames), (GOAL, 8))
                cycles, events = count_zone_cycles(rows, ZONES)
                self.assertEqual(len(cycles), 1)
                self.assertAlmostEqual(cycles[0]["start_t_s"], 0.0)
                self.assertAlmostEqual(cycles[0]["end_t_s"], (8 + transit_frames) * DT)
                self.assertAlmostEqual(events[0]["t_s"], cycles[0]["end_t_s"])

    def test_first_zone_obeys_debounce(self):
        # Starts in pickup for only 0.25 s: that is not a pickup visit.
        rows = track((LOAD, 3), (TRANSIT, 8), (GOAL, 8))
        self.assertEqual(count_zone_cycles(rows, ZONES)[0], [])

    def test_tracking_gap_restarts_dwell(self):
        # One goal sample, 2 s unobserved, then 0.125 s in the goal: never 0.5 s
        # of continuous observed presence, so no scoring entry.
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 1))
        rows += track((GOAL, 2), (TRANSIT, 8), start_frame=33)
        self.assertEqual(count_zone_cycles(rows, ZONES)[0], [])

    def test_out_of_order_rows_are_sorted(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 8))
        expected = count_zone_cycles(rows, ZONES)
        shuffled = list(rows)
        random.Random(5).shuffle(shuffled)
        self.assertEqual(count_zone_cycles(shuffled, ZONES), expected)
        self.assertEqual(len(expected[0]), 1)

    def test_dwelling_in_or_reentering_scoring_without_pickup_adds_no_cycle(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 24), (TRANSIT, 1), (GOAL, 8),
                     (TRANSIT, 8), (GOAL, 8))
        cycles, events = count_zone_cycles(rows, ZONES)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(events), 1)

    def test_second_cycle_requires_reentering_pickup(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 8), (TRANSIT, 8), (LOAD, 8),
                     (TRANSIT, 8), (GOAL, 8))
        cycles, _ = count_zone_cycles(rows, ZONES)
        self.assertEqual(len(cycles), 2)
        self.assertAlmostEqual(cycles[1]["start_t_s"], 32 * DT)
        self.assertAlmostEqual(cycles[1]["end_t_s"], 48 * DT)
        self.assertEqual(cycles[0]["confidence"], "medium")


class ZoneTests(unittest.TestCase):
    def test_point_in_polygon(self):
        square = [[0, 0], [2, 0], [2, 2], [0, 2]]
        self.assertTrue(point_in_polygon(1, 1, square))
        self.assertFalse(point_in_polygon(3, 1, square))
        self.assertFalse(point_in_polygon(-1, 1, square))
        self.assertFalse(point_in_polygon(1, 1, [[0, 0], [2, 2]]))
        concave = [[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]]  # notch at the top
        self.assertTrue(point_in_polygon(1, 0.5, concave))
        self.assertFalse(point_in_polygon(2, 3, concave))

    def test_zone_at_skips_roi_and_uses_first_match(self):
        zones = {"goal_roi": {"type": "goal_roi", "points": [[0, 0], [5, 0], [5, 5], [0, 5]]},
                 "a": {"type": "garden", "points": [[0, 0], [3, 0], [3, 3], [0, 3]]},
                 "b": {"type": "hive", "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}}
        self.assertEqual(zone_at(2, 2, zones), "a")
        self.assertEqual(zone_at(3.5, 3.5, zones), "b")
        self.assertIsNone(zone_at(4.5, 0.5, zones))
        self.assertIsNone(zone_at(20, 20, {}))

    def test_field_position_clamps_near_edge_and_drops_far_outside(self):
        self.assertEqual(field_position(12.3, -0.2), (12.0, 0.0))
        self.assertEqual(field_position(6.0, 6.0), (6.0, 6.0))
        self.assertIsNone(field_position(30.0, 5.0))
        self.assertIsNone(field_position(5.0, -1.0))
        self.assertIsNone(field_position(float("nan"), 5.0))


class RobotStatisticsTests(unittest.TestCase):
    def test_straight_line_distance_and_speed(self):
        rows = [(1, frame, frame * DT, 2.0 + 2.0 * frame * DT, 6.0, 0.9) for frame in range(33)]
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertAlmostEqual(stats["distance_ft"], 8.0, delta=0.05)
        self.assertAlmostEqual(stats["top_speed_ft_s"], 2.0, delta=0.05)
        shuffled = list(rows)
        random.Random(2).shuffle(shuffled)
        self.assertAlmostEqual(robot_statistics(shuffled, [], ZONES)[1]["distance_ft"], stats["distance_ft"])

    def test_single_frame_spike_is_ignored(self):
        rows = track(((5.0, 5.0), 12)) + [(1, 12, 12 * DT, 11.0, 11.0, 0.9)]
        rows += [(1, frame, frame * DT, 5.0, 5.0, 0.9) for frame in range(13, 24)]
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertLess(stats["distance_ft"], 0.05)
        self.assertLess(stats["top_speed_ft_s"], 0.1)

    def test_identity_jump_is_not_integrated(self):
        rows = track(((3.0, 3.0), 8), ((10.0, 10.0), 8))
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertLess(stats["distance_ft"], 0.05)
        self.assertLess(stats["top_speed_ft_s"], 0.1)

    def test_stationary_jitter_is_smoothed(self):
        rows, raw = [], 0.0
        for frame in range(80):
            offset = 0.05 if frame % 2 else -0.05
            rows.append((1, frame, frame * DT, 6.0 + offset, 6.0 + offset, 0.9))
            if frame:
                raw += 0.1 * 2 ** 0.5
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertLess(stats["distance_ft"], 0.25 * raw)
        self.assertLess(stats["top_speed_ft_s"], 0.5)

    def test_tracking_gap_is_not_integrated_or_credited_to_zone(self):
        rows = track((LOAD, 8))
        rows += track(((4.0, 4.0), 8), start_frame=33)  # 3.25 s unobserved
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertLess(stats["distance_ft"], 0.05)
        self.assertAlmostEqual(stats["zone_time_s"]["load"], 7 * DT)
        self.assertLess(stats["top_speed_ft_s"], 0.1)

    def test_zone_time_uses_debounced_zones(self):
        rows = track((TRANSIT, 8), (GOAL, 1), (TRANSIT, 8), (GOAL, 16))
        stats = robot_statistics(rows, [], ZONES)[1]
        # The one-sample spike earns nothing; the 2 s dwell earns 15 intervals.
        self.assertAlmostEqual(stats["zone_time_s"]["goal"], 15 * DT)
        self.assertEqual(stats["zone_time_s"]["load"], 0.0)

    def test_off_field_samples_are_dropped(self):
        rows = [(1, frame, frame * DT, 14.0 + 0.25 * frame, 6.0, 0.9) for frame in range(24)]
        stats = robot_statistics(rows, [], ZONES)[1]
        self.assertEqual(stats["distance_ft"], 0.0)
        self.assertEqual(stats["confidence"], "low")

    def test_zone_trip_cycles_cap_confidence_at_medium(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 8), conf=0.95)
        cycles, _ = count_zone_cycles(rows, ZONES)
        self.assertEqual(robot_statistics(rows, cycles, ZONES)[1]["confidence"], "medium")
        db_cycles = [{key: value for key, value in cycle.items() if key != "confidence"} for cycle in cycles]
        self.assertEqual(robot_statistics(rows, db_cycles, ZONES)[1]["confidence"], "medium")
        ball_cycles = [dict(cycle, confidence="high") for cycle in cycles]
        self.assertEqual(robot_statistics(rows, ball_cycles, ZONES)[1]["confidence"], "high")
        low = track((LOAD, 8), conf=0.2)
        self.assertEqual(robot_statistics(low, [], ZONES)[1]["confidence"], "low")

    def test_cycle_count_and_average(self):
        cycles = [{"robot_id": 1, "start_t_s": 0.0, "end_t_s": 4.0},
                  {"robot_id": 1, "start_t_s": 10.0, "end_t_s": 16.0},
                  {"robot_id": 2, "start_t_s": 0.0, "end_t_s": 1.0}]
        stats = robot_statistics(track((LOAD, 8)), cycles, ZONES)[1]
        self.assertEqual(stats["cycles"], 2)
        self.assertAlmostEqual(stats["cycle_time_avg_s"], 5.0)

    def test_sqlite_rows_are_accepted(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        cycle = connection.execute("SELECT 1 AS robot_id, 0.0 AS start_t_s, 3.0 AS end_t_s").fetchone()
        rows = [connection.execute("SELECT 1, ?, ?, 1.0, 1.0, 0.9", (frame, frame * DT)).fetchone()
                for frame in range(8)]
        connection.close()
        stats = robot_statistics(rows, [cycle], ZONES)[1]
        self.assertEqual((stats["cycles"], stats["cycle_time_avg_s"], stats["confidence"]), (1, 3.0, "medium"))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.robots = [{"id": 1, "team_number": 1001, "alliance": "red"},
                       {"id": 2, "team_number": 2002, "alliance": "blue"}]
        self.events = [
            {"id": 10, "robot_id": 1, "t_s": 10.0, "status": "confirmed", "points": 5, "event_type": "nectar_floor"},
            {"id": 11, "robot_id": 2, "t_s": 20.0, "status": "flagged", "points": 10, "event_type": "nectar_floor"},
            {"id": 12, "robot_id": 1, "t_s": 30.0, "status": "rejected", "points": 7, "event_type": "nectar_floor"},
            {"id": 13, "robot_id": 1, "t_s": 40.0, "status": "confirmed", "points": 15, "event_type": "major_foul",
             "scoring_alliance": "blue"},
        ]
        # Clip rows as clips_for_video returns them: no event_type or alliance.
        self.clips = [{"id": 100 + i, "scoring_event_id": event["id"], "t_s": event["t_s"],
                       "robot_id": event["robot_id"], "points": event["points"], "status": event["status"],
                       "team_number": 1001 if event["robot_id"] == 1 else 2002,
                       "path": f"C:\\project\\clips\\event_{event['id']}.mp4"} for i, event in enumerate(self.events)]
        self.video = {"path": "D:\\videos\\Q1 match.mp4", "event_code": "TEST", "match_number": "Q1",
                      "official_red": 100, "official_blue": 90, "red_disqualified": 0, "blue_disqualified": 0}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "scout.json"

    def export(self, video=None, **kwargs):
        export_scout_json(self.target, video or self.video, self.robots, kwargs.get("tracks", []), [],
                          self.events, self.clips, kwargs.get("photos", []), kwargs.get("zones", ZONES),
                          kwargs.get("official_scores"))
        return json.loads(self.target.read_text(encoding="utf-8"))

    def test_clips_export_status_recipient_and_credited_points(self):
        payload = self.export()
        clips = {clip["clip"]: clip for clip in payload["clips"]}
        confirmed = clips["event_10.mp4"]
        self.assertEqual((confirmed["status"], confirmed["recipient_alliance"], confirmed["points"],
                          confirmed["robot_points"], confirmed["robot"]), ("confirmed", "red", 5, 5, 1001))
        self.assertEqual(clips["event_11.mp4"]["status"], "flagged")
        self.assertIsNone(clips["event_11.mp4"]["points"])
        self.assertIsNone(clips["event_11.mp4"]["robot_points"])
        self.assertEqual((clips["event_12.mp4"]["points"], clips["event_12.mp4"]["robot_points"]), (0, 0))
        foul = clips["event_13.mp4"]
        self.assertEqual((foul["recipient_alliance"], foul["points"], foul["robot_points"], foul["robot"]),
                         ("blue", 15, 0, 1001))
        for alliance in ("red", "blue"):
            credited = sum(clip["points"] for clip in payload["clips"]
                           if clip["status"] == "confirmed" and clip["recipient_alliance"] == alliance)
            self.assertEqual(credited, payload["alliance_results"][alliance]["points_video"])
        teams = {team["team_number"]: team for team in payload["teams"]}
        robot_credit = sum(clip["robot_points"] for clip in payload["clips"]
                           if clip["robot"] == 1001 and clip["status"] == "confirmed")
        self.assertEqual(robot_credit, teams[1001]["points_video"])

    def test_clip_points_respect_alliance_disqualification(self):
        payload = self.export(dict(self.video, red_disqualified=1))
        clips = {clip["clip"]: clip for clip in payload["clips"]}
        self.assertEqual((clips["event_10.mp4"]["points"], clips["event_10.mp4"]["robot_points"]), (0, 0))
        self.assertEqual(clips["event_13.mp4"]["points"], 15)
        self.assertEqual(payload["alliance_results"]["red"]["points_video"], 0)
        self.assertEqual(payload["teams"][0]["points_video"], 0)

    def test_team_disqualification_clears_team_and_clip_robot_points(self):
        export_scout_json(self.target, self.video, self.robots, [], [], self.events, self.clips, [], ZONES,
                          qualification_disqualified_robot_ids={1})
        payload = json.loads(self.target.read_text(encoding="utf-8"))
        clip = next(clip for clip in payload["clips"] if clip["clip"] == "event_10.mp4")
        self.assertEqual((clip["points"], clip["robot_points"]), (5, 0))
        self.assertEqual(payload["teams"][0]["points_video"], 0)
        self.assertEqual(payload["teams"][0]["ranking_points_video"], 0)

    def test_points_official_is_never_invented_per_robot(self):
        payload = self.export()
        self.assertTrue(all(team["points_official"] is None for team in payload["teams"]))
        self.assertEqual(payload["official_alliance_scores"], {"red": 100, "blue": 90})

    def test_official_scores_fill_missing_alliances_from_video(self):
        self.assertEqual(self.export(official_scores={})["official_alliance_scores"], {"red": 100, "blue": 90})
        self.assertEqual(self.export(official_scores={"red": 50})["official_alliance_scores"],
                         {"red": 50, "blue": 90})
        self.assertEqual(self.export(official_scores={"red": 50, "blue": None})["official_alliance_scores"],
                         {"red": 50, "blue": None})

    def test_file_names_and_photo_team_mapping(self):
        photos = [{"robot_id": 2, "team_number": 2002, "path": "C:\\project\\photos\\robot_2002.png"},
                  {"robot_id": 1, "team_number": 1001, "path": "/data/photos/robot_1001.png"}]
        payload = self.export(photos=photos)
        self.assertEqual(payload["video"], "Q1 match.mp4")
        self.assertEqual(payload["photos"], [{"team_number": 2002, "photo": "robot_2002.png"},
                                             {"team_number": 1001, "photo": "robot_1001.png"}])
        self.assertEqual({clip["clip"] for clip in payload["clips"]},
                         {f"event_{event['id']}.mp4" for event in self.events})

    def test_zone_keys_and_rounding_match_for_tracked_and_untracked_robots(self):
        zones = dict(ZONES, goal_roi={"type": "goal_roi", "points": [[0, 0], [1, 0], [1, 1]]})
        rows = [(1, frame, frame / 3, 1.0, 1.0, 0.9) for frame in range(8)]
        payload = self.export(tracks=rows, zones=zones)
        tracked, untracked = payload["teams"]
        self.assertEqual(set(tracked["zone_time_s"]), {"load", "goal"})
        self.assertEqual(set(untracked["zone_time_s"]), {"load", "goal"})
        self.assertEqual(tracked["zone_time_s"]["load"], round(7 / 3, 3))
        self.assertIsInstance(untracked["distance_ft"], float)
        moving = [(1, frame, frame * DT, 2.0 + frame * DT / 3, 6.0, 0.9) for frame in range(25)]
        team = self.export(tracks=moving)["teams"][0]
        self.assertEqual(team["distance_ft"], 1.0)
        self.assertEqual(team["top_speed_ft_s"], round(1 / 3, 3))
        self.assertIsInstance(untracked["cycle_time_avg_s"], float)

    def test_empty_input_exports_valid_documents(self):
        export_scout_json(self.target, {"path": "match.mp4"}, [], [], [], [], [], [], {})
        payload = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual((payload["teams"], payload["clips"], payload["photos"]), ([], [], []))
        self.assertEqual(payload["alliance_results"]["red"]["points_video"], 0)
        self.assertEqual(payload["video"], "match.mp4")
        csv_path = Path(self.directory.name) / "empty.csv"
        export_stats_csv(csv_path, [], [], [], {}, events=[], video={})
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(list(csv.reader(stream)), [
                ["team_number", "alliance", "cycles", "cycle_time_avg_s", "top_speed_ft_s", "distance_ft",
                 "zone_time_s", "points_video", "points_official", "ranking_points_video", "confidence"]])

    def test_csv_matches_scout_json_numbers(self):
        rows = track((LOAD, 8), (TRANSIT, 8), (GOAL, 8))
        cycles, _ = count_zone_cycles(rows, ZONES)
        reviewed = [event for event in self.events if event["status"] != "flagged"]
        # Red 5 vs blue 15 (foul credit): blue wins; with red DQ red's team points clear.
        expected = {0: ([5, 0], [0, 3]), 1: ([0, 0], [0, 3])}
        for red_dq, (points, ranking) in expected.items():
            with self.subTest(red_disqualified=red_dq):
                video = dict(self.video, red_disqualified=red_dq)
                export_scout_json(self.target, video, self.robots, rows, cycles, reviewed, [], [], ZONES)
                teams = json.loads(self.target.read_text(encoding="utf-8"))["teams"]
                csv_path = Path(self.directory.name) / "stats.csv"
                export_stats_csv(csv_path, self.robots, rows, cycles, ZONES, events=reviewed, video=video)
                with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                    csv_rows = list(csv.DictReader(stream))
                self.assertEqual(len(csv_rows), len(teams))
                for csv_row, team in zip(csv_rows, teams):
                    self.assertEqual(csv_row["points_official"], "")
                    self.assertEqual(json.loads(csv_row["zone_time_s"]), team["zone_time_s"])
                    for key in ("team_number", "cycles", "points_video", "ranking_points_video"):
                        self.assertEqual(int(csv_row[key]), team[key], key)
                    for key in ("cycle_time_avg_s", "top_speed_ft_s", "distance_ft"):
                        self.assertEqual(float(csv_row[key]), team[key], key)
                    self.assertEqual(csv_row["confidence"], team["confidence"])
                    self.assertEqual(csv_row["alliance"], team["alliance"])
                self.assertEqual([int(row["points_video"]) for row in csv_rows], points)
                self.assertEqual([int(row["ranking_points_video"]) for row in csv_rows], ranking)
                self.assertEqual(teams[0]["cycles"], 1)
                self.assertEqual(teams[0]["confidence"], "medium")

    def test_csv_legacy_points_use_int_robot_keys(self):
        csv_path = Path(self.directory.name) / "legacy.csv"
        export_stats_csv(csv_path, self.robots, [], [], ZONES, {1: 5, 2: 0}, {1: 2})
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row["points_video"], row["ranking_points_video"]) for row in rows],
                         [("5", "2"), ("0", "0")])

    def test_scout_export_accepts_sqlite_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            db = ProjectDatabase(Path(directory) / "event.db")
            project = db.add_project("event")
            video_id = db.add_video(project, Path(directory) / "videos" / "Q2.mp4",
                                    {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}, "TEST", "Q2")
            red = db.add_robot(video_id, 1001, "red", 50, 60)
            db.add_robot(video_id, 2002, "blue", 70, 80)
            scored = db.add_scoring_event(red, 12.0)
            db.review_event(scored, "confirmed", 5, red, "nectar_floor", "high")
            foul = db.add_scoring_event(red, 30.0)
            db.review_event(foul, "confirmed", 15, red, "major_foul", "high", "blue")
            pending = db.add_scoring_event(red, 50.0)
            for event_id in (scored, foul, pending):
                db.add_clip(event_id, Path(directory) / "clips" / f"clip_{event_id}.mp4", 0.0, 6.0)
            db.set_official_scores(video_id, 5, 15)
            db.replace_tracks([(red, frame, frame * DT, 1.0, 1.0, 0.9) for frame in range(8)], [red])
            video = db.connection.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
            target = Path(directory) / "scout.json"
            export_scout_json(target, video, db.robots_for_video(video_id), db.tracks_for_video(video_id),
                              db.cycles_for_video(video_id), db.events_for_video(video_id),
                              db.clips_for_video(video_id), db.photos_for_video(video_id), ZONES)
            db.close()
            payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["video"], "Q2.mp4")
        self.assertEqual(payload["official_alliance_scores"], {"red": 5, "blue": 15})
        self.assertEqual([(clip["recipient_alliance"], clip["points"], clip["robot_points"]) for clip in payload["clips"]],
                         [("red", 5, 5), ("blue", 15, 0), ("red", None, None)])
        self.assertEqual([team["points_video"] for team in payload["teams"]], [5, 0])
        self.assertEqual(payload["teams"][0]["zone_time_s"], {"load": 0.875, "goal": 0.0})


if __name__ == "__main__":
    unittest.main()
