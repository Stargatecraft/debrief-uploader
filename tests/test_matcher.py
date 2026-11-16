"""The matching engine, exercised against the cases from the spec (section 13).

Timing behaviour is the whole risk in this app, so it is tested against
synthetic event streams rather than a running game. Times are relative
seconds from an arbitrary session start; the engine only ever sees floats.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debrief_uploader.config import DEFAULTS
from debrief_uploader.matcher import (
    AUTO, HOLD, NONE, AWAITING_SHOTS, REVIEW_REASON, UPLOADED, ReplayRec,
    ShotRec, cluster_shots, match_cluster, is_orphan, clock_skew, skew_is_sane,
)

T0 = 1_756_000_000.0    # session start, arbitrary epoch


def cfg(**over):
    c = dict(DEFAULTS)
    c.update(over)
    return c


def replay(key, close_min, group="pvp", state=AWAITING_SHOTS, **kw):
    """A replay that closed `close_min` minutes into the session."""
    return ReplayRec(key=key, t_close=T0 + close_min * 60, state=state,
                     match_group=group, **kw)


def shots(*offsets_min):
    return [ShotRec("shot%d.png" % i, T0 + m * 60) for i, m in enumerate(offsets_min)]


def one_cluster(offsets, c=None, now=None):
    c = c or cfg()
    sh = shots(*offsets)
    now = now if now is not None else max(s.t for s in sh) + c["pair_settle"] + 1
    closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"], now,
                                  c["pair_settle_full"])
    assert len(closed) == 1, "expected one closed cluster, got %d" % len(closed)
    return closed[0]


class TestClustering(unittest.TestCase):

    def test_two_shots_seconds_apart_are_one_cluster(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 8)]
        closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"], T0 + 200)
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0]), 2)

    def test_two_battles_worth_of_shots_split(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 8),
              ShotRec("c", T0 + 900), ShotRec("d", T0 + 906)]
        closed, _ = cluster_shots(sh, c["pair_gap"], c["pair_settle"], T0 + 1200)
        self.assertEqual([len(g) for g in closed], [2, 2])

    def test_cluster_stays_open_until_settled(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 8)]
        closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"], T0 + 20)
        self.assertEqual(closed, [])
        self.assertEqual(len(open_), 1)

    def test_chained_shots_within_gap_keep_one_cluster(self):
        # gap is measured from the PREVIOUS shot, not the first: a slow
        # three-tab capture is still one battle.
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 100), ShotRec("c", T0 + 200)]
        closed, _ = cluster_shots(sh, c["pair_gap"], c["pair_settle"], T0 + 400)
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0]), 3)


class TestSpecCases(unittest.TestCase):
    """The ten cases the matcher was designed against."""

    def test_01_happy_path(self):
        r = replay("A", 10)
        d = match_cluster(one_cluster([10 + 40 / 60.0, 10 + 50 / 60.0]), [r], cfg())
        self.assertEqual(d.kind, AUTO)
        self.assertEqual(d.replay.key, "A")

    def test_02_back_to_back_battles_four_minutes_apart(self):
        # the measured p05 gap. Both battles have their shots; each pair must
        # land on its own replay.
        a, b = replay("A", 10), replay("B", 14)
        ca = one_cluster([10.5, 10.6])
        da = match_cluster(ca, [a, b], cfg())
        self.assertEqual(da.kind, AUTO)
        self.assertEqual(da.replay.key, "A")
        a.state = UPLOADED                      # A is paired and gone
        db = match_cluster(one_cluster([14.5, 14.6]), [a, b], cfg())
        self.assertEqual(db.kind, AUTO)
        self.assertEqual(db.replay.key, "B")

    def test_03_skipped_battle_does_not_hold_up_the_next(self):
        # A never got screenshots; B's arrive. Nearest-preceding says B, and
        # pairing never asks (Greg, 2026-11-16).
        a, b = replay("A", 10), replay("B", 25)
        d = match_cluster(one_cluster([25.5, 25.6]), [a, b], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "B"))

    def test_04_early_exit_goes_to_the_battle_just_played(self):
        # The delayed-results trap. A closes 18s after starting (player
        # left), B closes 12 min later, and shots come after B. By file time
        # they go to B. If they are really A's, the battle time printed on
        # them settles it first (test_screen_time), and the site sets aside
        # a scorecard whose players are not B's.
        a = replay("A", 0.3)
        b = replay("B", 12)
        d = match_cluster(one_cluster([13, 13.1]), [a, b], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "B"))

    def test_05_three_shots_all_upload(self):
        r = replay("A", 10)
        c = one_cluster([10.5, 10.6, 10.7])
        d = match_cluster(c, [r], cfg())
        self.assertEqual(d.kind, AUTO)
        self.assertEqual(len(c), 3)

    def test_06_single_shot_pairs(self):
        # one tab only: the site reads whichever tab it is given
        r = replay("A", 10)
        d = match_cluster(one_cluster([10.5]), [r], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "A"))

    def test_07_training_room_takes_no_shots(self):
        # A training room has no scorecard screens, so shots after one belong
        # to neither it nor the battle before it.
        t = replay("T", 20, group="training_room")
        old = replay("A", 5)
        d = match_cluster(one_cluster([20.5, 20.6]), [old, t], cfg())
        self.assertEqual(d.kind, NONE)
        self.assertIn("training room", d.reason)

    def test_08_nine_hour_clock_skew_still_matches(self):
        # The filename/header clock is 9h ahead of the filesystem clock. The
        # engine only reads t_close, so the match is unaffected -- but the
        # skew must be reported as insane so the UI stops showing header time.
        r = replay("A", 10, header_dt=T0 + 10 * 60 + 9 * 3600)
        d = match_cluster(one_cluster([10.5, 10.6]), [r], cfg())
        self.assertEqual(d.kind, AUTO)
        self.assertFalse(skew_is_sane(clock_skew(r)))

    def test_08b_normal_clock_is_sane(self):
        r = replay("A", 10, header_dt=T0 + 10 * 60 - 600)   # 10 min battle
        self.assertTrue(skew_is_sane(clock_skew(r)))

    def test_09_shot_just_before_close_still_matches(self):
        # GRACE: a screenshot snapped 20s before the file finished closing.
        r = replay("A", 10)
        c = one_cluster([10 - 20 / 60.0, 10 + 5 / 60.0])
        d = match_cluster(c, [r], cfg())
        self.assertEqual(d.kind, AUTO)

    def test_09b_shot_well_before_close_does_not_match(self):
        r = replay("A", 10)
        d = match_cluster(one_cluster([5, 5.1]), [r], cfg())
        self.assertEqual(d.kind, NONE)

    def test_10_replay_with_no_shots_becomes_orphan(self):
        r = replay("A", 0)
        c = cfg()
        self.assertFalse(is_orphan(r, [r], T0 + 60, c))
        self.assertTrue(is_orphan(r, [r], T0 + c["orphan_after"] + 1, c))

    def test_10b_two_later_battles_orphan_it_early(self):
        a = replay("A", 0)
        rs = [a, replay("B", 16), replay("C", 32)]
        self.assertTrue(is_orphan(a, rs, T0 + 33 * 60, cfg()),
                        "player has clearly moved on")

    def test_10c_one_later_battle_does_not_orphan_it(self):
        a = replay("A", 0)
        rs = [a, replay("B", 8)]
        self.assertFalse(is_orphan(a, rs, T0 + 12 * 60, cfg()))


class TestGuards(unittest.TestCase):

    def test_too_old_is_refused(self):
        r = replay("A", 0)
        d = match_cluster(one_cluster([40, 40.1]), [r], cfg())
        self.assertEqual(d.kind, NONE)
        self.assertIn("earlier", d.reason)

    def test_already_uploaded_replay_is_not_a_candidate(self):
        r = replay("A", 10, state=UPLOADED)
        d = match_cluster(one_cluster([10.5, 10.6]), [r], cfg())
        self.assertEqual(d.kind, NONE)

    def test_too_many_shots_holds(self):
        r = replay("A", 10)
        c = one_cluster([10.5, 10.55, 10.6, 10.65, 10.7])
        d = match_cluster(c, [r], cfg())
        self.assertEqual(d.kind, HOLD)
        self.assertIn("5 screenshots", d.reason)

    def test_review_mode_holds_everything(self):
        r = replay("A", 10)
        d = match_cluster(one_cluster([10.5, 10.6]), [r], cfg(review_mode=True))
        self.assertEqual((d.kind, d.replay.key, d.reason),
                         (HOLD, "A", REVIEW_REASON))
        # one tab, and a skipped battle before it: paired all the same, and
        # held only because review mode is on
        d = match_cluster(one_cluster([10.5]), [r], cfg(review_mode=True))
        self.assertEqual((d.kind, d.replay.key, d.reason),
                         (HOLD, "A", REVIEW_REASON))
        a, b = replay("A", 10), replay("B", 25)
        d = match_cluster(one_cluster([25.5, 25.6]), [a, b], cfg(review_mode=True))
        self.assertEqual((d.kind, d.replay.key, d.reason),
                         (HOLD, "B", REVIEW_REASON))

    def test_no_replays_at_all(self):
        d = match_cluster(one_cluster([10, 10.1]), [], cfg())
        self.assertEqual(d.kind, NONE)

    def test_older_unpaired_beyond_max_lag_does_not_block(self):
        # A is unpaired but far too old to be a plausible owner: don't make the
        # user resolve a choice that isn't real.
        a, b = replay("A", 0), replay("B", 40)
        d = match_cluster(one_cluster([40.5, 40.6]), [a, b], cfg())
        self.assertEqual(d.kind, AUTO)
        self.assertEqual(d.replay.key, "B")

    def test_older_unpaired_training_room_does_not_block(self):
        t, b = replay("T", 20, group="training_room"), replay("B", 30)
        d = match_cluster(one_cluster([30.5, 30.6]), [t, b], cfg())
        self.assertEqual(d.kind, AUTO)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSettleWindow(unittest.TestCase):
    """How long a group of screenshots waits before it is decided.

    A user watched two scorecards sit for 90 seconds and reasonably concluded
    it was broken. A group that already holds both tabs is complete, so it is
    decided quickly; a lone screenshot still waits, because its partner is
    exactly what is worth waiting for.
    """

    def test_pair_is_decided_quickly(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 6)]
        closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"],
                                      T0 + 6 + c["pair_settle_full"] + 1,
                                      c["pair_settle_full"])
        self.assertEqual(len(closed), 1)
        self.assertEqual(open_, [])

    def test_pair_is_not_decided_before_its_short_window(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 6)]
        closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"],
                                      T0 + 10, c["pair_settle_full"])
        self.assertEqual(closed, [])
        self.assertEqual(len(open_), 1)

    def test_single_screenshot_waits_the_long_window(self):
        c = cfg()
        sh = [ShotRec("a", T0)]
        closed, open_ = cluster_shots(sh, c["pair_gap"], c["pair_settle"],
                                      T0 + c["pair_settle_full"] + 1,
                                      c["pair_settle_full"])
        self.assertEqual(closed, [], "keep waiting for the second tab")
        closed, _ = cluster_shots(sh, c["pair_gap"], c["pair_settle"],
                                  T0 + c["pair_settle"] + 1,
                                  c["pair_settle_full"])
        self.assertEqual(len(closed), 1)

    def test_decides_at_is_reportable(self):
        from debrief_uploader.matcher import decides_at
        c = cfg()
        pair = one_cluster([0, 0.1], c=c)
        self.assertEqual(decides_at(pair, c["pair_settle"], c["pair_settle_full"]),
                         pair.t_last + c["pair_settle_full"])

    def test_a_third_shot_still_joins_before_the_window_closes(self):
        c = cfg()
        sh = [ShotRec("a", T0), ShotRec("b", T0 + 6), ShotRec("c", T0 + 20)]
        closed, _ = cluster_shots(sh, c["pair_gap"], c["pair_settle"],
                                  T0 + 20 + c["pair_settle_full"] + 1,
                                  c["pair_settle_full"])
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0]), 3)


class TestOlderUnpairedBattlesNeverHoldUpAMatch(unittest.TestCase):
    """A battle you never screenshotted must not hold up the ones after it.

    Reported from a real machine: an 08:56 game that had ended by 09:11 at
    the latest was still being treated as a rival claimant for 09:26
    screenshots that plainly belonged to the 09:12 game. Builds before 1.9
    still asked when the older battle's results could genuinely have been on
    screen (an early exit whose battle ran on); pairing no longer asks at
    all (Greg, 2026-11-16), so every one of these goes to the battle just
    played.
    """

    def battle(self, key, start_min, close_min, duration=900):
        """start/close in minutes from T0; header_dt is the game clock."""
        return ReplayRec(key=key, t_close=T0 + close_min * 60,
                         state=AWAITING_SHOTS, match_group="pvp",
                         header_dt=T0 + start_min * 60, duration=duration)

    def assertGoesTo(self, key, replays, offsets):
        d = match_cluster(one_cluster(offsets), replays, cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, key))

    def test_the_reported_case(self):
        # A: starts 0, 15-min limit -> cannot run past minute 15
        # B: starts 16, closes 29;  screenshots at 30
        self.assertGoesTo("B", [self.battle("A", 0, 12), self.battle("B", 16, 29)],
                          [30, 30.1])

    def test_early_exit_whose_battle_ran_on(self):
        # A starts at 16 and the player leaves at 17, but A runs until 31,
        # so its results could appear after B closed
        self.assertGoesTo("B", [self.battle("A", 16, 17), self.battle("B", 18, 29)],
                          [31, 31.1])

    def test_unknown_duration_or_skewed_clock(self):
        a = ReplayRec(key="A", t_close=T0 + 12 * 60, state=AWAITING_SHOTS,
                      match_group="pvp", header_dt=T0, duration=None)
        self.assertGoesTo("B", [a, self.battle("B", 16, 29)], [30, 30.1])
        # the -9h archive case: header_dt on a different clock
        a = ReplayRec(key="A", t_close=T0 + 12 * 60, state=AWAITING_SHOTS,
                      match_group="pvp", header_dt=T0 + 12 * 60 + 9 * 3600,
                      duration=900)
        self.assertGoesTo("B", [a, self.battle("B", 16, 29)], [30, 30.1])

    def test_a_skipped_battle_does_not_snowball(self):
        """Three battles, only the last screenshotted."""
        rs = [self.battle("A", 0, 13), self.battle("B", 16, 29),
              self.battle("C", 32, 45)]
        self.assertGoesTo("C", rs, [46, 46.1])


class BattleGroupTest(unittest.TestCase):
    """Bot rooms on any map count as training rooms (no scorecards to wait for)."""

    def meta(self, group, names):
        return {"matchGroup": group,
                "vehicles": [{"name": n, "relation": 2 if i else 0} for i, n in enumerate(names)]}

    def test_ocean_room(self):
        from debrief_uploader.matcher import battle_group, TRAINING
        self.assertEqual(battle_group(self.meta("training_room", ["me", "friend"])), TRAINING)

    def test_bot_room_on_a_pvp_map(self):
        from debrief_uploader.matcher import battle_group, TRAINING
        self.assertEqual(battle_group(self.meta("pvp", ["me", ":Hipper:", ":Scheer:"])), TRAINING)

    def test_real_pvp(self):
        from debrief_uploader.matcher import battle_group
        self.assertEqual(battle_group(self.meta("PvP", ["me"] + ["p%d" % i for i in range(17)])), "pvp")

    def test_human_only_room_still_reads_as_pvp(self):
        # indistinguishable from PvP in the header: keeps waiting for scorecards
        from debrief_uploader.matcher import battle_group
        self.assertEqual(battle_group(self.meta("pvp", ["me", "friend"])), "pvp")

    def test_coop_bots_stay_coop(self):
        from debrief_uploader.matcher import battle_group
        self.assertEqual(battle_group(self.meta("cooperative", ["me", ":Hipper:"])), "cooperative")

    def test_no_vehicles(self):
        from debrief_uploader.matcher import battle_group
        self.assertEqual(battle_group({"matchGroup": "pvp"}), "pvp")
        self.assertEqual(battle_group({}), "")


class BattleTypeTest(unittest.TestCase):
    """The battle-type setting (2026-11-13): every match group seen in real
    replays maps onto one of its five keys; training rooms and groups no one
    has seen yet are not covered, so a new game mode is never dropped."""

    def test_every_group_seen_in_real_replays(self):
        from debrief_uploader.matcher import battle_type, TRAINING
        seen = {"pvp": "pvp", "ranked": "ranked",
                "clan_elimination_cup": "clan", "clan": "clan",
                "cooperative": "coop", "pve": "events",
                "war_tales_event": "events", "arcade": "events",
                "event_alpha": "events", TRAINING: None}
        for group, kind in seen.items():
            self.assertEqual(battle_type(group), kind, group)

    def test_case_and_unknown_groups(self):
        from debrief_uploader.matcher import battle_type
        self.assertEqual(battle_type("PvP"), "pvp")
        self.assertEqual(battle_type("Cooperative"), "coop")
        self.assertIsNone(battle_type("brand_new_mode"))
        self.assertIsNone(battle_type(""))
        self.assertIsNone(battle_type(None))

    def test_every_key_is_a_setting(self):
        from debrief_uploader import config
        from debrief_uploader.matcher import battle_type
        for g in ("pvp", "ranked", "clan", "cooperative", "pve"):
            self.assertIn(battle_type(g), config.ALL_BATTLE_TYPES)


class FilteredBattleTest(unittest.TestCase):
    """A battle whose type is off never takes screenshots -- and the
    scorecards taken after it are its own, so they must not go to the older
    battle still waiting for scorecards."""

    def test_scorecards_after_a_filtered_battle_go_nowhere(self):
        from debrief_uploader.matcher import FILTERED
        rs = [replay("A", 10), replay("C", 13, group="cooperative", state=FILTERED)]
        d = match_cluster(one_cluster([14, 14.1]), rs, cfg())
        self.assertEqual(d.kind, NONE)
        self.assertIsNone(d.replay)
        self.assertIn("off in Settings", d.reason)

    def test_positive_control_without_the_filtered_battle(self):
        # the same shots with no co-op battle in between go to A: the refusal
        # above is the filtered battle's doing
        d = match_cluster(one_cluster([14, 14.1]), [replay("A", 10)], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "A"))

    def test_an_older_filtered_battle_does_not_block_a_later_one(self):
        from debrief_uploader.matcher import FILTERED
        rs = [replay("C", 5, group="cooperative", state=FILTERED), replay("A", 10)]
        d = match_cluster(one_cluster([11, 11.1]), rs, cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "A"))

    def test_a_filtered_battle_is_never_an_orphan(self):
        from debrief_uploader.matcher import FILTERED
        r = replay("C", 0, group="cooperative", state=FILTERED)
        self.assertFalse(is_orphan(r, [r], T0 + 3 * 3600, cfg()))


class LastBattleOwnsTheScorecardsTest(unittest.TestCase):
    """Screenshots taken after a battle are that battle's or no battle's.
    When it is not waiting for any, they are never handed past it to an
    older battle that is: with pairing never asking, that is how one wrong
    pairing dragged the next ones after it."""

    def test_a_tab_taken_after_the_last_battle_was_paired(self):
        rs = [replay("A", 10), replay("B", 13, state=UPLOADED)]
        d = match_cluster(one_cluster([14, 14.1]), rs, cfg())
        self.assertEqual((d.kind, d.replay), (NONE, None))
        self.assertIn("not waiting", d.reason)

    def test_positive_control_without_the_later_battle(self):
        d = match_cluster(one_cluster([14, 14.1]), [replay("A", 10)], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "A"))

    def test_after_a_training_room(self):
        # as the engine holds one: queued (or uploaded) the moment it is
        # identified, never waiting for scorecards
        from debrief_uploader.matcher import READY
        rs = [replay("A", 5), replay("T", 20, group="training_room", state=READY)]
        d = match_cluster(one_cluster([20.5, 20.6]), rs, cfg())
        self.assertEqual(d.kind, NONE)
        self.assertIn("training room", d.reason)

    def test_waits_while_the_last_battle_is_identified(self):
        from debrief_uploader.matcher import NEW
        a, b = replay("A", 10), replay("B", 13, state=NEW)
        d = match_cluster(one_cluster([14, 14.1]), [a, b], cfg())
        self.assertEqual(d.kind, NONE)
        self.assertIn("identified", d.reason)
        b.state = AWAITING_SHOTS                 # replay-meta answered
        d = match_cluster(one_cluster([14, 14.1]), [a, b], cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "B"))

    def test_an_older_paired_battle_does_not_block_a_later_one(self):
        rs = [replay("A", 5, state=UPLOADED), replay("B", 10)]
        d = match_cluster(one_cluster([11, 11.1]), rs, cfg())
        self.assertEqual((d.kind, d.replay.key), (AUTO, "B"))
