"""One-time pre-draft league info capture.

Given a league id (and/or draft id) this pulls everything that shapes draft
strategy from Sleeper, derives a human-readable summary, diffs the scoring
against Sleeper's base scoring, resolves *your* slot and pick numbers, and
persists a snapshot under ``home_dir()/leagues/<id>.json`` so draft day starts
instantly (and the report is available offline).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import DEFAULT_SEASON, NON_STARTING_SLOTS, SKILL_POSITIONS, SLOT_ELIGIBILITY, home_dir
from .models import DraftSettings, LeagueSettings, Manager, Pick
from .scoring import ScoringEngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sleeper base scoring (the values a freshly created Sleeper league starts with,
# receptions excluded because that is the league's declared PPR type)
# ---------------------------------------------------------------------------

SLEEPER_BASE_SCORING: dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_2pt": 2.0, "pass_int": -1.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0, "fum_rec_td": 6.0, "st_td": 6.0, "st_fum_rec": 1.0, "st_ff": 1.0,
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
    "fgmiss": -1.0, "xpm": 1.0, "xpmiss": -1.0,
    "def_td": 6.0, "sack": 1.0, "int": 2.0, "ff": 1.0, "fum_rec": 2.0, "safe": 2.0, "blk_kick": 2.0,
    "def_st_td": 6.0, "def_st_ff": 1.0, "def_st_fum_rec": 1.0,
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
    "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0, "pts_allow_35p": -4.0,
}

SCORING_LABELS: dict[str, str] = {
    "pass_yd": "Passing yards (per yd)", "pass_td": "Passing TD", "pass_int": "Interception thrown",
    "pass_2pt": "Passing 2-pt", "pass_cmp": "Completion", "pass_inc": "Incompletion", "pass_att": "Pass attempt",
    "pass_sack": "Sacked", "pass_fd": "Passing first down", "pass_td_40p": "40+ yd pass TD", "pass_td_50p": "50+ yd pass TD",
    "pass_cmp_40p": "40+ yd completion", "bonus_pass_yd_300": "300-yd passing game bonus",
    "bonus_pass_yd_400": "400-yd passing game bonus", "bonus_pass_cmp_25": "25+ completions bonus",
    "rush_yd": "Rushing yards (per yd)", "rush_td": "Rushing TD", "rush_2pt": "Rushing 2-pt", "rush_att": "Rush attempt",
    "rush_fd": "Rushing first down", "rush_40p": "40+ yd rush", "rush_td_40p": "40+ yd rush TD", "rush_td_50p": "50+ yd rush TD",
    "bonus_rush_yd_100": "100-yd rushing game bonus", "bonus_rush_yd_200": "200-yd rushing game bonus",
    "bonus_rush_att_20": "20+ carries bonus",
    "rec": "Reception", "rec_yd": "Receiving yards (per yd)", "rec_td": "Receiving TD", "rec_2pt": "Receiving 2-pt",
    "rec_tgt": "Target", "rec_fd": "Receiving first down", "rec_40p": "40+ yd reception", "rec_td_40p": "40+ yd rec TD",
    "rec_td_50p": "50+ yd rec TD", "bonus_rec_yd_100": "100-yd receiving game bonus", "bonus_rec_yd_200": "200-yd receiving game bonus",
    "bonus_rec_rb": "RB reception bonus", "bonus_rec_wr": "WR reception bonus", "bonus_rec_te": "TE reception bonus (TE premium)",
    "bonus_rush_rec_yd_100": "100 rush+rec yd bonus", "bonus_rush_rec_yd_200": "200 rush+rec yd bonus",
    "fum": "Fumble", "fum_lost": "Fumble lost", "fum_rec_td": "Fumble recovery TD", "st_td": "Special teams TD",
    "st_ff": "ST forced fumble", "st_fum_rec": "ST fumble recovery", "kr_yd": "Kick return yards", "pr_yd": "Punt return yards",
    "fgm": "FG made", "fga": "FG attempt", "fgmiss": "FG missed", "fgm_yds": "FG yards", "fgm_yds_over_30": "FG yards over 30",
    "fgm_0_19": "FG 0-19", "fgm_20_29": "FG 20-29", "fgm_30_39": "FG 30-39", "fgm_40_49": "FG 40-49", "fgm_50p": "FG 50+",
    "fgm_50_59": "FG 50-59", "fgm_60p": "FG 60+", "fgmiss_0_19": "FG miss 0-19", "fgmiss_20_29": "FG miss 20-29",
    "fgmiss_30_39": "FG miss 30-39", "fgmiss_40_49": "FG miss 40-49", "fgmiss_50p": "FG miss 50+",
    "xpm": "XP made", "xpa": "XP attempt", "xpmiss": "XP missed",
    "def_td": "Defensive TD", "sack": "Sack", "int": "Interception", "ff": "Forced fumble", "fum_rec": "Fumble recovery",
    "safe": "Safety", "blk_kick": "Blocked kick", "def_2pt": "Defensive 2-pt return", "def_st_td": "DEF/ST TD",
    "def_st_ff": "DEF/ST forced fumble", "def_st_fum_rec": "DEF/ST fumble recovery", "def_forced_punts": "Forced punt",
    "def_pass_def": "Pass defended", "def_4_and_stop": "4th down stop", "def_3_and_out": "3-and-out",
    "pts_allow": "Points allowed (per pt)", "pts_allow_0": "0 pts allowed", "pts_allow_1_6": "1-6 pts allowed",
    "pts_allow_7_13": "7-13 pts allowed", "pts_allow_14_20": "14-20 pts allowed", "pts_allow_21_27": "21-27 pts allowed",
    "pts_allow_28_34": "28-34 pts allowed", "pts_allow_35p": "35+ pts allowed",
    "yds_allow": "Yards allowed (per yd)", "yds_allow_0_100": "<100 yds allowed", "yds_allow_100_199": "100-199 yds allowed",
    "yds_allow_200_299": "200-299 yds allowed", "yds_allow_300_349": "300-349 yds allowed", "yds_allow_350_399": "350-399 yds allowed",
    "yds_allow_400_449": "400-449 yds allowed", "yds_allow_450_499": "450-499 yds allowed", "yds_allow_500_549": "500-549 yds allowed",
    "yds_allow_550p": "550+ yds allowed",
    "idp_tkl": "IDP tackle", "idp_tkl_solo": "IDP solo tackle", "idp_tkl_ast": "IDP assisted tackle", "idp_sack": "IDP sack",
    "idp_int": "IDP interception", "idp_ff": "IDP forced fumble", "idp_fum_rec": "IDP fumble recovery", "idp_pass_def": "IDP pass defended",
    "idp_qb_hit": "IDP QB hit", "idp_tkl_loss": "IDP tackle for loss", "idp_safe": "IDP safety", "idp_blk_kick": "IDP blocked kick",
    "idp_def_td": "IDP defensive TD",
}


def label_for(key: str) -> str:
    return SCORING_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Scoring diff
# ---------------------------------------------------------------------------


@dataclass
class ScoringDelta:
    key: str
    label: str
    base: float | None      # None = not scored in Sleeper's base scoring
    league: float | None    # None = removed / zero in this league

    @property
    def kind(self) -> str:
        if self.base is None:
            return "added"
        if self.league is None:
            return "removed"
        return "changed"


@dataclass
class ScoringDiff:
    scoring_type: str                   # "ppr" | "half_ppr" | "std"
    rec_points: float
    deltas: list[ScoringDelta] = field(default_factory=list)

    @property
    def changed(self) -> list[ScoringDelta]:
        return [d for d in self.deltas if d.kind == "changed"]

    @property
    def added(self) -> list[ScoringDelta]:
        return [d for d in self.deltas if d.kind == "added"]

    @property
    def removed(self) -> list[ScoringDelta]:
        return [d for d in self.deltas if d.kind == "removed"]

    @property
    def is_base(self) -> bool:
        return not self.deltas

    def to_dict(self) -> dict:
        return {"scoring_type": self.scoring_type, "rec_points": self.rec_points,
                "deltas": [asdict(d) for d in self.deltas]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScoringDiff":
        return cls(scoring_type=d.get("scoring_type", "std"), rec_points=float(d.get("rec_points", 0.0)),
                   deltas=[ScoringDelta(**x) for x in d.get("deltas", [])])


def _nz(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if abs(f) < 1e-12 else f


def scoring_diff(scoring_settings: Mapping[str, Any]) -> ScoringDiff:
    """Compare a league's ``scoring_settings`` with :data:`SLEEPER_BASE_SCORING`.

    Zero-valued keys count as "not scored". Receptions are reported through
    ``scoring_type`` rather than as a delta.
    """
    rec = float(scoring_settings.get("rec", 0.0) or 0.0)
    stype = "ppr" if rec >= 0.75 else "half_ppr" if rec >= 0.25 else "std"
    deltas: list[ScoringDelta] = []
    keys = set(SLEEPER_BASE_SCORING) | {k for k, v in scoring_settings.items() if _nz(v) is not None}
    for k in sorted(keys, key=lambda x: (x not in SLEEPER_BASE_SCORING, list(SCORING_LABELS).index(x) if x in SCORING_LABELS else 999, x)):
        if k == "rec":
            continue
        base = _nz(SLEEPER_BASE_SCORING.get(k))
        league = _nz(scoring_settings.get(k))
        if base is None and league is None:
            continue
        if base is not None and league is not None and abs(base - league) < 1e-9:
            continue
        deltas.append(ScoringDelta(key=k, label=label_for(k), base=base, league=league))
    return ScoringDiff(scoring_type=stype, rec_points=rec, deltas=deltas)


# ---------------------------------------------------------------------------
# Strategy flags
# ---------------------------------------------------------------------------


def strategy_flags(league: LeagueSettings | None, draft: DraftSettings | None) -> list[str]:
    """Short, plain-English facts that change how you should draft."""
    flags: list[str] = []
    if league is None:
        return flags
    s = league.scoring_settings
    g = lambda k: float(s.get(k, 0.0) or 0.0)  # noqa: E731
    flags.append({"ppr": "Full PPR (1.0 per reception)", "half_ppr": "Half PPR (0.5 per reception)",
                  "std": "Standard scoring (no points per reception)"}[league.scoring_type])
    if abs(g("pass_td") - 4.0) > 1e-9:
        flags.append(f"{g('pass_td'):g}-pt passing TD (QBs {'worth more' if g('pass_td') > 4 else 'worth less'} than usual)")
    if abs(g("pass_int") + 1.0) > 1e-9:
        flags.append(f"Interceptions {g('pass_int'):g} (turnover-prone QBs {'punished harder' if g('pass_int') < -1 else 'punished less'})")
    if g("pass_yd") and abs(g("pass_yd") - 0.04) > 1e-9:
        flags.append(f"Passing yards at {1 / g('pass_yd'):.0f} yd/pt")
    if g("bonus_rec_te") > 0:
        flags.append(f"TE premium: +{g('bonus_rec_te'):g} per TE reception (elite TEs worth more)")
    if g("bonus_rec_rb") > 0 or g("bonus_rec_wr") > 0:
        flags.append("Position-specific reception bonuses")
    if any(g(k) for k in ("pass_fd", "rush_fd", "rec_fd")):
        flags.append("First-down bonuses (volume players gain)")
    if any(g(k) for k in s if k.startswith("bonus_") and k not in ("bonus_rec_te", "bonus_rec_rb", "bonus_rec_wr")):
        flags.append("Yardage / big-game bonuses in play (boom players gain)")
    if g("fum_lost") and abs(g("fum_lost") + 2.0) > 1e-9:
        flags.append(f"Fumbles lost {g('fum_lost'):g}")
    if league.is_superflex:
        flags.append("SUPERFLEX / 2-QB: quarterbacks are scarce, plan on 2-3 QBs")
    slots = league.roster_positions
    if "K" not in slots:
        flags.append("No kicker slot")
    if "DEF" not in slots:
        flags.append("No team-defense slot")
    if any(x in slots for x in ("DL", "LB", "DB", "IDP_FLEX")):
        flags.append("IDP slots present (individual defenders are not projected by this tool)")
    n_flex = sum(1 for x in slots if x in ("FLEX", "WRRB_FLEX", "REC_FLEX"))
    if n_flex >= 2:
        flags.append(f"{n_flex} flex slots: RB/WR depth matters more")
    if league.roster_positions.count("TE") >= 2:
        flags.append("2 TE starters: TEs are scarce")
    bench = league.bench_slots
    if bench <= 4:
        flags.append(f"Short bench ({bench}): prioritise starters, fewer speculative picks")
    elif bench >= 8:
        flags.append(f"Deep bench ({bench}): late-round upside stashes are cheap")
    ir = slots.count("IR")
    if ir:
        flags.append(f"{ir} IR slot{'s' if ir > 1 else ''}: injured stars are cheaper to hold")
    if slots.count("TAXI"):
        flags.append(f"{slots.count('TAXI')} taxi slots (rookie stashes)")
    if int(league.settings.get("best_ball", 0) or 0):
        flags.append("Best ball: no weekly lineup setting, draft for depth and upside")
    if int(league.settings.get("max_keepers", 0) or 0) > 1 or league.settings.get("type", 0) in (1, 2):
        flags.append("Keeper / dynasty league: age and long-term value matter")
    if draft is not None:
        if draft.reversal_round:
            flags.append(f"Round-{draft.reversal_round} reversal draft order")
        if draft.type == "linear":
            flags.append("Linear (non-snake) draft: the top slots get every round's first picks")
        if draft.type == "auction":
            flags.append("Auction draft: this tool's pick-order advice does not apply")
        if draft.pick_timer:
            flags.append(f"{draft.pick_timer}-second pick clock at capture time (commissioners can change it; the live value is shown during the draft)")
    return flags


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def leagues_dir() -> Path:
    return home_dir() / "leagues"


@dataclass
class LeagueSnapshot:
    captured_at: float
    season: int
    league: LeagueSettings | None
    draft: DraftSettings | None
    managers: dict[str, Manager] = field(default_factory=dict)
    keepers: list[Pick] = field(default_factory=list)
    picks_made: int = 0
    my_user_id: str | None = None
    my_slot: int | None = None
    my_roster_id: int | None = None
    my_picks: list[int] = field(default_factory=list)
    diff: ScoringDiff | None = None
    flags: list[str] = field(default_factory=list)
    nfl_state: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)         # league, draft, users, rosters, traded_picks, picks, drafts

    # -- identity ---------------------------------------------------------------
    @property
    def league_id(self) -> str | None:
        return self.league.league_id if self.league else (self.draft.league_id if self.draft else None)

    @property
    def draft_id(self) -> str | None:
        return self.draft.draft_id if self.draft else (self.league.draft_id if self.league else None)

    @property
    def name(self) -> str:
        if self.league:
            return self.league.name
        if self.draft:
            return self.draft.metadata.get("name") or f"Draft {self.draft.draft_id}"
        return "Unknown league"

    def manager_for_slot(self, slot: int) -> Manager | None:
        for m in self.managers.values():
            if m.slot == slot:
                return m
        return None

    def roster_players(self, roster_id: int) -> list[str]:
        for r in self.raw.get("rosters") or []:
            if int(r.get("roster_id") or -1) == roster_id:
                return [str(p) for p in (r.get("players") or [])]
        return []

    # -- persistence -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "captured_at": self.captured_at,
            "season": self.season,
            "my_user_id": self.my_user_id,
            "my_slot": self.my_slot,
            "my_roster_id": self.my_roster_id,
            "my_picks": list(self.my_picks),
            "picks_made": self.picks_made,
            "diff": self.diff.to_dict() if self.diff else None,
            "flags": list(self.flags),
            "nfl_state": self.nfl_state,
            "raw": self.raw,
        }

    def save(self, path: Path | None = None) -> Path:
        d = leagues_dir()
        d.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=1)
        targets = [path] if path else []
        if not path:
            if self.league_id:
                targets.append(d / f"{self.league_id}.json")
            if self.draft_id and self.draft_id != self.league_id:
                targets.append(d / f"{self.draft_id}.json")
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(t)
        return targets[0]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LeagueSnapshot":
        from .sleeper.parsing import parse_draft, parse_league, parse_managers, parse_picks

        raw = dict(d.get("raw") or {})
        league = parse_league(raw["league"]) if raw.get("league") else None
        draft = parse_draft(raw["draft"], raw.get("traded_picks")) if raw.get("draft") else None
        managers = parse_managers(raw.get("users") or [], draft, raw.get("rosters")) if draft else {}
        picks = parse_picks(raw.get("picks") or [])
        snap = cls(
            captured_at=float(d.get("captured_at", 0.0)),
            season=int(d.get("season", DEFAULT_SEASON)),
            league=league, draft=draft, managers=managers,
            keepers=[p for p in picks if p.is_keeper],
            picks_made=int(d.get("picks_made", len(picks))),
            my_user_id=d.get("my_user_id"), my_slot=d.get("my_slot"), my_roster_id=d.get("my_roster_id"),
            my_picks=[int(x) for x in d.get("my_picks", [])],
            diff=ScoringDiff.from_dict(d["diff"]) if d.get("diff") else (scoring_diff(league.scoring_settings) if league else None),
            flags=list(d.get("flags") or []), nfl_state=dict(d.get("nfl_state") or {}), raw=raw,
        )
        return snap

    @classmethod
    def load(cls, league_or_draft_id: str, max_age_hours: float | None = None) -> "LeagueSnapshot | None":
        p = leagues_dir() / f"{league_or_draft_id}.json"
        if not p.exists():
            return None
        if max_age_hours is not None and (time.time() - p.stat().st_mtime) > max_age_hours * 3600:
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            log.warning("could not load snapshot %s: %s", p, e)
            return None

    # -- report ------------------------------------------------------------------
    def _fmt_time(self, ms: int | None) -> str:
        if not ms:
            return "not scheduled"
        try:
            return datetime.fromtimestamp(ms / 1000).strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
        except (OverflowError, OSError, ValueError):
            return str(ms)

    def slot_summary(self) -> str:
        if not self.league:
            return "unknown"
        counts: dict[str, int] = {}
        for s in self.league.roster_positions:
            counts[s] = counts.get(s, 0) + 1
        starters = [f"{k}x{v}" if v > 1 else k for k, v in counts.items() if k not in NON_STARTING_SLOTS]
        extras = [f"{k}x{v}" if v > 1 else k for k, v in counts.items() if k in NON_STARTING_SLOTS]
        return ", ".join(starters) + (" | " + ", ".join(extras) if extras else "")

    def report(self, width: int | None = None) -> RenderableType:
        parts: list[RenderableType] = []
        lg, dr = self.league, self.draft
        # -- league panel
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", justify="right")
        t.add_column()
        if lg:
            eng = ScoringEngine(lg.scoring_settings)
            t.add_row("League", f"{lg.name}  (id {lg.league_id})")
            t.add_row("Season / status", f"{lg.season} / {lg.status or '?'}")
            t.add_row("Teams", str(lg.total_rosters))
            t.add_row("Scoring", eng.describe())
            t.add_row("Lineup", self.slot_summary())
            t.add_row("Roster size", f"{lg.roster_size} ({len(lg.starting_slots)} starters, {lg.bench_slots} bench)")
            st = lg.settings
            misc = []
            if st.get("playoff_teams"):
                misc.append(f"{st['playoff_teams']} playoff teams from week {st.get('playoff_week_start', '?')}")
            if st.get("trade_deadline"):
                misc.append(f"trade deadline week {st['trade_deadline']}")
            if st.get("max_keepers"):
                misc.append(f"max keepers {st['max_keepers']}")
            if st.get("waiver_type") is not None:
                misc.append({0: "rolling waivers", 1: "reverse standings waivers", 2: "FAAB waivers"}.get(int(st["waiver_type"]), f"waiver type {st['waiver_type']}"))
            if st.get("best_ball"):
                misc.append("best ball")
            if misc:
                t.add_row("Settings", "; ".join(misc))
        else:
            t.add_row("League", "(no league linked to this draft)")
        parts.append(Panel(t, title="[bold]League", border_style="cyan"))
        # -- scoring diff
        if self.diff:
            dt = Table(title=None, show_header=True, header_style="bold", box=None, pad_edge=False)
            dt.add_column("Scoring rule")
            dt.add_column("Sleeper base", justify="right")
            dt.add_column("This league", justify="right", style="bold yellow")
            dt.add_column("Key", style="dim")
            rec_lbl = {"ppr": "1.0 (PPR)", "half_ppr": "0.5 (half PPR)", "std": "0 (standard)"}[self.diff.scoring_type]
            dt.add_row("Reception", "league choice", f"{self.diff.rec_points:g}  {rec_lbl.split(' ', 1)[1]}", "rec")
            for d in self.diff.deltas:
                base = "—" if d.base is None else f"{d.base:g}"
                league = "removed" if d.league is None else f"{d.league:g}"
                dt.add_row(d.label, base, league, d.key)
            if self.diff.is_base:
                dt.add_row("[green]Everything else matches Sleeper base scoring[/green]", "", "", "")
            else:
                dt.add_row(f"[dim]{len(self.diff.deltas)} rule(s) differ from Sleeper base; everything else is base[/dim]", "", "", "")
            parts.append(Panel(dt, title="[bold]Scoring vs Sleeper base", border_style="yellow"))
        # -- flags
        if self.flags:
            parts.append(Panel(Text("\n".join(f"• {f}" for f in self.flags)), title="[bold]What changes strategy here", border_style="magenta"))
        # -- draft panel
        t2 = Table.grid(padding=(0, 2))
        t2.add_column(style="bold cyan", justify="right")
        t2.add_column()
        if dr:
            kind = dr.type + (f", round-{dr.reversal_round} reversal" if dr.reversal_round else "")
            t2.add_row("Draft", f"{kind}  (id {dr.draft_id})")
            t2.add_row("Status", dr.status)
            t2.add_row("Rounds x teams", f"{dr.rounds} x {dr.teams} = {dr.total_picks} picks")
            t2.add_row("Pick clock", f"{dr.pick_timer} s" if dr.pick_timer else "none")
            t2.add_row("Start", self._fmt_time(dr.start_time))
            if self.keepers:
                t2.add_row("Keepers on board", str(len(self.keepers)))
            if dr.traded_picks:
                t2.add_row("Traded picks", str(len(dr.traded_picks)))
            if self.picks_made:
                t2.add_row("Picks made", str(self.picks_made))
            if self.my_slot is not None:
                mine = ", ".join(str(p) for p in self.my_picks[:6]) + (" …" if len(self.my_picks) > 6 else "")
                who = self.managers.get(self.my_user_id or "")
                label = (who.team_name or who.display_name) if who else "you"
                t2.add_row("You", f"slot {self.my_slot} ({label}, roster {self.my_roster_id}) — picks {mine}")
            elif self.my_user_id is not None and not dr.draft_order:
                who = self.managers.get(self.my_user_id)
                label = (who.team_name or who.display_name) if who else self.my_user_id
                t2.add_row("You", Text(f"{label} — draft order not set yet; your slot and pick numbers "
                                       "resolve when the commissioner sets the order / the draft starts",
                                       style="yellow"))
            else:
                t2.add_row("You", "[red]not identified — pass --username / --user-id / --slot[/red]")
        else:
            t2.add_row("Draft", "(none found)")
        parts.append(Panel(t2, title="[bold]Draft", border_style="green"))
        # -- draft order
        if dr and (dr.draft_order or dr.slot_to_roster_id or self.managers):
            ot = Table(box=None, header_style="bold", pad_edge=False)
            ot.add_column("Slot", justify="right")
            ot.add_column("Manager")
            ot.add_column("Team name")
            ot.add_column("Roster", justify="right")
            ot.add_column("Picks (first 5)")
            ot.add_column("Roster now", justify="right")
            for slot in range(1, dr.teams + 1):
                m = self.manager_for_slot(slot)
                rid = dr.original_roster_for_slot(slot)
                picks = dr.picks_for_slot(slot)[:5]
                style = "bold green" if slot == self.my_slot else ""
                n_roster = len(self.roster_players(rid)) if rid is not None else 0
                ot.add_row(
                    Text(str(slot) + (" ◀ you" if slot == self.my_slot else ""), style=style),
                    Text(m.display_name if m else "?", style=style),
                    Text((m.team_name or "") if m else "", style=style),
                    Text(str(rid) if rid is not None else "?", style=style),
                    Text(", ".join(str(p) for p in picks), style=style),
                    Text(str(n_roster) if n_roster else "-", style=style),
                )
            parts.append(Panel(ot, title="[bold]Draft order", border_style="blue"))
        stamp = datetime.fromtimestamp(self.captured_at).strftime("%Y-%m-%d %H:%M")
        parts.append(Text(f"captured {stamp} • saved under {leagues_dir()}", style="dim"))
        return Group(*parts)

    def to_text(self, width: int = 120) -> str:
        console = Console(record=True, width=width, force_terminal=False, color_system=None)
        console.print(self.report())
        return console.export_text()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _pick_draft(drafts: list[dict], draft_id: str | None, season: int) -> dict | None:
    if not drafts:
        return None
    if draft_id:
        for d in drafts:
            if str(d.get("draft_id")) == str(draft_id):
                return d
    same_season = [d for d in drafts if str(d.get("season")) == str(season)] or drafts
    order = {"drafting": 0, "paused": 1, "pre_draft": 2, "complete": 3}
    same_season.sort(key=lambda d: (order.get(d.get("status"), 9), -(d.get("created") or 0)))
    return same_season[0]


def resolve_identity(draft: DraftSettings, managers: dict[str, Manager], users_raw: list[dict] | None = None, *,
                     username: str | None = None, user_id: str | None = None,
                     slot: int | None = None) -> tuple[str | None, int | None]:
    """``(my_user_id, my_slot)`` for one manager of a captured Sleeper draft (pure; shared by the capture and
    the web server, which resolves "me" per request on an identity-free capture).

    Raises :class:`ValueError` (naming the managers) when ``username`` / ``user_id`` matches nobody in a
    published draft order, or nobody among the league's managers when the order is not set yet. An
    identity that matches a manager whose slot is unknown keeps the id and leaves the slot ``None``.
    """
    from .sleeper.parsing import resolve_my_slot

    # ``users`` lets a Sleeper *username* that differs from the display name resolve too
    my_uid, my_slot = resolve_my_slot(draft, managers, username=username, user_id=user_id, slot=slot, users=users_raw)
    if (username or user_id) and my_slot is None:
        names = ", ".join(sorted(f"{m.display_name}" + (f" ({m.team_name})" if m.team_name else "")
                                 for m in managers.values()))
        if draft.draft_order:
            # the order is published and the user is not in it: a typo or the wrong league
            raise ValueError(f"could not find {username or user_id!r} in the draft order. Managers: {names}")
        if my_uid is None and managers:
            # no order yet, but the league's members are known and the user is not one of them
            raise ValueError(f"could not find {username or user_id!r} among the league's managers: {names}")
        # order not set yet (typical before the draft starts): keep the identity, resolve the slot later
        log.info("draft order not set yet for draft %s; slot for %r resolves when the draft starts",
                 draft.draft_id, username or user_id)
    return my_uid, my_slot


async def capture_league(client, league_id: str | None = None, draft_id: str | None = None, *,
                         username: str | None = None, user_id: str | None = None, slot: int | None = None,
                         season: int | None = None, save: bool = True) -> LeagueSnapshot:
    """Fetch and persist everything about a league/draft that matters before the draft.

    ``client`` is a :class:`draftadvisor.sleeper.client.SleeperClient` (or anything with the same
    async ``get_*`` methods). Either ``league_id`` or ``draft_id`` is required.
    """
    from .sleeper.client import SleeperAPIError
    from .sleeper.parsing import parse_draft, parse_league, parse_managers, parse_picks

    if not league_id and not draft_id:
        raise ValueError("capture_league needs a league_id or a draft_id")
    season = season or DEFAULT_SEASON
    raw: dict[str, Any] = {}
    try:
        raw["state"] = await client.get_state()
    except Exception as e:  # noqa: BLE001
        log.info("nfl state unavailable: %s", e)
        raw["state"] = {}

    async def optional(name: str, coro, default):
        """Optional endpoints degrade like the poller's ``_fetch_optional``: a 404 *or* a 5xx / 429
        that exhausted the client's retries must not discard the whole capture."""
        try:
            return await coro
        except SleeperAPIError as e:
            log.warning("%s unavailable (%s); continuing without it", name, e)
            return default

    draft_raw: dict | None = None
    if draft_id:
        draft_raw = await client.get_draft(draft_id)
        league_id = league_id or draft_raw.get("league_id")
    league_raw: dict | None = None
    users_raw: list[dict] = []
    rosters_raw: list[dict] = []
    drafts_raw: list[dict] = []
    if league_id:
        league_raw = await client.get_league(league_id)
        users_raw = await optional("league users", client.get_league_users(league_id), []) or []
        rosters_raw = await optional("league rosters", client.get_league_rosters(league_id), []) or []
        drafts_raw = await optional("league drafts", client.get_league_drafts(league_id), []) or []
        if draft_raw is None:
            chosen = _pick_draft(drafts_raw, draft_id or league_raw.get("draft_id"), int(league_raw.get("season") or season))
            if chosen is not None:
                draft_id = str(chosen.get("draft_id"))
                draft_raw = await client.get_draft(draft_id)
            elif league_raw.get("draft_id"):
                draft_id = str(league_raw["draft_id"])
                draft_raw = await client.get_draft(draft_id)
    traded_raw: list[dict] = []
    picks_raw: list[dict] = []
    if draft_raw is not None and draft_id:
        traded_raw = await optional("traded picks", client.get_traded_picks(draft_id), []) or []
        picks_raw = await optional("draft picks", client.get_draft_picks(draft_id), []) or []
    raw.update({"league": league_raw, "draft": draft_raw, "users": users_raw, "rosters": rosters_raw,
                "traded_picks": traded_raw, "picks": picks_raw, "drafts": drafts_raw})

    league = parse_league(league_raw) if league_raw else None
    draft = parse_draft(draft_raw, traded_raw) if draft_raw else None
    managers = parse_managers(users_raw, draft, rosters_raw) if draft else {}
    picks = parse_picks(picks_raw)
    my_uid, my_slot = (None, None)
    if draft:
        my_uid, my_slot = resolve_identity(draft, managers, users_raw, username=username, user_id=user_id, slot=slot)
    my_roster = draft.original_roster_for_slot(my_slot) if (draft and my_slot is not None) else None
    my_picks = draft.picks_for_slot(my_slot) if (draft and my_slot is not None) else []
    snap = LeagueSnapshot(
        captured_at=time.time(),
        season=int((league_raw or {}).get("season") or (draft_raw or {}).get("season") or season),
        league=league, draft=draft, managers=managers,
        keepers=[p for p in picks if p.is_keeper], picks_made=len(picks),
        my_user_id=my_uid, my_slot=my_slot, my_roster_id=my_roster, my_picks=my_picks,
        diff=scoring_diff(league.scoring_settings) if league else None,
        flags=strategy_flags(league, draft),
        nfl_state=raw.get("state") or {}, raw=raw,
    )
    if save:
        snap.save()
    return snap


def print_snapshot(snap: LeagueSnapshot, console: Console | None = None) -> None:
    (console or Console()).print(snap.report())
