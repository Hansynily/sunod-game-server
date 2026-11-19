"""Loops: what carries over from one run to the next.

The story: the disaster hits at the end of every run, the player is sent back in time to
being a child, and keeps their memories. Each run is one loop. Between loops the player
keeps:

- remembered skills: chosen inside the going-back cutscene from the skills they owned at
  the end of the last loop; they start the next loop already in their kind's slot,
- keepsakes: 12 from townspeople (passing that person's quest unlocks it in the stall; it's
  bought with stars) and 3 from town events (free). Owned keepsakes stay forever, but only
  the ones in the player's pockets come along on a loop (2 pockets, more can be bought).
  A keepsake can be upgraded once (Lv2, "friendship") after helping its giver again, their
  preferred way, in a later loop,
- the stall's other goods, all bought with stars (the only currency): pockets, one-use run
  helpers and cosmetics,
- a journal of every loop: the threat, the Holland code and career it ended on, what was
  carried, and what changed since the loop before.

Nothing here helps one RIASEC letter more than another, touches a quest's scored stars or
item score, retries or auto-passes anything. Loop 1 is the baseline; every finished run in
`session_runs` is stamped with its loop, threat, remembered skills, carried keepsakes and
the helpers used, so later loops can be told apart from the first in the data.

One document per player in `player_loop_meta`. Finishing a run closes the loop
(`complete_loop`, idempotent); "Go Back" starts the next one (`begin_loop`); New Game wipes
everything (`reset`).

The stall works offline. The game applies a purchase, a helper use or an outfit change on
the spot and saves it as a marker in the run's progress ids, like bonus stars:
"sb:<item>" (buy), "su:<helper>" (use), "se:<n>:<slot>:<item>" (wear / show). The markers
reach the server with the run save and are applied here in order, each exactly once
(`stall_applied`), with the same checks as before (unlocked, owned, one per loop, enough
stars). Stars spent this way during a loop are kept in `stall_spent_loop` and taken off the
bank when the loop ends, so the bank only changes at a loop's end.
"""

import random
from datetime import datetime
from typing import Any

THREATS = ["typhoon", "flood", "earthquake", "fire"]

# ── Keepsakes ────────────────────────────────────────────────────────────────
# Giver keepsakes: id -> (question code of the quest that unlocks it, price, Lv2 price).
# Effects live in the game; none touch the quest timer's scored stars or scoring, and none
# favour a RIASEC letter. A full run earns up to 240 stars (48 quests x 5).
MEMENTOS = {
    "compass": ("I6", 40, 90),     # Nico: hint arrow faster | Lv2 also points to today's event
    "sneakers": ("E2", 30, 80),    # Jun: walk faster | Lv2 faster still + dust trail
    "ribbon": ("A6", 50, 100),     # Paolo: +1 skill pick per new town | Lv2 +2
    "hammer": ("R2", 60, 110),     # Mang Ben: +1 press per quest | Lv2 +2
    "lantern": ("S8", 60, 110),    # Lola Ising: +1 skill pick each night | Lv2 4 night cards
    "locket": ("C5", 70, 120),     # Aling Nena: remember 4 skills | Lv2 remembered start +2 XP
    "notebook": ("I8", 50, 100),   # Prof. Lim: prompt shows the prep skill | Lv2 + preferred
    "jacket": ("E6", 40, 90),      # Ate Joan: preferred-way bonus x2 | Lv2 x3
    "whistle": ("R5", 50, 100),    # Mang Erning: +1 request each morning | Lv2 +2
    "nametag": ("S1", 90, 120),    # Ms. Rivera: forgotten skills keep 1/2 XP | Lv2 3/4
    "presspass": ("A5", 70, 110),  # Kevin: +1 XP to the night-pick skill | Lv2 +2
    "coinpurse": ("C7", 40, 90),   # Ms. Uy: +1 bank star per quest passed | Lv2 +2
}
# Event keepsakes: free, never sold, count toward pockets. Earned from the town events the
# player joined (progress markers "ev:<stage>:<event id>" in the run's completed ids).
EVENT_MEMENTOS = ["banderitas", "shell", "luckycoin"]
EVENT_LETTERS = {
    "bayanihan": "R", "sky_talk": "I", "mural_day": "A",
    "health_day": "S", "assembly": "E", "census": "C",
}
ALL_MEMENTOS = list(MEMENTOS) + EVENT_MEMENTOS

EQUIP_SLOTS = ("outfit", "hat", "trail", "pet")
# school uniforms: free, never sold; the game unlocks them by reaching that school in the loop
STAGE_UNIFORMS = ("outfit_elem", "outfit_hs", "outfit_college")
BASE_MEMORY_SLOTS = 3
BASE_POCKETS = 2
MAX_POCKETS = 4

# ── The stall's other goods ──────────────────────────────────────────────────
POCKET_PRICES = {3: 100, 4: 200}        # the 3rd pocket, then the 4th
BOARD_PRICE = 120                       # the plaza board's Requests tab + the Journal (kept forever)
HELPERS = {"reroll": 15, "luckycard": 25, "studynotes": 30}   # one-use, max 1 each per loop
COSMETICS = {
    # for you: outfits (palette swaps) and hats (an overlay that follows the frames)
    "outfit_fiesta": ("outfit", 60),    # barong (boy) / Filipiniana (girl)
    "outfit_raincoat": ("outfit", 40),  # threat gear
    "outfit_farmer": ("outfit", 40),    # camisa de chino + rolled pants
    "hat_salakot": ("hat", 30),
    "hat_hardhat": ("hat", 30),         # threat gear
    "hat_banig": ("hat", 30),
    "hat_sampaguita": ("hat", 20),      # flower crown
    "hat_cap": ("hat", 20),
    # trail
    "trail_petals": ("trail", 30),
    "trail_leaves": ("trail", 30),
    "trail_sparkles": ("trail", 40),
    "trail_bubbles": ("trail", 40),
    # the town cat follows you
    "pet_cat": ("pet", 60),
    # home add-ons (several can be on at once)
    "home_garden": ("home", 40),
    "home_capiz": ("home", 60),
    "home_hammock": ("home", 50),
    "home_sarisari": ("home", 80),
}

TRAITS = {"R": "Realistic", "I": "Investigative", "A": "Artistic",
          "S": "Social", "E": "Enterprising", "C": "Conventional"}

# skill id (the game's skillName) -> (English name, RIASEC letter). For the loop journal's
# reasons ("You remembered TEACH"), never for scoring.
SKILLS = {
    "AYOS": ("REPAIR", "R"), "TULAK": ("PUSH", "R"), "BUO": ("BUILD", "R"),
    "BALAK": ("PLAN", "I"), "SURI": ("SURVEY", "I"), "LUTAS": ("SOLVE", "I"),
    "PINTA": ("PAINT", "A"), "HULMA": ("BARRIER", "A"), "KATHA": ("CRAFT", "A"),
    "GABAY": ("GUIDE", "S"), "LUNAS": ("HEAL", "S"), "TURO": ("TEACH", "S"),
    "SENSUS": ("MAP", "E"), "UTOS": ("COMMAND", "E"), "HIMOK": ("PERSUADE", "E"),
    "SIKAT": ("CHARM", "E"), "TAWAG": ("CALL", "E"),
    "SILIP": ("INSPECT", "C"), "IPON": ("COLLECT", "C"), "DAAN": ("DIRECT", "C"),
}


class LoopError(Exception):
    """A request that doesn't fit the player's current loop (shown to the player)."""


# ── small helpers ────────────────────────────────────────────────────────────

def _list(doc: dict[str, Any], key: str) -> list:
    return list(doc.get(key) or [])


def pockets(doc: dict[str, Any]) -> int:
    return max(BASE_POCKETS, min(MAX_POCKETS, int(doc.get("pockets") or BASE_POCKETS)))


def carried(doc: dict[str, Any]) -> list[str]:
    """Keepsakes in the player's pockets this loop. Records from before pockets existed
    (no `carried_mementos`) carry everything they own until their next Go Back."""
    owned = _list(doc, "owned_mementos")
    if doc.get("carried_mementos") is None:
        return owned
    return [m for m in _list(doc, "carried_mementos") if m in owned]


def is_carried(doc: dict[str, Any], memento_id: str) -> bool:
    return memento_id in carried(doc)


def memento_level(doc: dict[str, Any], memento_id: str) -> int:
    return 2 if memento_id in _list(doc, "lv2_owned") else 1


def memory_slots(carried_ids: list[str]) -> int:
    return BASE_MEMORY_SLOTS + (1 if "locket" in (carried_ids or []) else 0)


def unlocks_from_records(records: list[dict[str, Any]]) -> list[str]:
    """Giver keepsakes unlocked by the passed quests in these run records."""
    passed = {str(r.get("question_code") or "").upper() for r in records or [] if r.get("completed")}
    return [m for m, v in MEMENTOS.items() if v[0] in passed]


def stars_from_records(records: list[dict[str, Any]]) -> int:
    """Stars the player saw (0-5 per passed quest) across these run records."""
    total = 0
    for r in records or []:
        if not r.get("completed"):
            continue
        s5 = r.get("stars5")
        if s5 is None or int(s5) < 0:
            s5 = round(int(r.get("stars") or 0) * 5 / 3)
        total += max(0, int(s5))
    return total


BONUS_PREFIX = "bs:"
MAX_BONUS_PER_MARK = 10


def bonus_stars_from_ids(ids: list[str]) -> int:
    """Bonus stars earned outside quests (town events, morning letters, NPC preferences,
    the Lucky Coin). The game saves each one as a marker "bs:<source>:<n>" in the run's
    completed ids, so Continue keeps them and every marker counts once. Capped per marker."""
    total = 0
    for raw in dict.fromkeys(ids or []):
        s = str(raw)
        if not s.startswith(BONUS_PREFIX):
            continue
        try:
            n = int(s.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            continue
        total += max(0, min(n, MAX_BONUS_PER_MARK))
    return total


def _passed_count(records: list[dict[str, Any]]) -> int:
    return len({str(r.get("quest_id")) for r in records or [] if r.get("completed") and r.get("quest_id")})


def purse_stars(doc: dict[str, Any], records: list[dict[str, Any]]) -> int:
    """Coin Purse (carried): +1 bank star per quest passed (+2 at Lv2). Bank only."""
    if not is_carried(doc, "coinpurse"):
        return 0
    return _passed_count(records) * memento_level(doc, "coinpurse")


def _state_stars(state: dict[str, Any], doc: dict[str, Any] | None = None) -> int:
    """Quest stars + bonus stars (+ the Coin Purse) of one run-state."""
    records = state.get("quest_records") or []
    total = stars_from_records(records) + bonus_stars_from_ids(state.get("completed_quest_ids") or [])
    if doc is not None:
        total += purse_stars(doc, records)
    return total


def letter_passed(records: list[dict[str, Any]]) -> dict[str, int]:
    """Passed quests per RIASEC kind (the question code's letter), for the loop journal."""
    out = {k: 0 for k in TRAITS}
    seen = set()
    for r in records or []:
        qid = r.get("quest_id")
        code = str(r.get("question_code") or "").upper()
        if not r.get("completed") or not code or code[0] not in out or qid in seen:
            continue
        seen.add(qid)
        out[code[0]] += 1
    return out


def events_from_ids(ids: list[str]) -> tuple[set[str], bool]:
    """(RIASEC letters of the town events joined, whether a fiesta was seen) in one run."""
    letters: set[str] = set()
    fiesta = False
    for raw in ids or []:
        parts = str(raw).split(":")
        if len(parts) != 3 or parts[0] != "ev":
            continue
        if parts[2] == "fiesta":
            fiesta = True
        elif parts[2] in EVENT_LETTERS:
            letters.add(EVENT_LETTERS[parts[2]])
    return letters, fiesta


def preferred_codes(state: dict[str, Any]) -> set[str]:
    """Question codes solved the giver's preferred way this run ("bs:pref:<quest id>:n")."""
    records = state.get("quest_records") or []
    code_of = {str(r.get("quest_id")): str(r.get("question_code") or "").upper() for r in records if r.get("quest_id")}
    out = set()
    for raw in state.get("completed_quest_ids") or []:
        s = str(raw)
        if not s.startswith(BONUS_PREFIX + "pref:"):
            continue
        qid = s[len(BONUS_PREFIX + "pref:"):].rsplit(":", 1)[0]
        if code_of.get(qid):
            out.add(code_of[qid])
    return out


def _roll_threat(previous: str | None, seen: list[str]) -> str:
    choices = [t for t in THREATS if t != previous]
    unseen = [t for t in choices if t not in (seen or [])]
    return random.choice(unseen or choices)


def _new_doc(player_id: str) -> dict[str, Any]:
    now = datetime.utcnow()
    return {
        "player_id": player_id,
        "current_loop": 1,
        "loops_completed": 0,
        "current_finished": False,
        "threat": _roll_threat(None, []),
        "threats_seen": [],
        "kept_skills": [],
        "unlocked_mementos": [],
        "owned_mementos": [],
        "carried_mementos": [],
        "pockets": BASE_POCKETS,
        "board_unlocked": False,
        "stall_applied": [],
        "stall_spent_loop": 0,
        "lv2_unlocked": [],
        "lv2_owned": [],
        "events_joined": [],
        "fiesta_seen": False,
        "helpers_held": [],
        "helpers_bought_loop": [],
        "helpers_used_loop": [],
        "cosmetics_owned": [],
        "equipped": {},
        "stars": 0,
        "history": [],
        "last_owned_skills": [],
        "last_passed": [],
        "last_failed": [],
        "created_at": now,
        "updated_at": now,
    }


def get(repo, player_id: str) -> dict[str, Any]:
    """The player's loop record, created (loop 1, a random threat) if there is none yet."""
    doc = repo.player_loop_meta.find_one({"player_id": player_id})
    if doc:
        return doc
    doc = _new_doc(player_id)
    repo.player_loop_meta.update_one({"player_id": player_id}, {"$setOnInsert": doc}, upsert=True)
    return repo.player_loop_meta.find_one({"player_id": player_id}) or doc


def reset(repo, player_id: str) -> dict[str, Any]:
    """New Game: forget every loop and start again from loop 1."""
    repo.player_loop_meta.delete_one({"player_id": player_id})
    return get(repo, player_id)


def _add(repo, player_id: str, field: str, ids: list[str]) -> None:
    if ids:
        repo.player_loop_meta.update_one(
            {"player_id": player_id},
            {"$addToSet": {field: {"$each": ids}}, "$set": {"updated_at": datetime.utcnow()}},
        )


def _grant_event_mementos(repo, player_id: str, doc: dict[str, Any]) -> list[str]:
    """Free event keepsakes earned so far: Banderitas (a fiesta seen), Lucky Shell (any town
    event joined), Lucky Coin (all 6 event kinds joined, across loops). A new one goes
    straight into a free pocket."""
    owned = _list(doc, "owned_mementos")
    joined = set(_list(doc, "events_joined"))
    new = []
    if doc.get("fiesta_seen") and "banderitas" not in owned:
        new.append("banderitas")
    if joined and "shell" not in owned:
        new.append("shell")
    if joined >= set(TRAITS) and "luckycoin" not in owned:
        new.append("luckycoin")
    if not new:
        return []
    _add(repo, player_id, "owned_mementos", new)
    _fill_pockets(repo, player_id, new)
    return new


def _fill_pockets(repo, player_id: str, ids: list[str]) -> None:
    """A newly owned keepsake goes into a free pocket right away (if there is one)."""
    doc = get(repo, player_id)
    if doc.get("carried_mementos") is None:
        return                          # legacy record: everything owned is carried anyway
    bag = carried(doc)
    room = pockets(doc) - len(bag)
    add = [m for m in ids if m not in bag][:max(0, room)]
    _add(repo, player_id, "carried_mementos", add)


def sync_with_run_state(repo, player_id: str) -> dict[str, Any]:
    """The loop record, up to date with the current run: keepsakes unlocked so far this loop,
    town events joined, Lv2 upgrades earned (so Continue on any device still has them), and
    the loop is closed if the run is already finished."""
    doc = get(repo, player_id)
    state = repo.get_run_state(player_id)
    if state and not doc.get("current_finished"):
        records = state.get("quest_records") or []
        ids = state.get("completed_quest_ids") or []
        changed = False

        new = [m for m in unlocks_from_records(records) if m not in _list(doc, "unlocked_mementos")]
        if new:
            _add(repo, player_id, "unlocked_mementos", new)
            changed = True

        letters, fiesta = events_from_ids(ids)
        new_letters = sorted(letters - set(_list(doc, "events_joined")))
        if new_letters:
            _add(repo, player_id, "events_joined", new_letters)
            changed = True
        if fiesta and not doc.get("fiesta_seen"):
            repo.player_loop_meta.update_one({"player_id": player_id}, {"$set": {"fiesta_seen": True}})
            changed = True

        # Friendship (Lv2): a keepsake owned since before this loop, whose giver was helped
        # again this loop their preferred way.
        prefs = preferred_codes(state)
        started_with = set(_list(doc, "loop_start_owned"))
        lv2 = [m for m, v in MEMENTOS.items()
               if v[0] in prefs and m in started_with
               and m not in _list(doc, "lv2_unlocked") and m not in _list(doc, "lv2_owned")]
        if lv2:
            _add(repo, player_id, "lv2_unlocked", lv2)
            changed = True

        if changed:
            doc = get(repo, player_id)
        if _grant_event_mementos(repo, player_id, doc):
            doc = get(repo, player_id)
        if apply_stall_markers(repo, player_id, state):
            doc = get(repo, player_id)
        if state.get("run_finished"):
            doc = complete_loop(repo, player_id)
    return doc


# ── The loop journal: what changed between loops ─────────────────────────────

def _top(code: str) -> str:
    code = (code or "").strip().upper()
    return code[0] if code and code[0] in TRAITS else ""


def _join(parts: list[str]) -> str:
    if not parts:
        return ""
    s = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return s[0].upper() + s[1:] + "."


def compare_loops(prev: dict[str, Any] | None, cur: dict[str, Any]) -> dict[str, Any]:
    """What changed from one loop to the next, in plain words, built only from what the two
    loops recorded. `reason_line` is empty when the loops didn't record enough to say why."""
    out = {"changed": False, "change_line": "", "reason_line": ""}
    if not prev:
        return out
    a, b = _top(prev.get("holland_code")), _top(cur.get("holland_code"))
    pc, cc = str(prev.get("holland_code") or "").upper(), str(cur.get("holland_code") or "").upper()
    if not a or not b:
        return out
    if a != b:
        out["changed"] = True
        out["change_line"] = f"Your path moved from {TRAITS[a]} to {TRAITS[b]}."
    elif pc != cc:
        out["changed"] = True
        out["change_line"] = f"Still {TRAITS[a]} first, but your code shifted from {pc} to {cc}."
    elif str(prev.get("career") or "") != str(cur.get("career") or ""):
        out["changed"] = True
        out["change_line"] = f"Same code ({cc}), a different career this time."
    else:
        out["change_line"] = f"Same path as loop {int(prev.get('loop') or 0)}: {TRAITS[b]}."
        return out

    reasons = []
    kept_b = [SKILLS[s][0] for s in cur.get("kept_skills") or [] if s in SKILLS and SKILLS[s][1] == b]
    if kept_b:
        reasons.append("you remembered " + " and ".join(kept_b))
    lp, lc = prev.get("letter_passed") or {}, cur.get("letter_passed") or {}
    if lp and lc:
        nb_prev, nb_cur = int(lp.get(b) or 0), int(lc.get(b) or 0)
        if nb_cur > nb_prev:
            reasons.append(f"finished more {TRAITS[b]} quests ({nb_cur}, up from {nb_prev})")
        if a != b:
            na_prev, na_cur = int(lp.get(a) or 0), int(lc.get(a) or 0)
            if na_cur < na_prev:
                reasons.append(f"fewer {TRAITS[a]} ones ({na_cur}, down from {na_prev})")
    p_prev, p_cur = int(prev.get("passed") or 0), int(cur.get("passed") or 0)
    if p_cur > p_prev and len(reasons) < 2:
        reasons.append(f"helped more people ({p_cur}, up from {p_prev})")
    out["reason_line"] = _join(reasons[:3])
    return out


def journal(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History entries, each with what changed since the loop before."""
    out = []
    prev = None
    for h in history or []:
        e = dict(h)
        e.update(compare_loops(prev, h))
        out.append(e)
        prev = h
    return out


def paths(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The final result rule when loops disagree: show both the first path (loop 1, the
    baseline) and the latest. None until there are two finished loops."""
    if not history or len(history) < 2:
        return None
    first, last = history[0], history[-1]
    differ = (str(first.get("holland_code") or "").upper() != str(last.get("holland_code") or "").upper()
              or str(first.get("career") or "") != str(last.get("career") or ""))
    cmp = compare_loops(first, last)
    return {
        "differ": differ,
        "baseline": {"loop": int(first.get("loop") or 1), "holland_code": first.get("holland_code") or "",
                     "career": first.get("career") or ""},
        "latest": {"loop": int(last.get("loop") or 0), "holland_code": last.get("holland_code") or "",
                   "career": last.get("career") or ""},
        "change_line": cmp["change_line"] if differ else "",
        "reason_line": cmp["reason_line"] if differ else "",
    }


def complete_loop(repo, player_id: str) -> dict[str, Any]:
    """End of a run: write the loop into the journal, bank its stars, keep the unlocks,
    remember what the player had. Safe to call twice - the second call changes nothing."""
    doc = get(repo, player_id)
    if doc.get("current_finished"):
        return doc

    state = repo.get_run_state(player_id) or {}
    records = state.get("quest_records") or []
    passed = sorted({str(r.get("quest_id")) for r in records if r.get("completed") and r.get("quest_id")})
    failed = sorted({str(r.get("quest_id")) for r in records if not r.get("completed") and r.get("quest_id")} - set(passed))
    _add(repo, player_id, "unlocked_mementos", unlocks_from_records(records))
    apply_stall_markers(repo, player_id, state)
    doc = get(repo, player_id)
    earned = _state_stars(state, doc)
    spent = int(doc.get("stall_spent_loop") or 0)

    latest = repo.find_latest_run_for_player(player_id) or {}
    loop_no = int(doc.get("current_loop") or 1)
    threat = doc.get("threat") or ""
    now = datetime.utcnow()
    bag = carried(doc)
    helpers_used = _list(doc, "helpers_used_loop")

    entry = {
        "loop": loop_no,
        "threat": threat,
        "holland_code": str(latest.get("cluster_holland_code") or latest.get("holland_code") or ""),
        "career": str(latest.get("career_result") or latest.get("career_family") or latest.get("cluster_label") or ""),
        "passed": len(passed),
        "failed": len(failed),
        "letter_passed": letter_passed(records),
        "riasec_scores": dict(latest.get("riasec_scores") or {}),
        "kept_skills": _list(doc, "kept_skills"),
        "mementos": bag,                               # carried this loop (in the pockets)
        "mementos_lv2": [m for m in bag if memento_level(doc, m) == 2],
        "helpers_used": helpers_used,
        "stars_earned": earned,
        "stars_spent": spent,
        "finished_at": now,
    }

    seen = _list(doc, "threats_seen")
    if threat and threat not in seen:
        seen.append(threat)

    repo.player_loop_meta.update_one(
        {"player_id": player_id},
        {
            "$set": {
                "current_finished": True,
                "loops_completed": loop_no,
                "threats_seen": seen,
                "last_owned_skills": list(state.get("owned_skills") or []),
                "last_skill_xp": list(state.get("skill_xp") or []),
                "last_passed": passed,
                "last_failed": failed,
                "stall_spent_loop": 0,
                "updated_at": now,
            },
            "$inc": {"stars": earned - spent},
            "$push": {"history": entry},
        },
    )

    # Label the finished run itself, so analysis can separate loop 1 from later loops.
    if latest.get("_id") is not None:
        repo.session_runs.update_one(
            {"_id": latest["_id"]},
            {"$set": {
                "loop": loop_no,
                "threat": threat,
                "kept_skills": entry["kept_skills"],
                "mementos": bag,
                "mementos_lv2": entry["mementos_lv2"],
                "helpers_used": helpers_used,
                # skill XP at the end of the run: later loops have stronger skills (faster
                # solves, more stars), so analysis can see what each run was played with
                "skill_xp": list(state.get("skill_xp") or []),
            }},
        )
    return get(repo, player_id)


def run_stars(repo, player_id: str, doc: dict[str, Any]) -> int:
    """Stars earned so far in the loop being played (not banked yet)."""
    if doc.get("current_finished"):
        return 0
    state = repo.get_run_state(player_id) or {}
    return _state_stars(state, doc)


def available_stars(repo, player_id: str, doc: dict[str, Any]) -> int:
    """What the stall can spend: the bank plus this loop's stars so far, minus what the stall
    already took this loop (offline buys, applied from the run's markers)."""
    spent = 0 if doc.get("current_finished") else int(doc.get("stall_spent_loop") or 0)
    return int(doc.get("stars") or 0) + run_stars(repo, player_id, doc) - spent


def _spend(repo, player_id: str, cost: int, earned_now: int, guard: dict[str, Any], update: dict[str, Any]) -> None:
    """One atomic purchase: spends only if bank + this loop's stars cover it and `guard`
    still holds (so a double tap can't buy twice)."""
    update = {k: dict(v) for k, v in update.items()}
    update.setdefault("$inc", {})["stars"] = -cost
    update.setdefault("$set", {})["updated_at"] = datetime.utcnow()
    result = repo.player_loop_meta.update_one(
        {"player_id": player_id, "stars": {"$gte": cost - earned_now}, **guard}, update)
    if result.matched_count == 0:
        raise LoopError("Not enough stars.")


def buy_memento(repo, player_id: str, memento_id: str) -> dict[str, Any]:
    """Stall, online: buy an unlocked giver keepsake with stars. Goes into a free pocket."""
    if memento_id not in MEMENTOS:
        raise LoopError("Unknown keepsake.")
    return buy(repo, player_id, memento_id)


def _purchase(doc: dict[str, Any], item_id: str):
    """The checks and the change for one stall item, shared by the endpoint and the offline
    markers. Returns None when there is nothing to do (already owned), else
    (cost, guard, update, keepsakes to put in a free pocket). Raises LoopError when it can't
    be bought. Item ids: a keepsake, "lv2:<keepsake>", "pocket" / "pocket3" / "pocket4", a
    helper or a cosmetic."""
    item_id = str(item_id or "").strip()
    if item_id in MEMENTOS:
        if item_id in _list(doc, "owned_mementos"):
            return None
        if item_id not in _list(doc, "unlocked_mementos"):
            raise LoopError("Locked.")
        return (MEMENTOS[item_id][1], {"owned_mementos": {"$ne": item_id}},
                {"$addToSet": {"owned_mementos": item_id}}, [item_id])
    if item_id.startswith("lv2:"):
        m = item_id[4:]
        if m not in MEMENTOS:
            raise LoopError("Unknown keepsake.")
        if m in _list(doc, "lv2_owned"):
            return None
        if m not in _list(doc, "lv2_unlocked") or m not in _list(doc, "owned_mementos"):
            raise LoopError("Locked.")
        return MEMENTOS[m][2], {"lv2_owned": {"$ne": m}}, {"$addToSet": {"lv2_owned": m}}, []
    if item_id in ("pocket", "pocket3", "pocket4"):
        have = pockets(doc)
        want = have + 1 if item_id == "pocket" else int(item_id[-1])
        if have >= MAX_POCKETS and item_id == "pocket":
            raise LoopError("All pockets bought.")
        if want <= have:
            return None
        if want != have + 1:
            raise LoopError("Buy the pockets in order.")
        guard = {"pockets": have} if doc.get("pockets") is not None else {"pockets": {"$exists": False}}
        # a pocket bought mid-loop holds a keepsake that was left at home
        rest = [m for m in _list(doc, "owned_mementos") if m not in carried(doc)]
        return POCKET_PRICES[want], guard, {"$set": {"pockets": want}}, rest[:1]
    if item_id == "board":
        if doc.get("board_unlocked"):
            return None
        return BOARD_PRICE, {"board_unlocked": {"$ne": True}}, {"$set": {"board_unlocked": True}}, []
    if item_id in HELPERS:
        if item_id in _list(doc, "helpers_held"):
            raise LoopError("You already have one.")
        if item_id in _list(doc, "helpers_bought_loop"):
            raise LoopError("One per loop.")
        return (HELPERS[item_id], {"helpers_held": {"$ne": item_id}, "helpers_bought_loop": {"$ne": item_id}},
                {"$addToSet": {"helpers_held": item_id, "helpers_bought_loop": item_id}}, [])
    if item_id in COSMETICS:
        if item_id in _list(doc, "cosmetics_owned"):
            return None
        return (COSMETICS[item_id][1], {"cosmetics_owned": {"$ne": item_id}},
                {"$addToSet": {"cosmetics_owned": item_id}}, [])
    raise LoopError("Unknown item.")


def buy(repo, player_id: str, item_id: str) -> dict[str, Any]:
    """Stall, online: buy anything with stars right now (atomic). The game itself buys
    offline through the run's markers (apply_stall_markers); this stays for other callers."""
    doc = sync_with_run_state(repo, player_id)
    res = _purchase(doc, item_id)
    if res is None:
        return doc
    cost, guard, update, fill = res
    spent = 0 if doc.get("current_finished") else int(doc.get("stall_spent_loop") or 0)
    _spend(repo, player_id, cost, run_stars(repo, player_id, doc) - spent, guard, update)
    if fill:
        _fill_pockets(repo, player_id, fill)
    return get(repo, player_id)


BUY_PREFIX, USE_PREFIX, EQUIP_PREFIX = "sb:", "su:", "se:"


def _equip_update(doc: dict[str, Any], slot: str, item_id: str) -> dict[str, Any]:
    """The change for wearing / showing a cosmetic (shared by equip() and the markers)."""
    item_id = str(item_id or "")
    if slot in ("home", "homeon", "homeoff"):
        if item_id not in COSMETICS or COSMETICS[item_id][0] != "home":
            raise LoopError("Unknown item.")
        if item_id not in _list(doc, "cosmetics_owned"):
            raise LoopError("Buy it first.")
        home = list((doc.get("equipped") or {}).get("home") or [])
        # "homeon" / "homeoff" (what the game saves) are absolute; "home" toggles (older callers)
        on = (item_id not in home) if slot == "home" else (slot == "homeon")
        home = [h for h in home if h != item_id] + ([item_id] if on else [])
        return {"$set": {"equipped.home": home}}
    if slot not in EQUIP_SLOTS:
        raise LoopError("Unknown slot.")
    if slot == "outfit" and item_id in STAGE_UNIFORMS:
        return {"$set": {"equipped.outfit": item_id}}
    if item_id:
        if item_id not in COSMETICS or COSMETICS[item_id][0] != slot:
            raise LoopError("Unknown item.")
        if item_id not in _list(doc, "cosmetics_owned"):
            raise LoopError("Buy it first.")
    return {"$set": {f"equipped.{slot}": item_id}}


def apply_stall_markers(repo, player_id: str, state: dict[str, Any] | None) -> bool:
    """Applies the stall actions the game saved offline in this run's progress ids, in order,
    each once. One that doesn't pass the checks (or can't be afforded) is skipped and never
    retried. Returns True when anything was applied."""
    if not state:
        return False
    ids = [str(m) for m in dict.fromkeys(state.get("completed_quest_ids") or [])
           if str(m).startswith((BUY_PREFIX, USE_PREFIX, EQUIP_PREFIX))]
    doc = get(repo, player_id)
    if doc.get("current_finished"):
        return False
    applied = set(_list(doc, "stall_applied"))
    todo = [m for m in ids if m not in applied]
    if not todo:
        return False
    balance = int(doc.get("stars") or 0) + _state_stars(state, doc) - int(doc.get("stall_spent_loop") or 0)
    for m in todo:
        guard: dict[str, Any] = {}
        update: dict[str, Any] = {}
        fill: list[str] = []
        try:
            if m.startswith(BUY_PREFIX):
                res = _purchase(doc, m[len(BUY_PREFIX):])
                if res is not None:
                    cost, guard, update, fill = res
                    if cost > balance:
                        raise LoopError("Not enough stars.")
                    update = {k: dict(v) for k, v in update.items()}
                    update.setdefault("$inc", {})["stall_spent_loop"] = cost
                    balance -= cost
            elif m.startswith(USE_PREFIX):
                h = m[len(USE_PREFIX):]
                if h not in HELPERS or h not in _list(doc, "helpers_held") or h in _list(doc, "helpers_used_loop"):
                    raise LoopError("You don't have that helper.")
                guard = {"helpers_held": h}
                update = {"$pull": {"helpers_held": h}, "$addToSet": {"helpers_used_loop": h}}
            else:
                parts = m.split(":", 3)
                if len(parts) != 4:
                    raise LoopError("Bad marker.")
                update = _equip_update(doc, parts[2], parts[3])
        except LoopError:
            guard, update, fill = {}, {}, []
        update = {k: dict(v) for k, v in update.items()}
        update.setdefault("$addToSet", {})["stall_applied"] = m
        update.setdefault("$set", {})["updated_at"] = datetime.utcnow()
        result = repo.player_loop_meta.update_one(
            {"player_id": player_id, "stall_applied": {"$ne": m}, **guard}, update)
        if result.matched_count == 0:
            # the guard failed (e.g. bought meanwhile) or another sync got here first
            repo.player_loop_meta.update_one({"player_id": player_id}, {"$addToSet": {"stall_applied": m}})
        elif fill:
            _fill_pockets(repo, player_id, fill)
        doc = get(repo, player_id)
    return True


def debug_set_stars(repo, player_id: str, stars: int) -> dict[str, Any]:
    """Admin debug tool: the stall shows exactly this many stars. Sets the bank so that bank +
    this loop's stars - what the stall spent this loop = `stars`."""
    doc = sync_with_run_state(repo, player_id)
    now_available = available_stars(repo, player_id, doc)
    bank = int(doc.get("stars") or 0) + max(0, int(stars)) - now_available
    repo.player_loop_meta.update_one({"player_id": player_id},
                                     {"$set": {"stars": bank, "updated_at": datetime.utcnow()}})
    return get(repo, player_id)


def use_helper(repo, player_id: str, helper_id: str) -> dict[str, Any]:
    """Uses a one-use run helper (atomic: only if held and not used this loop yet). The use
    is recorded on the loop, so the finished run says which helpers it was played with."""
    if helper_id not in HELPERS:
        raise LoopError("Unknown helper.")
    doc = get(repo, player_id)
    if doc.get("current_finished"):
        raise LoopError("This loop is over.")
    result = repo.player_loop_meta.update_one(
        {"player_id": player_id, "helpers_held": helper_id, "helpers_used_loop": {"$ne": helper_id}},
        {"$pull": {"helpers_held": helper_id}, "$addToSet": {"helpers_used_loop": helper_id},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise LoopError("You don't have that helper.")
    return get(repo, player_id)


def equip(repo, player_id: str, slot: str, item_id: str) -> dict[str, Any]:
    """Online: wears / shows an owned cosmetic (empty item = take it off). Home add-ons
    toggle. The game itself equips offline through "se:" markers."""
    doc = get(repo, player_id)
    repo.player_loop_meta.update_one({"player_id": player_id}, _equip_update(doc, slot, item_id))
    return get(repo, player_id)


def begin_loop(repo, player_id: str, kept_skills: list[str], carried_ids: list[str] | None = None) -> dict[str, Any]:
    """Go Back: start the next loop with the skills the player chose to remember and the
    keepsakes they put in their pockets (None = an older game: keep last loop's pockets)."""
    doc = sync_with_run_state(repo, player_id)
    if not doc.get("current_finished"):
        raise LoopError("Finish this loop first.")

    owned = _list(doc, "owned_mementos")
    ending_bag = carried(doc)
    allowed = set(_list(doc, "last_owned_skills")) | set(_list(doc, "kept_skills"))
    kept = [s for s in dict.fromkeys(kept_skills or []) if s in allowed][:memory_slots(ending_bag)]

    room = pockets(doc)
    if carried_ids is None:
        carried_ids = ending_bag
    bag = [m for m in dict.fromkeys(carried_ids) if m in owned][:room]

    # Remembered skills keep their XP; every other skill starts the next loop at 0.
    xp_by_skill: dict[str, int] = {}
    for e in _list(doc, "kept_skill_xp") + _list(doc, "last_skill_xp"):
        name = str(e.get("skill") or "")
        if name:
            xp_by_skill[name] = max(xp_by_skill.get(name, 0), int(e.get("xp") or 0))
    # Memory Locket Lv2 (carried in the loop that ended): remembered skills start with +2 XP.
    bonus = 2 if "locket" in ending_bag and memento_level(doc, "locket") == 2 else 0
    kept_xp = [{"skill": s, "xp": xp_by_skill.get(s, 0) + bonus} for s in kept if xp_by_skill.get(s, 0) + bonus > 0]
    # Name Tag (carried in the loop that ended): forgotten skills keep 1/2 of their XP (3/4
    # at Lv2), waiting for the skill to be found again.
    faded_xp = []
    if "nametag" in ending_bag:
        num, den = (3, 4) if memento_level(doc, "nametag") == 2 else (1, 2)
        faded_xp = [{"skill": s, "xp": x * num // den} for s, x in xp_by_skill.items()
                    if s not in kept and x * num // den > 0]

    previous = doc.get("threat")
    repo.player_loop_meta.update_one(
        {"player_id": player_id},
        {"$set": {
            "current_loop": int(doc.get("current_loop") or 1) + 1,
            "current_finished": False,
            "threat": _roll_threat(previous, _list(doc, "threats_seen")),
            "kept_skills": kept,
            "kept_skill_xp": kept_xp,
            "faded_skill_xp": faded_xp,
            "carried_mementos": bag,
            # a new childhood: school uniforms come off until you reach that school again
            **({"equipped.outfit": ""} if (doc.get("equipped") or {}).get("outfit") in STAGE_UNIFORMS else {}),
            "loop_start_mementos": bag,        # what this loop carried (for the run label)
            "loop_start_owned": owned,         # owned before this loop (for Lv2 unlocks)
            "helpers_bought_loop": [],
            "helpers_used_loop": [],
            "stall_applied": [],
            "stall_spent_loop": 0,
            "updated_at": datetime.utcnow(),
        }},
    )
    # The finished run's save is done with (its records are already on the finished
    # session_run). Clearing it here too means a stale "finished" save can never close the
    # new loop, even if the game's own reset call didn't land.
    repo.clear_run_state(player_id)
    return get(repo, player_id)
