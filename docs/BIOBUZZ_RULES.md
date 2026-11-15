# BIOBUZZ game reference for Control Vision

**Source of truth:** the official 2026–27 FIRST Tech Challenge BIOBUZZ Competition Manual, Version TU09, updated November 12, 2026. This guide paraphrases the rules most relevant to video scouting; the manual and event-specific Team Updates take precedence. Control Vision records evidence and calculates confirmed scoring entries. It is not a referee and cannot determine rule violations, card decisions, scoring ownership, or the official result.

Official sources: [Competition Manual TU09](https://ftc-resources.firstinspires.org/ftc/archive/2027/game/cm-html/BIOBUZZ%20Competition%20Manual%20-%20TU09.htm), [official game and season materials](https://ftc-resources.firstinspires.org/ftc/archive/2027/game), and [Team Update 09](https://ftc-resources.firstinspires.org/ftc/archive/2027/game/tu-09).

## Match timing

The match has 30 seconds AUTO, an 8-second AUTO-to-TELEOP transition, and 120 seconds TELEOP. Thus there are 150 seconds of AUTO plus TELEOP gameplay and 158 seconds on a continuous source-video timeline if the transition is represented. The field timer is authoritative. Flower scoring unlocks when the TELEOP timer reaches 60 seconds remaining: 98 seconds after the start on a continuous elapsed timeline. Before that point, putting NECTAR into the FLOWER scoring volume earns the scoring achievement but incurs a Major Foul for each early NECTAR. Events completed before match start, during transition, or after 0:00 can also be penalized.

## Field and scoring elements

- FIELD: 12 by 12 feet, six 24-inch tiles across each side. The manual's field diagrams are illustrative and state they are not to scale; field setup has tolerances. Control Vision therefore requires camera-specific corners and user confirmation of drawn zones.
- POLLEN: yellow, approximately 2.8 inches. NECTAR: red or blue, approximately 3.6 inches. A match stages 40 POLLEN, 8 red NECTAR, and 8 blue NECTAR.
- HIVE: centered frame with one red and one blue two-CELL assembly. A HIVE TIP is a transition between stable states that leaves the formerly free damper in contact with the frame; assess TIPS through the end after robots/elements settle. A TIP completed before TELEOP is an AUTO TIP.
- FLOWER: four field structures. An element scores when at least partially inside the scoring volume between the top and middle rings. Elements may only enter from above. Only POLLEN may be removed from below. The top-most qualifying NECTAR of its color establishes FLOWER ownership; the owner earns points for every qualifying POLLEN and NECTAR there, regardless of who placed them. The bottom-most qualifying NECTAR of its color earns the Bottom NECTAR Bonus.
- GARDEN: two alliance-colored zones. An element earns for the Garden color, regardless of which alliance placed it. A Garden is not protected; either alliance can remove elements from either Garden.
- CELL end state: count POLLEN/NECTAR left in the upward-facing CELL after all robots and elements settle.
- LEAVE: robot no longer touches the perimeter wall at end of AUTO.
- PARK: robot at least partially inside its LOADING ZONE at the relevant period end.

## Match-point scoring (Table 10-2)

| Achievement | Match points | Assessment |
|---|---:|---|
| LEAVE | 3 | Each qualifying robot, end of AUTO |
| PARK | 5 | Each qualifying robot; AUTO or TELEOP, assessed at corresponding end |
| HIVE TIP | 20 | Per achieved TIP; AUTO/TELEOP period classification follows when completed |
| POLLEN or NECTAR remaining in upward CELL | 2 each | End of match after settling |
| Bottom NECTAR Bonus in FLOWER | 5 | Per FLOWER, awarded to bottom-most qualifying NECTAR color |
| POLLEN or NECTAR in an owned FLOWER | 2 each | Awarded to FLOWER owner after final end-state assessment |
| POLLEN or NECTAR in GARDEN | 1 each | Awarded to the Garden color at end of TELEOP |
| MINOR FOUL | 5 to opponent | Referee-assessed penalty credit |
| MAJOR FOUL | 20 to opponent | Referee-assessed penalty credit |

These achievements belong to ALLIANCES. The robot that caused an event can be unknown, on the other alliance, or different from the alliance receiving points. The review screen therefore stores robot attribution separately from scoring-alliance recipient. Match scores can include referee penalties, disqualification, or event corrections that cannot be inferred from video.

## Ranking Points (Table 10-3)

For standard events, each eligible team on an alliance receives:

- **SWARM RP (1):** combined LEAVE + PARK points meet or exceed 16.
- **POLLINATOR 1 RP (1):** at least 4 HIVE TIPS.
- **POLLINATOR 2 RP (1):** at least 7 HIVE TIPS.
- **Win (3):** the alliance has more match points than its opponent.
- **Tie (1):** match points are equal.

Regional and FIRST Championship thresholds are published in later Team Updates; Premier Events may set their own thresholds. Control Vision exposes threshold overrides and defaults to the all-other-events thresholds. In Qualification, disqualification applies only to the disqualified team, which receives zero match and ranking points; its alliance partner is unaffected (T601). In Playoffs, a team disqualification applies to its entire alliance, which receives zero match points. The current alliance-DQ control represents the Playoff case only; team-level Qualification disqualification still needs a separate implementation.

## Game-rule reference (Section 11)

All enforcement and intent calls below belong to event referees. The app must not automatically call fouls based only on vision.

- **G101–G102, safety:** people stay off the field during play; do not climb on, hang from, deform, or damage arena elements.
- **G201–G205, conduct/integrity:** civil behavior and the Competition Integrity Contract are required; no match manipulation, forced opponent violations, or egregious behavior. Head Referees may issue cards.
- **G301–G305, pre-match:** avoid significant match delays; arrive match-ready; robots must be safe, inspected, correctly identified, legally placed on their alliance side, touching the perimeter, outside the Loading Zone and Flower scoring volume, motionless after initialization, and staged with four pre-load POLLEN; select and initialize an OpMode.
- **G401–G402, AUTO:** no drive-team interaction with robots/console during AUTO except permitted start/stop/safety actions; do not disrupt the opponent's AUTO. Red has priority over columns A–C and blue over D–F.
- **G403–G404, transitions/end:** no powered robot/mechanism motion during the AUTO-to-TELEOP transition or after TELEOP ends until retrieval is signaled.
- **G405–G409, elements:** no deliberate scoring-element ejection from the field; avoid arena damage; control at most four scoring elements; do not control opponent NECTAR; do not catch/deflect an element released by a tipped HIVE before it hits another object.
- **G410–G411, scoring access:** NECTAR may enter the FLOWER scoring volume only in the last 60 seconds (Major Foul per early NECTAR); an alliance may not prevent the opponent from accessing more than eight elements.
- **G412–G418, robot actions:** robots must be safe/controlled and stop when instructed; team number/alliance color must remain identifiable; do not strategically attach to arena elements, exceed construction limits, manipulate a HIVE except by launching elements into its upward CELL, or score/remove elements from Flowers except through their permitted openings. Only POLLEN may be removed from a Flower bottom.
- **G419–G421, defense:** do not strategically damage/disable, tip/entangle, or pin an opponent robot. Pinning has a 3-second count and reset/stop conditions; referee determines contact and intent.
- **G422–G428, humans:** drive team stays in its alliance area; coaches cannot handle controls or scoring elements; drive team cannot contact robots, tiles/elements, or field elements to affect play; NECTAR can enter only after HIVE-TIP release opportunities or in the last 60 seconds; only the matching-color drive team may introduce its NECTAR through its Loading Zone without a tool, with tile contact before robot/field contact; humans may not remove scoring elements. See manual rule text for exceptions and penalties.

## Penalties, cards, and authority

A Minor Foul credits 5 points to the opponent; a Major Foul credits 20. Yellow/Red Cards, disabling, and disqualification depend on referee judgement and may affect a match or later matches. A Red Card disqualifies a team for the match; a team disqualified in Qualification receives zero match and ranking points, while a Playoff alliance receives zero match points. The Head Referee has final authority, and FIRST event staff do not use video replay to overturn a match ruling. Users should enter official corrections and disqualification from the official score record; video observations are review notes only.

## Assessment and provenance

The official manual says scoring is assessed by human volunteers. HIVE TIP assessment continues until rest; CELL, FLOWER, and GARDEN have end-state checks; LEAVE/AUTO PARK are checked at AUTO end; TELEOP PARK is checked at match end. Video candidates stay flagged until a human confirms them. BIOBUZZ values and event categories are versioned in `control_vision/config/scoring_biobuzz.json`; each season uses a separate config.
