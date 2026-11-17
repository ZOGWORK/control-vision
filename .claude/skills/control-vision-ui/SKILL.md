---
name: control-vision-ui
description: Build Control Vision UI with PySide6-Fluent-Widgets, not hand-rolled QSS. Use for any UI work in ZOGWORK/control-vision: new pages, fixing pages, sidebar, buttons, cards, dialogs, theme.
---

# Control Vision UI Skill

## Rule 1 — Use the library, do not hand-roll

- Use `qfluentwidgets` for every standard component.
- Never write a custom QPushButton with QSS for a normal button.
- Never write a custom sidebar with QListWidget + QSS.
- Never write a custom card with a QFrame + QSS.
- Never write a custom dialog with QDialog + QSS.
- If a component exists in the library, use it.
- Only build custom widgets for Vision-specific things, see Rule 3.

## Rule 2 — One shell, one theme

- Main window is `FluentWindow` from `qfluentwidgets`.
- Navigation is `NavigationInterface`, with `FluentIcon` icons.
- Theme is dark, set with `qfluentwidgets.setTheme(Theme.DARK)`.
- Override the Fluent accent to Control volt, see theme.py.
- Do not use Fluent's default blue accent anywhere.
- Dark graphite base, volt primary, Red / Blue for alliances, Green for confirmed, Amber for needs review.
- Buttons must visibly look like buttons: border or fill, hover visible.

## Rule 3 — Custom widgets only for these

- Video player with boxes / IDs / timeline scrubber
- Field map, top-down 12x12 ft, with trails
- Scoring timeline, with phase bands and markers
- Labelling canvas, with boxes and frame strip
- Calibration canvas, with 4 corners and zones
- Everything else is a library component.

## Rule 4 — Page structure

- Every page is a library `CardWidget` layout, or a `FluentWindow` sub-interface.
- Header: TitleLabel, bold, 24px, plus a grey subtitle.
- Content: cards, with a visible border, 16px padding.
- No full-page centred empty box.
- Empty state is a small card at the top of a real layout, with one clear action.
- A match cannot be scored until Review Queue is cleared, show that in the UI, do not hide it.

## Rule 5 — State

- One source of truth for current project and current match.
- Every page reads it from the same store.
- Match Results must not say "Open a project" when a project is open.
- If no results yet, say "No results yet for [match] — Process this match", with a button to Set up and process.

## Rule 6 — No invented data

- No invented scores, team numbers, resolutions, or event codes.
- Missing data is "—", with a label.
- Official score is from ftc-events.firstinspires.org, shown next to ours, never merged.
- No accuracy claim until it is in docs/ACCURACY_REPORT.md with its holdout match list.
