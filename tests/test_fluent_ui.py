import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]


def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FluentFoundationTests(unittest.TestCase):
    def setUp(self):
        self.app = qt_app()
        from control_vision.ui import theme
        theme.apply()

    def test_theme_is_dark_with_the_volt_accent_not_fluent_blue(self):
        from PySide6.QtGui import QColor
        from qfluentwidgets import isDarkTheme, qconfig, themeColor
        from control_vision.ui import theme
        self.assertTrue(isDarkTheme())
        self.assertEqual(qconfig.get(qconfig.themeColor).name().upper(), "#DFFF1A")
        # On dark, Fluent draws a lighter shade of the accent; it must still be volt (yellow-green), not blue.
        self.assertLess(abs(themeColor().hslHue() - QColor(theme.VOLT).hslHue()), 6)

    def test_buttons_are_the_library_widgets(self):
        from qfluentwidgets import PrimaryPushButton, PushButton
        from control_vision.ui import components, theme
        self.assertIsInstance(components.VisionPrimaryButton("Process"), PrimaryPushButton)
        self.assertIsInstance(components.VisionSecondaryButton("Cancel"), PushButton)
        delete = components.VisionDestructiveButton("Delete correction")
        self.assertIsInstance(delete, PrimaryPushButton)
        self.assertIn(theme.DESTRUCTIVE, delete.property("darkCustomQss"))

    def test_missing_scores_and_teams_show_a_dash_never_a_number(self):
        from qfluentwidgets import StrongBodyLabel
        from control_vision.ui import components
        strip = components.VisionScoreStrip(ours=(120, None), official=(None, 95))
        texts = [label.text() for label in strip.findChildren(StrongBodyLabel)]
        self.assertEqual(texts, ["Red  120  ·  official —", "Blue  —  ·  official 95"])
        card = components.VisionRobotCard(None, None)
        self.assertEqual([label.text() for label in card.findChildren(StrongBodyLabel)], ["Team —"])

    def test_cards_use_the_library_card_with_16px_padding(self):
        from qfluentwidgets import CardWidget
        from control_vision.ui import components
        clicked = []
        card = components.empty_state_card("No results yet for Q3", "Set up and process", lambda: clicked.append(1))
        self.assertIsInstance(card, CardWidget)
        self.assertEqual(card.body.contentsMargins().left(), 16)
        button = card.findChild(components.VisionPrimaryButton)
        button.click()
        self.assertEqual(clicked, [1])


class UiSkillTests(unittest.TestCase):
    def test_skill_is_in_the_repo_and_in_docs_with_the_same_text(self):
        skill = (ROOT / ".claude" / "skills" / "control-vision-ui" / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill.startswith("---\nname: control-vision-ui\n"))
        self.assertEqual(skill, (ROOT / "docs" / "CONTROL_VISION_UI_SKILL.md").read_text(encoding="utf-8"))

    def test_library_version_is_pinned(self):
        for name in ("requirements.in", "requirements.txt"):
            text = (ROOT / name).read_text(encoding="utf-8").lower()
            self.assertIn("pyside6-fluent-widgets==1.11.3", text, name)


if __name__ == "__main__":
    unittest.main()
