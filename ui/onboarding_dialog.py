"""
onboarding_dialog.py - First-boot local vault onboarding.
"""

from __future__ import annotations

import asyncio
import random
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.settings_store import SettingsStore
from core.vault import VaultError, VaultManager, VaultValidationError
from ui.fan_item import SpinningFanItem


_SCREW_COLOR = QColor("#3a3d42")
_SCREW_BORDER = QColor("#2a2d32")
_SCREW_SLOT = QColor("#22242a")


class _ScrewOverlay(QWidget):
    """Transparent child widget that only draws corner screws.

    Sits on top of its parent frame, passes all mouse events through,
    and never contains child widgets — so it cannot conflict with
    QGraphicsEffect rendering on siblings.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect().adjusted(1, 1, -1, -1)
        for x, y in [
            (r.left() + 14, r.top() + 14),
            (r.right() - 14, r.top() + 14),
            (r.left() + 14, r.bottom() - 14),
            (r.right() - 14, r.bottom() - 14),
        ]:
            self._draw_screw(painter, x, y)
        painter.end()

    @staticmethod
    def _draw_screw(painter: QPainter, x: float, y: float) -> None:
        painter.save()
        painter.setPen(QPen(_SCREW_BORDER, 0.8))
        painter.setBrush(_SCREW_COLOR)
        painter.drawEllipse(QRectF(x - 5.0, y - 5.0, 10.0, 10.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_SCREW_SLOT)
        painter.drawRect(QRectF(x - 3.5, y - 0.8, 7.0, 1.6))
        painter.restore()


class _StepIndicator(QWidget):
    """Bottom LED-style progress indicator."""

    _LABELS = ("WELCOME", "RECOVERY", "CONFIRM", "PROFILE", "ONBOARD")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current = 0
        self.setMinimumHeight(58)

    def set_step(self, idx: int) -> None:
        self._current = max(0, min(idx, len(self._LABELS) - 1))
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        count = len(self._LABELS)
        content = self.rect().adjusted(42, 8, -42, -8)
        step_w = content.width() / max(1, count - 1)
        cy = content.top() + 12

        for i in range(count - 1):
            x1 = content.left() + i * step_w
            x2 = content.left() + (i + 1) * step_w
            painter.setPen(QPen(QColor("#3a4048"), 1.3))
            painter.drawLine(int(x1 + 8), int(cy), int(x2 - 8), int(cy))

        for i, label in enumerate(self._LABELS):
            cx = content.left() + i * step_w
            if i < self._current:
                fill = QColor("#1a6020")
                border = QColor("#2a8030")
            elif i == self._current:
                fill = QColor("#30e848")
                border = QColor("#18a830")
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(48, 232, 72, 60))
                painter.drawEllipse(QRectF(cx - 8, cy - 8, 16, 16))
            else:
                fill = QColor("#6a3000")
                border = QColor("#3a1800")

            painter.setPen(QPen(border, 1.0))
            painter.setBrush(fill)
            painter.drawEllipse(QRectF(cx - 5, cy - 5, 10, 10))

            painter.setPen(QColor("#808890"))
            painter.setFont(QFont("Consolas", 8))
            text_w = 84.0
            painter.drawText(
                QRectF(cx - text_w / 2, cy + 10, text_w, 14),
                Qt.AlignmentFlag.AlignHCenter,
                label,
            )
        painter.end()


class _FanStrip(QWidget):
    """Static decorative rack fans using the real rack fan item."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(132)
        self.setMaximumHeight(140)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, 300, 126)

        self._view = QGraphicsView(self._scene, self)
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setStyleSheet("background: transparent;")
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        layout.addWidget(self._view, 1)

        self._left_fan = SpinningFanItem()
        self._right_fan = SpinningFanItem()
        self._scene.addItem(self._left_fan)
        self._scene.addItem(self._right_fan)
        self._left_fan.setScale(1.6)
        self._right_fan.setScale(1.6)
        self._left_fan.setPos(102, 63)
        self._right_fan.setPos(198, 63)

        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(16)
        self._spin_timer.timeout.connect(self._tick_spin)
        self._speed_factor = 0.0
        self._spin_elapsed_ms = 0
        self._spin_accumulator = 0.0
        self._spinup_duration_ms = 1200

    def stop_spinning(self, *, reset_angle: bool = False) -> None:
        self._spin_timer.stop()
        self._speed_factor = 0.0
        self._spin_elapsed_ms = 0
        self._spin_accumulator = 0.0
        if reset_angle:
            self._left_fan._angle = 0.0
            self._right_fan._angle = 0.0
            self._left_fan.update()
            self._right_fan.update()

    def start_spinup(self) -> None:
        self._speed_factor = 0.08
        self._spin_elapsed_ms = 0
        self._spin_accumulator = 0.0
        if not self._spin_timer.isActive():
            self._spin_timer.start()

    def _tick_spin(self) -> None:
        self._spin_elapsed_ms += self._spin_timer.interval()
        if self._spin_elapsed_ms < self._spinup_duration_ms:
            t = self._spin_elapsed_ms / self._spinup_duration_ms
            self._speed_factor = 0.08 + (0.92 * t)
        else:
            self._speed_factor = 1.0

        self._spin_accumulator += self._speed_factor
        steps = int(self._spin_accumulator)
        if steps <= 0:
            return
        self._spin_accumulator -= steps
        for _ in range(steps):
            self._left_fan.advance()
            self._right_fan.advance()


class _WelcomeStep(QWidget):
    """Welcome panel with logo and boot-like intro animation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(12)
        layout.addStretch(1)

        self.logo = QLabel("")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = Path(__file__).parent / "assets" / "miniloader-logo.png"
        logo_pix = QPixmap(str(logo_path))
        if logo_pix.isNull():
            self.logo.setText("MINILOADER")
            self.logo.setStyleSheet(
                "color: #d8d8d8; font-size: 28px; font-weight: 800; background: transparent;"
            )
        else:
            self.logo.setPixmap(
                logo_pix.scaled(
                    250,
                    54,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.logo.setStyleSheet("background: transparent;")

        self.title = QLabel("")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(
            "color: #30e848; font-family: Consolas, monospace;"
            " font-size: 30px; font-weight: 800; background: transparent;"
        )
        self.subtitle = QLabel("AI in the Palm of Your Hand. Your Machine. Your Rules")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("color: #8f949c; font-size: 13px; background: transparent;")
        self.fans = _FanStrip(self)

        layout.addWidget(self.logo)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.fans)
        layout.addStretch(1)

        self._title_full = "Your Local Intelligence Rack"
        self._title_index = 0
        self._subtitle_delay = QTimer(self)
        self._subtitle_delay.setSingleShot(True)
        self._subtitle_delay.timeout.connect(self._on_final_text_ready)

        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(90)
        self._typing_timer.timeout.connect(self._type_next_char)

    def start_animation(self) -> None:
        self._title_index = 0
        self.title.setText("")
        self.subtitle.hide()
        self._subtitle_delay.stop()
        self.fans.stop_spinning(reset_angle=True)
        self.fans.start_spinup()
        self._typing_timer.start()

    def _type_next_char(self) -> None:
        if self._title_index >= len(self._title_full):
            self._typing_timer.stop()
            self._subtitle_delay.start(120)
            return
        self._title_index += 1
        self.title.setText(self._title_full[: self._title_index])

    def _on_final_text_ready(self) -> None:
        self.subtitle.show()


class _OnboardingStep(QWidget):
    """Front Door multi-panel onboarding questionnaire (3 steps)."""

    _PERSONA_CHOICES: tuple[tuple[str, str], ...] = (
        ("professional", "I want to save money and beat cloud API rate limits."),
        ("gamer", "I want AI tools for gaming, streaming, or Discord."),
        ("tinkerer", "I want to build custom, private local AI workflows."),
        ("explorer", "I'm just exploring. Show me what it can do."),
    )
    _TECH_CHOICES: tuple[tuple[str, str], ...] = (
        ("guided", "Just make it work."),
        ("intermediate", "Let me tweak the dials."),
        ("advanced", "Give me the empty rack."),
    )

    _DEFAULT_PRESET = "Local AI Agent"

    _PRESET_MATRIX: dict[tuple[str, str], str] = {
        ("professional", "guided"):       "Local AI Agent",
        ("professional", "intermediate"): "Remote Secretary",
        ("professional", "advanced"):     "blank_rack",
        ("gamer", "guided"):             "Local AI Agent",
        ("gamer", "intermediate"):        "Local AI Agent",
        ("gamer", "advanced"):            "blank_rack",
        ("tinkerer", "guided"):           "Local AI Agent",
        ("tinkerer", "intermediate"):     "Local AI Agent",
        ("tinkerer", "advanced"):         "blank_rack",
        ("explorer", "guided"):           "Local AI Agent",
        ("explorer", "intermediate"):     "Local AI Agent",
        ("explorer", "advanced"):         "Local AI Agent",
    }

    _REVEAL_COPY: dict[str, tuple[str, str, str]] = {
        "Local AI Agent": (
            "Your pre-wired AI Agent rack is ready.",
            "A local brain, AI server, agent engine, and chat terminal are staged and "
            "connected. Press the power button to boot everything.",
            "Behind the scenes: Miniloader selected a pre-wired agent stack so you can "
            "start chatting with a local AI immediately.",
        ),
        "Remote Secretary": (
            "Your AI Secretary is ready.",
            "A full agent stack with voice, file vault, knowledge engine, Google Suite "
            "tools, and web gateway — all pre-wired and waiting for power-on.",
            "Behind the scenes: Miniloader staged an advanced multi-tool agent workflow "
            "designed for productivity and hands-free operation.",
        ),
        "blank_rack": (
            "Your empty rack is powered up.",
            "Drag modules from the tray, drop them into the rack, and wire your own "
            "local AI workflow from scratch.",
            "Behind the scenes: Miniloader gave you a blank canvas with full manual "
            "control. Right-click the rack for presets if you change your mind.",
        ),
    }

    def __init__(
        self,
        *,
        vault_provider: Callable[[], VaultManager | None],
        on_finish: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vault_provider = vault_provider
        self._on_finish = on_finish
        self._persona = ""
        self._tech_level = ""
        self._persisting = False
        self._finished = False
        self.chosen_preset: str = self._DEFAULT_PRESET

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(12)

        self._progress_label = QLabel("Onboarding 1 of 3")
        self._progress_label.setStyleSheet(
            "color: #8ca0b8; font-size: 11px; letter-spacing: 0.5px; background: transparent;"
        )
        layout.addWidget(self._progress_label)

        self._stack = QStackedWidget(self)
        self._persona_panel = self._build_persona_panel()
        self._tech_panel = self._build_tech_panel()
        self._reveal_panel = self._build_reveal_panel()
        self._stack.addWidget(self._persona_panel)
        self._stack.addWidget(self._tech_panel)
        self._stack.addWidget(self._reveal_panel)
        layout.addWidget(self._stack, 1)

    def begin(self) -> None:
        """Reset sub-flow when outer onboarding step is entered."""
        self._set_substep(0)
        self._persisting = False
        self._finished = False
        self._launch_btn.setEnabled(True)
        self._launch_btn.setText("Launch Miniloader")

    def _resolve_preset(self) -> str:
        persona = self._persona or "explorer"
        tech = self._tech_level or "guided"
        return self._PRESET_MATRIX.get((persona, tech), self._DEFAULT_PRESET)

    def _build_persona_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet("background: #13161c;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Welcome to Miniloader. What's your primary goal today?")
        title.setWordWrap(True)
        title.setStyleSheet("color: #e6ebf3; font-size: 20px; font-weight: 700; background: transparent;")
        subtitle = QLabel("We'll customize your workspace based on what you want to achieve.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #a9b4c5; font-size: 13px; background: transparent;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._persona_group = QButtonGroup(self)
        self._persona_group.setExclusive(True)
        self._persona_buttons: dict[str, QPushButton] = {}
        for key, text in self._PERSONA_CHOICES:
            btn = QPushButton(text)
            btn.setProperty("cardOption", True)
            btn.setCheckable(True)
            btn.setMinimumHeight(54)
            btn.clicked.connect(lambda checked, value=key: self._on_persona_selected(value, checked))
            self._persona_group.addButton(btn)
            self._persona_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        nav = QHBoxLayout()
        self._skip_btn = QPushButton("Skip for now")
        self._skip_btn.clicked.connect(self._on_skip)
        self._persona_next_btn = QPushButton("Continue")
        self._persona_next_btn.setEnabled(False)
        self._persona_next_btn.clicked.connect(lambda: self._set_substep(1))
        nav.addWidget(self._skip_btn)
        nav.addStretch(1)
        nav.addWidget(self._persona_next_btn)
        layout.addLayout(nav)
        return panel

    def _build_tech_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet("background: #13161c;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("How do you prefer to work?")
        title.setWordWrap(True)
        title.setStyleSheet("color: #e6ebf3; font-size: 20px; font-weight: 700; background: transparent;")
        subtitle = QLabel("Don't worry, you can always change this later.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #a9b4c5; font-size: 13px; background: transparent;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._tech_group = QButtonGroup(self)
        self._tech_group.setExclusive(True)
        self._tech_buttons: dict[str, QPushButton] = {}
        for key, text in self._TECH_CHOICES:
            btn = QPushButton(text)
            btn.setProperty("cardOption", True)
            btn.setCheckable(True)
            btn.setMinimumHeight(54)
            btn.clicked.connect(lambda checked, value=key: self._on_tech_selected(value, checked))
            self._tech_group.addButton(btn)
            self._tech_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        nav = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._set_substep(0))
        self._tech_next_btn = QPushButton("Continue")
        self._tech_next_btn.setEnabled(False)
        self._tech_next_btn.clicked.connect(lambda: self._set_substep(2))
        nav.addWidget(back_btn)
        nav.addStretch(1)
        nav.addWidget(self._tech_next_btn)
        layout.addLayout(nav)
        return panel

    def _build_reveal_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet("background: #13161c;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._reveal_headline = QLabel("Your personalized launchpad is ready.")
        self._reveal_headline.setWordWrap(True)
        self._reveal_headline.setStyleSheet(
            "color: #ebf3ff; font-size: 22px; font-weight: 700; background: transparent;"
        )
        self._reveal_detail = QLabel("")
        self._reveal_detail.setWordWrap(True)
        self._reveal_detail.setStyleSheet("color: #a9b4c5; font-size: 13px; background: transparent;")
        self._reveal_background = QLabel("")
        self._reveal_background.setWordWrap(True)
        self._reveal_background.setStyleSheet(
            "background: #121722; border: 1px solid #364c72; border-radius: 6px;"
            " color: #d1dcef; padding: 10px;"
        )
        layout.addWidget(self._reveal_headline)
        layout.addWidget(self._reveal_detail)
        layout.addWidget(self._reveal_background)
        layout.addStretch(1)

        nav = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._set_substep(1))
        self._launch_btn = QPushButton("Launch Miniloader")
        self._launch_btn.setProperty("primaryCta", True)
        self._launch_btn.clicked.connect(self._handle_launch_clicked)
        nav.addWidget(back_btn)
        nav.addStretch(1)
        nav.addWidget(self._launch_btn)
        layout.addLayout(nav)
        return panel

    def _set_substep(self, idx: int) -> None:
        self._stack.setCurrentIndex(max(0, min(idx, self._stack.count() - 1)))
        current = self._stack.currentIndex()
        self._progress_label.setText(f"Onboarding {current + 1} of 3")
        if current == 2:
            self._apply_reveal_copy()

    def _on_persona_selected(self, value: str, checked: bool) -> None:
        if not checked:
            return
        self._persona = value
        self._persona_next_btn.setEnabled(True)

    def _on_tech_selected(self, value: str, checked: bool) -> None:
        if not checked:
            return
        self._tech_level = value
        self._tech_next_btn.setEnabled(True)

    def _on_skip(self) -> None:
        self.chosen_preset = self._DEFAULT_PRESET
        self._on_finish()

    def _apply_reveal_copy(self) -> None:
        self.chosen_preset = self._resolve_preset()
        headline, detail, background = self._REVEAL_COPY.get(
            self.chosen_preset,
            self._REVEAL_COPY[self._DEFAULT_PRESET],
        )
        self._reveal_headline.setText(headline)
        self._reveal_detail.setText(detail)
        self._reveal_background.setText(background)

    def _handle_launch_clicked(self) -> None:
        if self._persisting:
            return
        self._persisting = True
        self._finished = False
        self._launch_btn.setEnabled(False)
        self._launch_btn.setText("Launching...")
        self.chosen_preset = self._resolve_preset()

        self._persist_timeout = QTimer(self)
        self._persist_timeout.setSingleShot(True)
        self._persist_timeout.setInterval(5_000)
        self._persist_timeout.timeout.connect(self._force_finish)
        self._persist_timeout.start()

        vault = self._vault_provider()
        if vault is None:
            self._force_finish()
            return

        async def _persist_settings() -> None:
            store = SettingsStore(vault)
            await store.set("onboarding_frontdoor_persona", self._persona or "explorer")
            await store.set("onboarding_frontdoor_tech_level", self._tech_level or "guided")
            await store.set("onboarding_frontdoor_preset", self.chosen_preset)

        def _finish(error: Exception | None = None) -> None:
            if self._finished:
                return
            self._persist_timeout.stop()
            self._finished = True
            if error is not None:
                QMessageBox.warning(
                    self,
                    "Onboarding Save Warning",
                    f"Could not save onboarding settings, but setup will continue.\n{error}",
                )
            self._on_finish()

        def _worker() -> None:
            try:
                asyncio.run(_persist_settings())
            except Exception as exc:
                QTimer.singleShot(0, lambda err=exc: _finish(err))
                return
            QTimer.singleShot(0, _finish)

        threading.Thread(target=_worker, daemon=True, name="frontdoor-settings-save").start()

    def _force_finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._on_finish()


class OnboardingDialog(QDialog):
    """5-step onboarding flow for creating a local encrypted vault."""

    _SHELL_SS = (
        "QFrame#rackShell {"
        "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        "    stop:0 #2a2d34, stop:0.2 #1e2028, stop:1 #191b20);"
        "  border: 1.5px solid #5c636e;"
        "  border-radius: 10px;"
        "}"
    )

    def __init__(self, vault_path: Path | None = None) -> None:
        super().__init__()
        self._vault_path = vault_path
        self.vault_manager: VaultManager | None = None
        self._mnemonic = VaultManager.generate_mnemonic()
        self._confirm_indices = sorted(random.sample(range(12), k=3))
        self._animating = False

        self.setWindowTitle("Miniloader Setup")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.resize(720, 580)

        self.setStyleSheet(
            "QDialog { background: #13161c; }"
            "QLabel { color: #d8dce4; }"
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2a2e36, stop:1 #1a1e25);"
            "  border: 1px solid #626a77; border-radius: 4px; color: #d3d8df;"
            "  padding: 7px 12px; font-size: 12px;"
            "}"
            "QPushButton:hover { border-color: #84a0c0; color: #f0f6ff; }"
            "QPushButton:pressed { background: #171a20; }"
            "QPushButton[cardOption='true'] {"
            "  text-align: left; padding: 12px 14px; border-radius: 8px;"
            "  border: 1px solid #394350; background: #151b25; color: #d5deeb;"
            "}"
            "QPushButton[cardOption='true']:hover { border-color: #6a8fc6; background: #1a2230; }"
            "QPushButton[cardOption='true']:checked { border-color: #30e848; background: #122616; color: #d7ffe1; }"
            "QPushButton[primaryCta='true'] {"
            "  border: 1px solid #1c9f34; color: #f0fff4;"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2abf45, stop:1 #1b8a31);"
            "  font-weight: 700; padding: 8px 16px;"
            "}"
            "QPushButton[primaryCta='true']:hover { border-color: #53d46e; }"
            "QPushButton[primaryCta='true']:pressed { background: #19722a; }"
            "QLineEdit {"
            "  background: #0a100a; border: 1px solid #2a3a2a; border-radius: 4px;"
            "  color: #b6d9b6; padding: 5px 7px; selection-background-color: #2a4a70;"
            "}"
            "QLineEdit:focus { border-color: #39d353; }"
            "QCheckBox { color: #9ea8b8; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
            "QCheckBox::indicator:unchecked { background: #1a1e28; border: 1px solid #3b4454; }"
            "QCheckBox::indicator:checked { background: #30e848; border: 1px solid #18a830; }"
            + self._SHELL_SS
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        shell = QFrame(self)
        shell.setObjectName("rackShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(24, 22, 24, 16)
        shell_layout.setSpacing(10)

        self._screw_overlay = _ScrewOverlay(shell)
        self._screw_overlay.raise_()

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(4)
        controls_row.addStretch(1)

        self._min_btn = QPushButton("\u2500")
        self._min_btn.setFixedSize(26, 20)
        self._min_btn.setToolTip("Minimize")
        self._min_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #5a6878; border: 1px solid #2a3442;"
            "  border-radius: 4px; font-size: 12px; padding: 0;"
            "}"
            "QPushButton:hover { background: #1a212b; color: #c0d0e0; border-color: #3f5268; }"
            "QPushButton:pressed { background: #111720; }"
        )
        self._min_btn.clicked.connect(self.showMinimized)

        self._close_btn = QPushButton("\u00d7")
        self._close_btn.setFixedSize(26, 20)
        self._close_btn.setToolTip("Close")
        self._close_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #8f5b5b; border: 1px solid #3f2a2a;"
            "  border-radius: 4px; font-size: 13px; padding: 0;"
            "}"
            "QPushButton:hover { background: #26161a; color: #e67878; border-color: #6c3b3b; }"
            "QPushButton:pressed { background: #1a1012; }"
        )
        self._close_btn.clicked.connect(self.reject)

        controls_row.addWidget(self._min_btn)
        controls_row.addWidget(self._close_btn)
        shell_layout.addLayout(controls_row)

        self._stack = QStackedWidget(self)
        self._stack_fx = QGraphicsOpacityEffect(self._stack)
        self._stack_fx.setOpacity(1.0)
        self._stack.setGraphicsEffect(self._stack_fx)

        self._welcome_step = self._build_step_welcome()
        self._words_step = self._build_step_words()
        self._confirm_step = self._build_step_confirm()
        self._profile_step = self._build_step_profile()
        self._onboarding_step: _OnboardingStep = self._build_step_onboarding()

        self._stack.addWidget(self._welcome_step)
        self._stack.addWidget(self._words_step)
        self._stack.addWidget(self._confirm_step)
        self._stack.addWidget(self._profile_step)
        self._stack.addWidget(self._onboarding_step)

        shell_layout.addWidget(self._stack, 1)

        self._step_indicator = _StepIndicator(self)
        self._step_indicator.set_step(0)
        shell_layout.addWidget(self._step_indicator)

        root.addWidget(shell, 1)

        QTimer.singleShot(120, self._welcome_step.start_animation)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._screw_overlay.setGeometry(self._screw_overlay.parentWidget().rect())

    def _build_step_welcome(self) -> QWidget:
        panel = _WelcomeStep(self)
        panel_layout = panel.layout()
        assert isinstance(panel_layout, QVBoxLayout)
        row = QHBoxLayout()
        row.addStretch(1)
        next_btn = QPushButton("Get Started")
        next_btn.clicked.connect(lambda: self._go_to_step(1))
        row.addWidget(next_btn)
        panel_layout.addLayout(row)
        return panel

    def _build_step_words(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(12)

        warning = QLabel(
            "Write these 12 words down. This is your only way to recover"
            " your account if you lose this PC."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "background: #241d10; border: 1px solid #8a6924;"
            " color: #f0d9a2; padding: 10px; border-radius: 6px;"
        )
        layout.addWidget(warning)

        words_frame = QFrame(panel)
        words_frame.setStyleSheet(
            "QFrame { background: #0a100a; border: 1px solid #2a3a2a; border-radius: 8px; }"
        )
        grid = QGridLayout(words_frame)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)

        words = self._mnemonic.split()
        for i, word in enumerate(words):
            lbl = QLabel(f"{i + 1:02d}. {word}")
            lbl.setStyleSheet(
                "color: #30e848; font-family: Consolas, monospace;"
                " font-size: 13px; background: transparent;"
            )
            grid.addWidget(lbl, i // 3, i % 3)
        layout.addWidget(words_frame)

        layout.addStretch(1)
        nav = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._go_to_step(0))
        next_btn = QPushButton("I Have Written These Down")
        next_btn.clicked.connect(lambda: self._go_to_step(2))
        nav.addWidget(back_btn)
        nav.addStretch(1)
        nav.addWidget(next_btn)
        layout.addLayout(nav)
        return panel

    def _build_step_confirm(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(12)

        msg = QLabel("Quick check: enter the requested words to confirm your backup.")
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #b8c4d8; font-size: 13px; background: transparent;")
        layout.addWidget(msg)

        form = QFormLayout()
        self._confirm_inputs: list[QLineEdit] = []
        for idx in self._confirm_indices:
            entry = QLineEdit()
            entry.setPlaceholderText(f"Word #{idx + 1}")
            form.addRow(f"Word #{idx + 1}", entry)
            self._confirm_inputs.append(entry)
        layout.addLayout(form)

        self._skip_quick_check = QCheckBox("Skip this")
        self._skip_quick_check.setStyleSheet("color: #9aa8bc; padding-top: 4px;")
        layout.addWidget(self._skip_quick_check)

        skip_warning = QLabel(
            "If these words are lost and you forget your password, Miniloader cannot recover your account."
        )
        skip_warning.setWordWrap(True)
        skip_warning.setStyleSheet(
            "color: #f0d9a2; background: #241d10; border: 1px solid #8a6924;"
            " border-radius: 6px; padding: 8px;"
        )
        layout.addWidget(skip_warning)

        self._confirm_error = QLabel("")
        self._confirm_error.setStyleSheet("color: #ff9f9f; background: transparent;")
        layout.addWidget(self._confirm_error)
        layout.addStretch(1)

        nav = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._go_to_step(1))
        next_btn = QPushButton("Confirm Backup")
        next_btn.clicked.connect(self._validate_backup_words)
        nav.addWidget(back_btn)
        nav.addStretch(1)
        nav.addWidget(next_btn)
        layout.addLayout(nav)
        return panel

    def _build_step_profile(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(12)

        msg = QLabel("Create your local profile and a password to encrypt your vault.")
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #b8c4d8; font-size: 13px; background: transparent;")
        layout.addWidget(msg)

        form = QFormLayout()
        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Username")
        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_input.setPlaceholderText("Password (min 8 chars)")
        self._password_confirm_input = QLineEdit()
        self._password_confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_confirm_input.setPlaceholderText("Confirm password")
        self._autologin_checkbox = QCheckBox("Enable Auto-Login on this PC")
        self._autologin_checkbox.setChecked(True)

        form.addRow("Username", self._username_input)
        form.addRow("Password", self._password_input)
        form.addRow("Confirm", self._password_confirm_input)
        layout.addLayout(form)
        layout.addWidget(self._autologin_checkbox)

        self._profile_error = QLabel("")
        self._profile_error.setStyleSheet("color: #ff9f9f; background: transparent;")
        layout.addWidget(self._profile_error)
        layout.addStretch(1)

        nav = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._go_to_step(2))
        create_btn = QPushButton("Create Account")
        create_btn.clicked.connect(self._create_account)
        nav.addWidget(back_btn)
        nav.addStretch(1)
        nav.addWidget(create_btn)
        layout.addLayout(nav)
        return panel

    @property
    def chosen_preset(self) -> str:
        return self._onboarding_step.chosen_preset

    def _build_step_onboarding(self) -> _OnboardingStep:
        return _OnboardingStep(
            vault_provider=lambda: self.vault_manager,
            on_finish=self.accept,
            parent=self,
        )

    def _go_to_step(self, idx: int) -> None:
        if self._animating or idx == self._stack.currentIndex():
            return
        self._animating = True
        fade_out = QPropertyAnimation(self._stack_fx, b"opacity", self)
        fade_out.setDuration(160)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        fade_in = QPropertyAnimation(self._stack_fx, b"opacity", self)
        fade_in.setDuration(180)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _on_fade_out_finished() -> None:
            self._stack.setCurrentIndex(idx)
            self._step_indicator.set_step(idx)
            self._step_indicator.setVisible(idx < 4)
            if idx == 0:
                self._welcome_step.start_animation()
            elif idx == 4:
                self._onboarding_step.begin()
            fade_in.start()

        def _on_fade_in_finished() -> None:
            self._animating = False

        fade_out.finished.connect(_on_fade_out_finished)
        fade_in.finished.connect(_on_fade_in_finished)
        self._fade_out_anim = fade_out
        self._fade_in_anim = fade_in
        fade_out.start()

    def _validate_backup_words(self) -> None:
        if getattr(self, "_skip_quick_check", None) is not None and self._skip_quick_check.isChecked():
            self._confirm_error.clear()
            self._go_to_step(3)
            return
        words = self._mnemonic.split()
        for field, idx in zip(self._confirm_inputs, self._confirm_indices, strict=False):
            if field.text().strip().lower() != words[idx]:
                self._confirm_error.setText(
                    "One or more words do not match. Please try again."
                )
                return
        self._confirm_error.clear()
        self._go_to_step(3)

    def _create_account(self) -> None:
        username = self._username_input.text().strip()
        password = self._password_input.text()
        confirm = self._password_confirm_input.text()
        if password != confirm:
            self._profile_error.setText("Passwords do not match.")
            return
        try:
            self.vault_manager = VaultManager.create(
                mnemonic=self._mnemonic,
                username=username,
                password=password,
                save_to_keyring=self._autologin_checkbox.isChecked(),
                vault_path=self._vault_path,
            )
        except VaultValidationError as exc:
            self._profile_error.setText(str(exc))
            return
        except VaultError as exc:
            QMessageBox.critical(self, "Vault Error", str(exc))
            return

        self.vault_manager.ensure_user_data_dir()
        self._profile_error.clear()
        self._go_to_step(4)
