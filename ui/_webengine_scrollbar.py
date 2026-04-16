"""Inject dark scrollbar CSS into QWebEngineView pages to match the app theme."""

from __future__ import annotations

_SCROLLBAR_CSS = r"""
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0c0e14;
}
::-webkit-scrollbar-thumb {
    background: #1e2636;
    border-radius: 4px;
    border: 1px solid #141a24;
}
::-webkit-scrollbar-thumb:hover {
    background: #2a3648;
}
::-webkit-scrollbar-corner {
    background: #0c0e14;
}
"""

_INJECT_JS = (
    "(function(){"
    "  var id='_ml_scrollbar_css';"
    "  if(document.getElementById(id)) return;"
    "  var s=document.createElement('style');"
    "  s.id=id;"
    "  s.textContent=" + repr(_SCROLLBAR_CSS) + ";"
    "  (document.head||document.documentElement).appendChild(s);"
    "})()"
)


def inject_dark_scrollbar_css(view) -> None:
    """Register a script on *view* that injects dark scrollbar CSS on every page load."""
    try:
        from PySide6.QtWebEngineCore import QWebEngineScript
    except ImportError:
        return

    script = QWebEngineScript()
    script.setName("miniloader_dark_scrollbar")
    script.setSourceCode(_INJECT_JS)
    script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
    script.setWorldId(QWebEngineScript.ScriptWorldId.ApplicationWorld)
    script.setRunsOnSubFrames(True)

    scripts = view.page().scripts()
    for existing in scripts.find("miniloader_dark_scrollbar"):
        scripts.remove(existing)
    scripts.insert(script)
