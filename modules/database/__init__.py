"""database — SQLite persistent storage for chat histories and threads."""
from modules.database.logic import DatabaseModule, register

__all__ = ["DatabaseModule", "register"]
