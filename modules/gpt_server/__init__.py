"""gpt_server — OpenAI-compatible local REST API bridge."""
from modules.gpt_server.logic import GptServerModule, register

__all__ = ["GptServerModule", "register"]
