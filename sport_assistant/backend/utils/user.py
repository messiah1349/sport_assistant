from dataclasses import dataclass


@dataclass
class User:
    user_id: int
    api_football_token: str

