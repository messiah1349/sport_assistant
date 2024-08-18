from enum import Enum
import openai
from pydantic import BaseModel
from typing import Optional, Union

from openai import OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
client = OpenAI()

system_content = """
You are part of sport application and you should transform user query to api calls. Lower list of existed api methods:
    1) player_stat(player_name, team=None, league=None, season=None) - get statistic about specific football player in specific season
    2) team_stat(team, league=None, season=None) - get statistic about specific team in specific season

You should understand what does User want and return suitable api call with parameters.

examples: 
    query - show me statistic of Pavel Nedved in season 2003 in seria A. Your response - player_stat(player_name="Pavel Nedved", league="Seria A", season=2003)
    query - get statistic of bayern Munich in season 2022-2023. Your response - team_stat(team="Bayern Munich", season="2022-2023")

Give only api name and list of passed parameters as in examples. I will parse your response, so don't add some extra information. If you don't know what do you need to return just return "Error"
"""

system_content = """
You are part of sport application and you should transform user query to api calls. Lower list of existed api methods:
    1) player_stat(player_name, team=None, league=None, season=None) - get statistic about specific football player in specific season
    2) team_stat(team, league=None, season=None) - get statistic about specific team in specific season

You should understand what does User want and return suitable method with parameters.

examples: 
    query - show me statistic of Pavel Nedved in season 2003 in seria A. Your response - player_stat(player_name="Pavel Nedved", league="Seria A", season=2003)
    query - get statistic of bayern Munich in season 2022-2023. Your response - team_stat(team="Bayern Munich", season="2022-2023")

"""

queries = [
    "give me a statistic of Kevin De Bruyne from Manchester City in season 2022 in England Premier League",
    "Joe Gomes, season 2021-2023",
    "show me statistic of Barcelona in current season",
    "I like diego maradonna",
    "Juventus season 2023",
]


class PlayerParameters(BaseModel):
    player_name: str
    team: Optional[str]
    league: Optional[str]
    season: Optional[str]


class TeamParameters(BaseModel):
    team: str
    league: Optional[str]
    season: Optional[str]


class ResponseType(str, Enum):
    player = "player_stat"
    team = "team_stat"


class ModelResponse(BaseModel):
    method_type: ResponseType
    parameters: Union[PlayerParameters, TeamParameters]


def call_gpt(user_prompt: str) -> ParsedChatCompletion:

    completion = client.beta.chat.completions.parse(
      model="gpt-4o-mini-2024-07-18",
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt}
      ],
      tools=[
        openai.pydantic_function_tool(ModelResponse)
      ],
    )

    return completion

for query in queries:
    response = call_gpt(query)
    choice = response.choices[0]
    print("-------------------")
    if choice.finish_reason == "stop":
        print(f"STOP!!! {choice.message.content}")
    elif choice.finish_reason == "tool_calls":
        print(f"PARSED!!! {choice.message.tool_calls[0].function.parsed_arguments}")
    # parsed_response = response.choices[0].message.tool_calls[0].function.parsed_arguments
    # print(f"{query=}\n{parsed_response=}\n\n\n")
    # print(f"{query=}\n{response=}\n\n\n")

