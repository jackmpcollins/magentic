from magentic import *


def get_weather(city: str) -> str:
    return f"The weather in {city} is 20°C."


@prompt_chain(
    "First, please greet me by calling me Mr. Geng. Then, ask for the weather information for {cities}. After answering, thank you for your inquiry!",
    functions=[get_weather],
    model=OpenaiChatModel("model", api_key="your_apikey", base_url="baseurl")
)
def describe_weather(cities: list[str]) -> StreamedResponse: ...

res = describe_weather(["New York"])

for item in res:
    if isinstance(item, StreamedStr):
        print(str(item))
    if isinstance(item, FunctionCall):
        print(item)
        
