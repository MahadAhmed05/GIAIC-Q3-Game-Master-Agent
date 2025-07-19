import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import random

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Gemini-compatible client ===
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Tool Functions ===

def roll_dice(sides: int = 20) -> int:
    return random.randint(1, sides)

def generate_event(context: str) -> str:
    events = {
        "forest": [
            "You hear rustling in the bushes. A goblin appears!",
            "You find an ancient tree with glowing runes.",
            "A traveling merchant offers you a mysterious potion."
        ],
        "dungeon": [
            "A trap triggers beneath your feet!",
            "A skeleton warrior blocks your path.",
            "You discover a chest filled with gold... or is it a mimic?"
        ],
        "village": [
            "A child runs up to you, asking for help.",
            "The blacksmith offers to upgrade your weapon.",
            "You overhear talk of a dragon nearby."
        ]
    }
    return random.choice(events.get(context.lower(), ["Nothing unusual happens..."]))

# === Agents ===

NarratorAgent = Agent(
    name="NarratorAgent",
    instructions="Narrate the fantasy adventure based on player decisions. Use vivid descriptions and advance the story."
)

MonsterAgent = Agent(
    name="MonsterAgent",
    instructions="Control monster behavior during combat. Ask the user what action they take (attack, defend, run), then narrate outcome using dice roll.",
    tools={"roll_dice": roll_dice}
)

ItemAgent = Agent(
    name="ItemAgent",
    instructions="Describe items found by the player and manage inventory. Assign rewards after events or combat.",
    tools={"generate_event": generate_event}
)

# === Chat Start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("current_agent", NarratorAgent)
    await cl.Message(content="🧙 Welcome, adventurer! Your quest begins now...\n\nTell me what you'd like to do — explore a forest, enter a dungeon, or visit a village?").send()

# === Message Handling ===
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    user_input = message.content.lower()

    # Agent Handoff Logic
    if any(word in user_input for word in ["attack", "defend", "monster", "fight", "battle"]):
        agent = MonsterAgent
    elif any(word in user_input for word in ["item", "chest", "reward", "loot", "inventory"]):
        agent = ItemAgent
    else:
        agent = NarratorAgent

    cl.user_session.set("current_agent", agent)

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Manual tool trigger for ItemAgent (event generator)
        if agent == ItemAgent:
            context = None
            for area in ["forest", "dungeon", "village"]:
                if area in user_input:
                    context = area
                    break

            if context:
                event = generate_event(context)
                await msg.update(content=f"🎁 You discover:\n\n{event}")
                history.append({"role": "assistant", "content": msg.content})
                cl.user_session.set("chat_history", history)
                return

        # Manual tool trigger for MonsterAgent (dice roller)
        if agent == MonsterAgent:
            roll = roll_dice()
            outcome = "🗡️ Critical Hit!" if roll > 15 else "💢 Weak strike..." if roll < 5 else "⚔️ You strike the enemy."
            await msg.update(content=f"You rolled a {roll}.\n{outcome}")
            history.append({"role": "assistant", "content": msg.content})
            cl.user_session.set("chat_history", history)
            return

        # Run streamed response for all other agents
        result = Runner.run_streamed(agent, history, run_config=cast(RunConfig, config))
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                await msg.stream_token(event.data.delta)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")
        print(f"Error: {e}")
