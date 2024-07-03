import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import InputMediaAudio
from openai_async import AsyncClient, AsyncAssistant
from whisper_api import WhisperAPI
from redis import asyncio as aioredis
from pydantic import BaseSettings
from loguru import logger

# Load environment variables
load_dotenv()

# PydanticSettings to load environment variables
class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN")
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    WHISPER_API_KEY: str = os.environ.get("WHISPER_API_KEY")
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.environ.get("REDIS_PORT", 6379))

settings = Settings()

# Initialize the Telegram bot, OpenAI API client, Whisper API client, and Redis connection
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
openai = AsyncClient(settings.OPENAI_API_KEY)
assistant = openai.assistant
whisper = WhisperAPI(settings.WHISPER_API_KEY)
redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

# Function to get a unique key for Redis caching
def get_redis_key(message_id: int) -> str:
    return f"openai_response_{message_id}"

# Function to voice the text using OpenAI TTS API
async def voice_text(text: str) -> InputMediaAudio:
    try:
        response = await openai.audio.speech.create(model="tts-1", text=text)
        return InputMediaAudio(media=response["audio_content"], caption=text)
    except Exception as e:
        logger.error(f"Error voicing text: {e}")
        return None

# Function to get answers from OpenAI Assistant API
async def get_answer(question: str) -> str:
    redis_key = get_redis_key(hash(question))
    cached_response = await redis.get(redis_key)

    if cached_response:
        return cached_response.decode("utf-8")

    try:
        response = await assistant.create(
            model="gpt-3.5-turbo-0125", input=question, max_tokens=1024
        )
        answer = response.output.text

        # Cache the response in Redis
        await redis.set(redis_key, answer, ex=3600)  # Cache for 1 hour

        return answer
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        return "Sorry, I couldn't understand that."

# Command handler for '/start' command
@dp.message(Command("start"))
async def start(message: types.Message, command: CommandObject):
    try:
        await message.reply("Hi! Send me a voice message to get started.")
    except Exception as e:
        logger.error(f"Error sending start message: {e}")

# Message handler for voice messages
@dp.message(types.Voice)
async def handle_voice(message: types.Message):
    try:
        voice_file = await bot.get_file(message.voice.file_id)
        voice_path = voice_file.file_path

        # Transcribe the voice message using Whisper API
        try:
            transcription = await whisper.transcribe(voice_path)
            question = transcription["text"]
        except Exception as e:
            logger.error(f"Error transcribing voice message: {e}")
            await message.reply("Sorry, I couldn't transcribe your voice message.")
            return

        # Get the answer to the question from OpenAI Assistant API
        answer = await get_answer(question)

        # Voice the answer using OpenAI TTS API
        voice = await voice_text(answer)

        if voice:
            # Send the voice response to the user
            await message.answer_audio(voice.media, caption=answer)
        else:
            await message.reply("Sorry, I couldn't voice the answer.")
    except Exception as e:
        logger.error(f"Error handling voice message: {e}")

if __name__ == "__main__":
    logger.add("debug.log", level="DEBUG")
    asyncio.run(dp.start_polling(bot))
