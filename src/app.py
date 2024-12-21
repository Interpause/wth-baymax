"""Main app."""

import asyncio
import logging
import os
import secrets
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .model import AudioCache, ModelManager, Msgs

load_dotenv()


__all__ = ["create_app"]
log = logging.getLogger(__name__)


def create_app():
    """App factory.

    Creating the app within a function prevents mishaps if using multiprocessing.
    """
    model_path = os.getenv("MODEL_PATH")
    audio_cache_path = os.getenv("AUDIO_CACHE_PATH")
    token = os.getenv("TOKEN") or uuid4().hex
    Path(audio_cache_path).mkdir(parents=True, exist_ok=True)
    assert Path(model_path).exists(), "Model path does not exist."

    log.info(f"User: user, Token: {token}")
    security = HTTPBasic()

    def _require_login(creds: HTTPBasicCredentials = Depends(security)):
        if secrets.compare_digest(
            creds.password.encode("utf-8"), token.encode("utf-8")
        ):
            return True
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )

    app = FastAPI(dependencies=[Depends(security)])
    audio_cache = AudioCache(audio_cache_path)
    model_manager = ModelManager(model_path, audio_cache)

    @app.get("/hello")
    async def hello(_=Depends(_require_login)):
        """Returns a greeting.

        Returns:
            dict: A greeting message.
        """
        log.warn("zzz... 1 more second...")
        await asyncio.sleep(1)
        log.info("...zzz... oh wha...?!")
        return {"message": "Hello, World!"}

    @app.post("/upload_audio/")
    async def upload_audio(file: UploadFile, _=Depends(_require_login)):
        """Uploads an audio clip."""
        audio_id = await asyncio.to_thread(audio_cache.save, file.file)
        return {"audio_id": audio_id}

    @app.post("/generate")
    async def generate(msgs: Msgs, _=Depends(_require_login)):
        """Generates audio from text."""
        start = time.monotonic()
        resp = await asyncio.to_thread(model_manager.generate, msgs)
        dur = time.monotonic() - start
        return {"response": resp, "duration": dur}

    return app
