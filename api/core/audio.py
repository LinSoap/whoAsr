"""Audio processing for speech recognition API 111"""

import numpy as np
import base64
from .config import DTYPE, logger, debug_logger


def decode_audio_chunk(audio_data: str) -> np.ndarray:
    """Decode base64 encoded audio data to numpy array"""
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_data)
        debug_logger.debug(f"Decoded audio bytes: {len(audio_bytes)}")

        # Ensure we have the right number of bytes for float32
        if len(audio_bytes) % 4 != 0:
            # Pad or trim to make it divisible by 4
            padding = 4 - (len(audio_bytes) % 4)
            if padding < 4:
                audio_bytes += b"\x00" * padding
            else:
                audio_bytes = audio_bytes[: len(audio_bytes) - (len(audio_bytes) % 4)]
            debug_logger.debug(f"Adjusted audio bytes to: {len(audio_bytes)}")

        # Try float32 first (most likely after our client fixes)
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            # Check if the data looks reasonable
            if np.any(np.abs(audio_array) > 10):
                logger.warning(
                    "Audio data has very large values, might be incorrectly decoded"
                )

            return audio_array
        except Exception as e:
            debug_logger.debug(f"Float32 decode failed: {e}")

        # Try int16 and convert to float32
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = (
                audio_array.astype(np.float32) / 32768.0
            )  # Normalize to [-1, 1]

            return audio_array
        except Exception as e:
            debug_logger.debug(f"Int16 decode failed: {e}")

        # Try int32 and convert to float32
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int32)
            audio_array = (
                audio_array.astype(np.float32) / 2147483648.0
            )  # Normalize to [-1, 1]

            return audio_array
        except Exception as e:
            debug_logger.debug(f"Int32 decode failed: {e}")

        logger.error(
            f"Failed to decode audio chunk with all formats, data length: {len(audio_bytes)}"
        )
        return np.array([], dtype=DTYPE)

    except Exception as e:
        logger.error(f"Error decoding audio chunk: {e}")
        return np.array([], dtype=DTYPE)


def process_audio_chunk(model, audio_chunk: np.ndarray, cache: dict, chunk_size: list):
    """Process audio chunk with FunASR model"""
    try:
        # Ensure speech_chunk is a numpy array
        if not isinstance(audio_chunk, np.ndarray):
            logger.error(f"Speech chunk is not numpy array: {type(audio_chunk)}")
            return None

        # Import here to avoid circular imports
        from .config import ENCODER_CHUNK_LOOK_BACK, DECODER_CHUNK_LOOK_BACK

        # Process with FunASR
        res = model.generate(
            input=audio_chunk,
            cache=cache,
            is_final=False,  # Always False for streaming
            chunk_size=chunk_size,
            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
        )

        if res and len(res) > 0 and "text" in res[0]:
            result_text = res[0]["text"]
            if result_text.strip():  # Only return non-empty results
                return result_text
            else:
                debug_logger.debug("Empty result, not returning")
        else:
            debug_logger.debug("No valid result from FunASR")

        return None

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        raise


def process_final_audio(model, audio_buffer: np.ndarray, cache: dict, chunk_size: list):
    """Process remaining audio buffer when recording stops"""
    try:
        if len(audio_buffer) == 0:
            return None

        # Import here to avoid circular imports
        from .config import ENCODER_CHUNK_LOOK_BACK, DECODER_CHUNK_LOOK_BACK

        # Process remaining audio
        res = model.generate(
            input=audio_buffer,
            cache=cache,
            is_final=True,  # Final chunk
            chunk_size=chunk_size,
            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
        )

        if res and len(res) > 0 and "text" in res[0]:
            result_text = res[0]["text"]
            if result_text.strip():
                return result_text

        return None

    except Exception as e:
        logger.error(f"Error processing final audio: {e}")
        raise
