"""The AI provider boundary.

Everything inside this package may know that Ollama speaks ``/api/chat`` and
that Gemini wants an ``x-goog-api-key`` header. Nothing outside it does.

The rest of the service depends on :class:`~app.providers.base.AIProvider` and
on the neutral types beside it. No vendor response object, vendor exception or
vendor SDK type crosses this line, which is what makes adding Anthropic or
OpenAI later a new file here and nothing else.
"""
