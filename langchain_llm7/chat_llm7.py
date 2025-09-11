"""LangChain integration for LLM7's chat API.

This module provides a ChatLLM7 class that implements the BaseChatModel interface,
allowing seamless integration with LangChain's chat ecosystem.

Example:
    >>> from langchain_core.messages import HumanMessage, SystemMessage
    >>> chat = ChatLLM7(model_name="gpt-4o-mini-2024-07-18")
    >>> messages = [
    ...     SystemMessage(content="You are a helpful assistant"),
    ...     HumanMessage(content="What's the weather in London?")
    ... ]
    >>> response = chat.invoke(messages)
    >>> print(response.content)
"""

from typing import Any, Dict, Iterator, List, Optional, Union
import json
import requests
import tokeniser

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, model_validator



class ChatLLM7(BaseChatModel):
    """Chat interface for LLM7's API compatible with LangChain.

    Attributes:
        base_url (str): Base URL for the LLM7 API. Defaults to "https://api.llm7.io/v1".
        model_name (str): Model name to use. Defaults to "gpt-4o-mini-2024-07-18".
        temperature (float): Sampling temperature between 0 and 2. Defaults to 1.0.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        timeout (int): Timeout in seconds for API requests. Defaults to 120.
        max_retries (int): Maximum number of retries for API calls. Defaults to 3.
        stop (Optional[List[str]]): List of stop sequences to halt generation.
        streaming (bool): Whether to stream responses. Defaults to False.

    Example:
        Basic usage with LangChain:
        ```python
        from langchain.chains import LLMChain
        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a poetic assistant"),
            ("human", "Write me a poem about {topic}")
        ])
        chain = LLMChain(llm=ChatLLM7(), prompt=prompt)
        print(chain.run(topic="quantum physics"))
        ```
    """

    base_url: str = Field(default="https://api.llm7.io/v1")
    model_name: str = Field(default="gpt-4o-mini-2024-07-18", alias="model")
    temperature: Optional[float] = Field(default=1.0)
    max_tokens: Optional[int] = None
    timeout: int = Field(default=120)
    max_retries: int = Field(default=3)
    stop: Optional[List[str]] = None
    streaming: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_environment(self) -> "ChatLLM7":
        """Validate configuration and environment setup.

        Note:
            LLM7 currently doesn't require an API key, but this method is maintained
            for future compatibility.

        Returns:
            The validated ChatLLM7 instance.
        """
        return self

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to LLM7 API format.

        Args:
            messages: List of LangChain message objects

        Returns:
            List of messages in LLM7 API format

        Raises:
            ValueError: If unsupported message type is encountered
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _create_payload(self, messages: List[BaseMessage], stream: bool) -> Dict[str, Any]:
        """Construct API request payload.

        Args:
            messages: List of formatted messages
            stream: Whether the payload is for streaming

        Returns:
            Dictionary containing the complete request payload
        """
        payload = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "stream": stream,
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.stop:
            payload["stop"] = self.stop

        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun]= None,
        ** kwargs: Any,
    ) -> ChatResult:
        """Execute non-streaming chat completion.

        Args:
            messages: Input messages for the conversation
            stop: Optional list of stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional keyword arguments

        Returns:
            ChatResult containing generated response

        Raises:
            ValueError: If API request fails
        """
        payload = self._create_payload(messages, stream=False)
        url = f"{self.base_url}/chat/completions"

        # Handle stop sequences
        stop = stop or self.stop
        if stop:
            payload["stop"] = stop

        text = ""
        for message in messages:
            try:
                text += message.content + "\n"
            except Exception:
                pass
        input_tokens = tokeniser.estimate_tokens(text)

        response = requests.post(
            url,
            json = payload,
            timeout = self.timeout,
            headers = {"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

        response_data = response.json()

        try:
            if response_data["error"]["message"]:
                raise ValueError(f"API error: {response_data['error']['message']}")
        except KeyError:
            pass


        content = response_data["choices"][0]["message"]["content"]

        output_tokens = tokeniser.estimate_tokens(content)
        total_tokens = input_tokens + output_tokens
        message = AIMessage(
            content=content,
            usage_metadata=UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        )

        return ChatResult(generations=[ChatGeneration(message=message)])


    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun]= None,
    ** kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Handle streaming chat completions.

        Args:
            messages: Input messages for the conversation
            stop: Optional list of stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional keyword arguments

        Yields:
            ChatGenerationChunk objects for each response chunk

        Raises:
            ValueError: If API request fails
        """
        payload = self._create_payload(messages, stream=True)
        url = f"{self.base_url}/chat/completions"

        # Handle stop sequences
        stop = stop or self.stop
        if stop:
            payload["stop"] = stop

        with requests.post(
            url,
            json = payload,
            timeout = self.timeout,
            headers = {"Content-Type": "application/json"},
            stream = True
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

            content_buffer = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        chunk_data = decoded_line[6:].strip()
                        if chunk_data == "[DONE]":
                            break

                        try:
                            json_chunk = json.loads(chunk_data)
                            delta = json_chunk["choices"][0]["delta"]
                            content = delta.get("content", "")

                            if content:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=content),
                                    generation_info=delta,
                                )

                                if run_manager:
                                    run_manager.on_llm_new_token(content, chunk=chunk)

                                yield chunk
                        except json.JSONDecodeError:
                            continue


    @property
    def _llm_type(self) -> str:
        """Identifier for the LLM type.

        Returns:
            str: Always returns "llm7-chat"
        """
        return "llm7-chat"


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters for the model.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url
        }