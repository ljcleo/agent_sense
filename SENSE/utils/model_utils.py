import copy
import sys
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import tiktoken
from termcolor import colored

from autogen import token_count_utils
from autogen.cache import AbstractCache, Cache
from autogen.types import MessageContentType
from autogen.agentchat.contrib.capabilities import transforms_util

class TextMessageTruncate:
    """A transform for truncating an incomplete message."""

    def __init__(
        self,
        trunc_symbol='\n'
    ):
        # Track the number of messages changed for logging
        self.trunc_symbol = trunc_symbol
        self._messages_changed = 0

    def apply_transform(self, messages: List[Dict]):
        """Applies the truncation change.

        Args:
            messages (List[Dict]): A list of message dictionaries.

        Returns:
            List[Dict]: A list of dictionaries with the message content updated.
        """
        # Make sure there is at least one message
        if not messages:
            return messages
        messages_changed = 0
        processed_messages = copy.deepcopy(messages)
        for message in processed_messages:
            # Some messages may not have content.
            if not transforms_util.is_content_right_type(
                message.get("content")
            ) or not transforms_util.is_content_right_type(message.get("name")):
                continue

            if transforms_util.is_content_text_empty(message["content"]) or transforms_util.is_content_text_empty(
                message["name"]
            ):
                continue

            # Get and format the name in the content
            content = message["content"]
            if len(content.split(self.trunc_symbol))>1:
                content = content.split(self.trunc_symbol)[0]

                message["content"] = content
                messages_changed += 1
        self._messages_changed = messages_changed
        return processed_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        if self._messages_changed > 0:
            return f"{self._messages_changed} message(s) changed to truncate excess paragraphs.", True
        else:
            return "No messages changed to truncate excess paragraphs.", False
