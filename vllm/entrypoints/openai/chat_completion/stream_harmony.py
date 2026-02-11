# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony-specific streaming delta extraction for chat completions.

This module handles the extraction of DeltaMessage objects from
harmony parser state during streaming chat completions.
"""

from typing import NamedTuple

from openai_harmony import StreamableParser

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


class TokenState(NamedTuple):
    channel: str | None
    recipient: str | None
    text: str


def extract_harmony_streaming_delta(
    harmony_parser: StreamableParser,
    token_states: list[TokenState],
    prev_recipient: str | None,
    include_reasoning: bool,
    tool_choice_none: bool = False,
) -> tuple[DeltaMessage | None, bool]:
    """
    Extract a DeltaMessage from harmony parser state during streaming.

    Args:
        harmony_parser: The StreamableParser instance tracking parse state
        token_states: List of TokenState tuples for each token
        prev_recipient: Previous recipient for detecting tool call transitions
        include_reasoning: Whether to include reasoning content

    Returns:
        A tuple of (DeltaMessage or None, tools_streamed_flag)
    """

    if not token_states:
        return None, False

    tools_streamed = False

    # Group consecutive tokens with same channel/recipient
    groups: list[TokenState] = []

    current_channel = token_states[0].channel
    current_recipient = token_states[0].recipient
    current_text = token_states[0].text

    for i in range(1, len(token_states)):
        state = token_states[i]
        if state.channel == current_channel and state.recipient == current_recipient:
            current_text += state.text
        else:
            groups.append(TokenState(current_channel, current_recipient, current_text))
            current_channel = state.channel
            current_recipient = state.recipient
            current_text = state.text

    groups.append(TokenState(current_channel, current_recipient, current_text))

    # Process each group and create delta messages
    delta_message = None
    combined_content = ""
    combined_reasoning = ""
    tool_messages = []
    content_encountered = False

    # Calculate base_index: count completed tool call messages.
    # If prev_recipient indicates an ongoing function call and the parser
    # has already moved that message into its completed list, exclude it
    # so the ongoing call keeps its original index.
    base_index = 0
    func_messages_in_parser = []
    for msg in harmony_parser.messages:
        if (
            (msg.channel == "commentary" or msg.channel == "analysis")
            and msg.recipient
            and msg.recipient.startswith("functions.")
        ):
            func_messages_in_parser.append(msg)
            base_index += 1

    # If prev_recipient is an ongoing function call and the last function
    # message in the parser matches it, the parser completed that message
    # mid-stream.  Don't count it as a finished prior call.
    if (
        prev_recipient
        and prev_recipient.startswith("functions.")
        and func_messages_in_parser
        and func_messages_in_parser[-1].recipient == prev_recipient
    ):
        base_index = max(0, base_index - 1)

    # If there's an ongoing tool call from previous chunk,
    # the next new tool call starts at base_index + 1
    if prev_recipient and prev_recipient.startswith("functions."):
        next_tool_index = base_index + 1
        # Ongoing call is at base_index
        ongoing_tool_index = base_index
    else:
        # No ongoing call, next new call is at base_index
        next_tool_index = base_index
        ongoing_tool_index = None

    for group in groups:
        if group.channel == "final":
            combined_content += group.text
            content_encountered = True
        elif (
            (group.channel == "commentary" or group.channel == "analysis")
            and group.recipient
            and group.recipient.startswith("functions.")
            and not tool_choice_none
        ):
            opened_new_call = False
            tool_name = group.recipient.split("functions.", 1)[1]
            # Parser can emit recipient="functions." (empty name) for tokens in the
            # middle of the same tool call's arguments, creating a new group and
            # making prev_recipient != group.recipient. Treat that as continuation,
            # not a new tool call, so one logical call is not split into two.
            is_continuation_glitch = (
                not tool_name
                and prev_recipient
                and prev_recipient.startswith("functions.")
                and tool_messages
            )
            if not is_continuation_glitch and prev_recipient != group.recipient:
                # New tool call - emit the opening message
                tool_messages.append(
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ),
                        index=next_tool_index,
                    )
                )
                opened_new_call = True
                prev_recipient = group.recipient
                # Increment for subsequent new tool calls
                next_tool_index += 1

            if group.text:
                if opened_new_call:
                    # We just created the opening entry in tool_messages;
                    # append arguments directly to it instead of creating
                    # a second DeltaToolCall with the same index.
                    tool_messages[-1].function.arguments = group.text
                elif tool_messages:
                    # There is already a tool call entry in this chunk.
                    # Append arguments to it rather than creating a
                    # duplicate entry with the same index.
                    existing = tool_messages[-1]
                    if existing.function.arguments is None:
                        existing.function.arguments = group.text
                    else:
                        existing.function.arguments += group.text
                else:
                    # First arguments-only entry in this chunk (continuing
                    # a tool call that was opened in a previous chunk).
                    tool_call_index = (
                        ongoing_tool_index
                        if ongoing_tool_index is not None
                        else base_index
                    )
                    tool_messages.append(
                        DeltaToolCall(
                            index=tool_call_index,
                            function=DeltaFunctionCall(arguments=group.text),
                        )
                    )
        elif group.channel == "commentary":
            # Tool call preambles meant to be shown to the user
            combined_content += group.text
            content_encountered = True
        elif group.channel == "analysis" and include_reasoning:
            combined_reasoning += group.text

    # Combine all non-empty fields into a single message
    if content_encountered or combined_reasoning or tool_messages:
        delta_kwargs: dict[str, str | list[DeltaToolCall]] = {}
        if content_encountered:
            delta_kwargs["content"] = combined_content
        if combined_reasoning:
            delta_kwargs["reasoning"] = combined_reasoning
        if tool_messages:
            delta_kwargs["tool_calls"] = tool_messages
            tools_streamed = True
        delta_message = DeltaMessage(**delta_kwargs)
    else:
        delta_message = None

    return delta_message, tools_streamed
