"""Shared request type counters for all OpenAI-compatible API endpoints.

These counters track workload composition (images, videos, tool calls,
structured output) across chat, completion, and responses APIs.
"""

from prometheus_client import Counter

request_type_image = Counter(
    name="request_type_image_total",
    documentation="Total requests containing images",
)
request_type_video = Counter(
    name="request_type_video_total",
    documentation="Total requests containing videos",
)
request_type_tool_call = Counter(
    name="request_type_tool_call_total",
    documentation="Total requests with tool calls enabled",
)
request_type_structured_output = Counter(
    name="request_type_structured_output_total",
    documentation="Total requests with structured output "
    "(json_schema, json_object, structural_tag, regex, choice, or grammar)",
)


def classify_chat_request(request) -> None:
    """Classify a ChatCompletionRequest and increment counters."""
    has_image = False
    has_video = False
    for msg in request.messages:
        content = (
            msg.get("content") if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        has_image = True
                    elif part_type == "video_url":
                        has_video = True
    if has_image:
        request_type_image.inc()
    if has_video:
        request_type_video.inc()
    if request.tools and request.tool_choice != "none":
        request_type_tool_call.inc()
    _classify_structured_output_chat_completion(request)


def classify_completion_request(request) -> None:
    """Classify a CompletionRequest and increment counters.

    Completions support structured output (response_format, structured_outputs)
    but not images, videos, or tool calls.
    """
    _classify_structured_output_chat_completion(request)


def classify_responses_request(request) -> None:
    """Classify a ResponsesRequest and increment counters.

    Responses support tools, structured output (via text.format),
    and images in input.
    """
    # Images in input
    _classify_responses_images(request)

    # Tool calls
    if getattr(request, "tools", None) and getattr(request, "tool_choice", None) != "none":
        request_type_tool_call.inc()

    # Structured output via text.format
    text = getattr(request, "text", None)
    if text is not None:
        fmt = getattr(text, "format", None)
        if fmt is not None and hasattr(fmt, "type"):
            if fmt.type in ("json_schema", "json_object"):
                request_type_structured_output.inc()


def _classify_responses_images(request) -> None:
    """Check for images in Responses API input items."""
    input_items = getattr(request, "input", None)
    if not isinstance(input_items, list):
        return
    for item in input_items:
        item_type = (
            item.get("type", "") if isinstance(item, dict)
            else getattr(item, "type", "")
        )
        if item_type == "input_image":
            request_type_image.inc()
            return
        # Check nested content list (e.g. message with content parts)
        content = (
            item.get("content", None) if isinstance(item, dict)
            else getattr(item, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                part_type = (
                    part.get("type", "") if isinstance(part, dict)
                    else getattr(part, "type", "")
                )
                if part_type in ("input_image", "image_url"):
                    request_type_image.inc()
                    return


def _classify_structured_output_chat_completion(request) -> None:
    """Check for structured output in chat/completion requests."""
    response_format = getattr(request, "response_format", None)
    if (
        response_format is not None
        and hasattr(response_format, "type")
        and response_format.type
        in ("json_schema", "json_object", "structural_tag")
    ):
        request_type_structured_output.inc()
        return
    if getattr(request, "structured_outputs", None) is not None:
        request_type_structured_output.inc()
        return
