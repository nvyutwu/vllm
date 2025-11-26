# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import cast

import jinja2
from fastapi import Request

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.renderer import RenderConfig
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs.data import EmbedsPrompt, TokensPrompt, is_embeds_prompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters

logger = init_logger(__name__)
payload_logger = logging.getLogger("vllm.payload")

# Import Harmony utilities for gpt-oss models
try:
    from vllm.entrypoints.harmony_utils import (
        get_stop_tokens_for_assistant_actions,
        get_streamable_parser_for_assistant,
        parse_chat_output,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("Harmony utilities not available. gpt-oss models will not be supported in Completion API.")


class OpenAIServingCompletion(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        # set up logits processors
        self.logits_processors = self.model_config.logits_processors

        self.enable_log_outputs = enable_log_outputs
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.enable_force_include_usage = enable_force_include_usage
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default completion sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )
        
        # Check if model is gpt-oss and needs Harmony format
        self.use_harmony = HARMONY_AVAILABLE and self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            logger.info("Enabling Harmony format for gpt-oss model in Completion API")
            if envs.VLLM_PARSE_HARMONY_OUTPUT:
                logger.info("Harmony output parsing is ENABLED (thinking, content, tool_calls will be separated)")
            else:
                logger.warning("Harmony output parsing is DISABLED - raw model output will be returned")
            # Add stop tokens for Harmony assistant actions
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response("suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Log request payload BEFORE any processing
        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("cmpl-") else request_id
        if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
            headers_json = ""
            try:
                if raw_request is not None:
                    headers_to_log = {k: v for k, v in raw_request.headers.items()}
                    headers_json = json.dumps(headers_to_log, ensure_ascii=False)
            except Exception:
                headers_json = ""
            try:
                req_dump = request.model_dump()
            except Exception:
                req_dump = None
            try:
                req_str = json.dumps(req_dump, ensure_ascii=False) if req_dump is not None else ""
            except Exception:
                req_str = ""
            try:
                payload_logger.info(
                    "openai.request",
                    extra={
                        "rid": rid_hint or "",
                        "endpoint": self.__class__.__name__,
                        "payload": req_str,
                        "headers": headers_json,
                    },
                )
            except Exception:
                pass

        try:
            lora_request = self._maybe_get_adapters(request)

            if self.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = await self.engine_client.get_tokenizer()
            
            # Use Harmony format for gpt-oss models (similar to Ollama)
            if self.use_harmony:
                from vllm.entrypoints.chat_utils import apply_hf_chat_template, ConversationMessage
                
                logger.info("Using Harmony format for gpt-oss model in Completion API")
                
                # Convert prompt to messages and use chat template (like Ollama does)
                prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
                engine_prompts = []
                
                for prompt_text in prompts:
                    # Convert to simple user message  
                    conversation = [ConversationMessage(role="user", content=prompt_text)]
                    
                    # Prepare chat template kwargs for gpt-oss
                    chat_template_kwargs = {}
                    if request.reasoning_effort is not None:
                        chat_template_kwargs["reasoning_effort"] = request.reasoning_effort
                    if request.builtin_tools is not None:
                        chat_template_kwargs["builtin_tools"] = request.builtin_tools
                        logger.debug(f"Enabling built-in tools: {request.builtin_tools}")
                    
                    # Apply chat template to get Harmony-formatted prompt string
                    # This produces: <|start|>system<|message|>...<|end|>
                    #                <|start|>user<|message|>{prompt}<|end|>
                    #                <|start|>assistant
                    # With builtin_tools, also includes browser/python tool definitions
                    prompt_str = apply_hf_chat_template(
                        tokenizer=tokenizer,
                        conversation=conversation,
                        chat_template=None,  # Use model's default chat template
                        tools=None,  # No custom tools for basic completion
                        model_config=self.model_config,
                        add_generation_prompt=True,  # Adds <|start|>assistant
                        **chat_template_kwargs,
                    )
                    
                    # Log the rendered prompt for debugging
                    logger.debug(f"[Harmony Input] Rendered prompt:\n{prompt_str}")
                    
                    # Tokenize the chat-templated string
                    # Special tokens are already in the template, so don't add them again
                    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
                    
                    engine_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
                    # Store the rendered prompt string for echo support
                    engine_prompt["prompt"] = prompt_str
                    if request.cache_salt is not None:
                        engine_prompt["cache_salt"] = request.cache_salt
                    engine_prompts.append(engine_prompt)
            else:
                renderer = self._get_renderer(tokenizer)
                engine_prompts = await renderer.render_prompt_and_embeds(
                    prompt_or_prompts=request.prompt,
                    prompt_embeds=request.prompt_embeds,
                    config=self._build_render_config(request),
                )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text, prompt_token_ids, prompt_embeds = (
                    self._get_prompt_components(engine_prompt)
                )

                input_length = None
                if prompt_token_ids is not None:
                    input_length = len(prompt_token_ids)
                elif prompt_embeds is not None:
                    input_length = len(prompt_embeds)
                else:
                    raise NotImplementedError

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=input_length,
                    default_sampling_params=self.default_sampling_params,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )
                    validate_logits_processors_parameters(
                        self.logits_processors,
                        sampling_params,
                    )

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                # Mypy inconsistently requires this second cast in different
                # environments. It shouldn't be necessary (redundant from above)
                # but pre-commit in CI fails without it.
                engine_prompt = cast(EmbedsPrompt | TokensPrompt, engine_prompt)
                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    engine_request, tokenization_kwargs = await self._process_inputs(
                        request_id_item,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                    generator = self.engine_client.generate(
                        engine_request,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        prompt_text=prompt_text,
                        tokenization_kwargs=tokenization_kwargs,
                        data_parallel_rank=data_parallel_rank,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self.models.model_name(lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (
            request.stream
            and (request.best_of is None or request.n == request.best_of)
            and not request.use_beam_search
        )

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                engine_prompts,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
            )

        # Non-streaming response
        final_res_batch: list[RequestOutput | None] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    engine_prompt = engine_prompts[i]
                    final_res.prompt = (
                        None
                        if is_embeds_prompt(engine_prompt)
                        else engine_prompt.get("prompt")
                    )

            final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        engine_prompts: list[TokensPrompt | EmbedsPrompt],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts
        num_cached_tokens = None
        first_iteration = True
        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("cmpl-") else request_id

        # Track accumulated content for output logging
        previous_texts = [""] * num_choices * num_prompts
        previous_thinking_texts = [""] * num_choices * num_prompts
        previous_tool_calls_content: list[list[tuple[str, str]]] = [[] for _ in range(num_choices * num_prompts)]

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )
        
        # Initialize Harmony parsers for streaming (similar to serving_chat.py)
        harmony_parsers = None
        # Track accumulated tool call content per choice (parsed at the end)
        accumulated_tool_content: dict[int, str] = {}
        current_tool_recipient: dict[int, str | None] = {}
        
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() 
                for _ in range(num_choices * num_prompts)
            ]
            logger.info(f"Initialized {len(harmony_parsers)} Harmony parsers for streaming")

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs

                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False

                prompt_text = res.prompt
                if prompt_text is None:
                    engine_prompt = engine_prompts[prompt_idx]
                    prompt_text = (
                        None
                        if is_embeds_prompt(engine_prompt)
                        else engine_prompt.get("prompt")
                    )

                # Prompt details are excluded from later streamed outputs
                if prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    # Useful when request.return_token_ids is True
                    # Returning prompt token IDs shares the same logic
                    # with the echo implementation.
                    prompt_token_ids_to_return: list[int] | None = None

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        if request.return_token_ids:
                            prompt_text = ""
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids,
                                *output.token_ids,
                            ]
                            out_logprobs = [
                                *(prompt_logprobs or []),
                                *(output.logprobs or []),
                            ]
                        prompt_token_ids_to_return = prompt_token_ids
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = ""
                        delta_thinking = ""
                        delta_tool_calls: list[ToolCall] = []
                        
                        # Parse Harmony output (serving_chat.py lines 761-853)
                        # Only parse if VLLM_PARSE_HARMONY_OUTPUT is enabled (default: True)
                        if self.use_harmony and harmony_parsers is not None and envs.VLLM_PARSE_HARMONY_OUTPUT:
                            harmony_parser = harmony_parsers[i]
                            prev_recipient = harmony_parser.current_recipient
                            
                            for token_id in output.token_ids:
                                harmony_parser.process(token_id)
                                content_delta = harmony_parser.last_content_delta or ""
                                
                                # Route content based on channel (like Ollama's AddContent)
                                cur_channel = harmony_parser.current_channel
                                cur_recipient = harmony_parser.current_recipient
                                
                                if cur_channel == "final":
                                    delta_text += content_delta
                                elif cur_channel == "analysis":
                                    # Check if it's a tool call on analysis channel (python can be on analysis)
                                    if cur_recipient and (cur_recipient.startswith("browser.") or cur_recipient == "python"):
                                        # Built-in tool on analysis channel
                                        if i not in accumulated_tool_content:
                                            accumulated_tool_content[i] = ""
                                            current_tool_recipient[i] = cur_recipient
                                        accumulated_tool_content[i] += content_delta
                                    else:
                                        # Regular reasoning content
                                        delta_thinking += content_delta
                                elif cur_channel == "commentary" and cur_recipient:
                                    # Tool call content - accumulate for both custom and built-in tools
                                    # Custom tools: functions.*
                                    # Built-in tools: browser.*, python
                                    if (cur_recipient.startswith("functions.") or 
                                        cur_recipient.startswith("browser.") or 
                                        cur_recipient == "python"):
                                        if i not in accumulated_tool_content:
                                            accumulated_tool_content[i] = ""
                                            current_tool_recipient[i] = cur_recipient
                                        accumulated_tool_content[i] += content_delta
                            
                            # Parse tool calls only when done (use output.finish_reason)
                            if output.finish_reason is not None:
                                if i in accumulated_tool_content and current_tool_recipient.get(i):
                                    # Drain and parse the accumulated tool call
                                    recipient = current_tool_recipient[i]
                                    tool_content = accumulated_tool_content[i]
                                    
                                    # Extract function name based on recipient format
                                    function_name = None
                                    if recipient and recipient.startswith("functions."):
                                        # Custom tools: functions.calc → "calc"
                                        function_name = recipient.split("functions.", 1)[1]
                                    elif recipient and recipient.startswith("browser."):
                                        # Built-in browser tools: browser.search → "browser.search"
                                        function_name = recipient
                                    elif recipient == "python":
                                        # Built-in python tool: python → "python"
                                        function_name = "python"
                                    
                                    if function_name:
                                        delta_tool_calls.append(
                                            ToolCall(
                                                function=FunctionCall(
                                                    name=function_name,
                                                    arguments=tool_content,
                                                )
                                            )
                                        )
                                        tool_type = "built-in" if not recipient.startswith("functions.") else "custom"
                                        logger.debug(
                                            f"[Harmony Output] {tool_type.capitalize()} tool call in final chunk: "
                                            f"{function_name}({tool_content[:50]}...)"
                                        )
                            
                            if delta_text or delta_thinking:
                                logger.debug(
                                    f"[Harmony Output] Delta - content: {repr(delta_text)}, "
                                    f"thinking: {repr(delta_thinking)}, "
                                    f"channel: {harmony_parser.current_channel}"
                                )
                        else:
                            delta_text = output.text
                        
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        # has_echoed[i] is reused here to indicate whether
                        # we have already returned the prompt token IDs.
                        if not has_echoed[i] and request.return_token_ids:
                            prompt_token_ids_to_return = prompt_token_ids
                            has_echoed[i] = True

                        if (
                            not delta_text
                            and not delta_token_ids
                            and not previous_num_tokens[i]
                        ):
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    # Prepare the choice with parsed Harmony content
                    choice = CompletionResponseStreamChoice(
                        index=i,
                        text=delta_text,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                        stop_reason=stop_reason,
                        prompt_token_ids=prompt_token_ids_to_return,
                        token_ids=(
                            as_list(output.token_ids)
                            if request.return_token_ids
                            else None
                        ),
                    )
                    
                    # Add thinking field if we have reasoning content (Harmony models)
                    if self.use_harmony and delta_thinking:
                        choice.thinking = delta_thinking
                    
                    # Add tool_calls only in final chunk when done 
                    if self.use_harmony and delta_tool_calls:
                        choice.tool_calls = delta_tool_calls

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[choice],
                    )
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

                    # Track content for output logging
                    if delta_text:
                        previous_texts[i] += delta_text
                    if delta_thinking:
                        previous_thinking_texts[i] += delta_thinking
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            previous_tool_calls_content[i].append(
                                (tc.function.name, tc.function.arguments)
                            )

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

            # Payload logging for streaming response summary
            if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
                try:
                    usage_dict = final_usage_info.model_dump() if final_usage_info else None
                except Exception:
                    usage_dict = None
                resp_summary = {
                    "id": rid_hint,
                    "object": "text_completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": [],
                    "usage": usage_dict,
                    "stream": True,
                }
                try:
                    payload_str = json.dumps(resp_summary, ensure_ascii=False)
                except Exception:
                    payload_str = ""
                try:
                    payload_logger.info(
                        "openai.response",
                        extra={
                            "rid": rid_hint,
                            "endpoint": self.__class__.__name__,
                            "payload": payload_str,
                        },
                    )
                except Exception:
                    pass

            # Output logging for streaming complete
            if self.enable_log_outputs and self.request_logger:
                for i in range(num_choices * num_prompts):
                    thinking_text = previous_thinking_texts[i]
                    content_text = previous_texts[i]
                    tool_calls_list = previous_tool_calls_content[i]

                    # Log reasoning with [reasoning] prefix
                    if thinking_text:
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=f"[reasoning] {thinking_text}",
                            output_token_ids=None,
                            finish_reason=None,
                            is_streaming=True,
                            delta=False,
                        )

                    # Log content
                    if content_text:
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=content_text,
                            output_token_ids=None,
                            finish_reason="streaming_complete",
                            is_streaming=True,
                            delta=False,
                        )

                    # Log tool calls with [tool_calls] prefix
                    if tool_calls_list:
                        tool_call_descriptions = [
                            f"{name}({args})" for name, args in tool_calls_list
                        ]
                        tool_calls_str = ", ".join(tool_call_descriptions)
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=f"[tool_calls: {tool_calls_str}]",
                            output_token_ids=None,
                            finish_reason="streaming_complete",
                            is_streaming=True,
                            delta=False,
                        )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        kv_transfer_params = None
        last_final_res = None
        for final_res in final_res_batch:
            last_final_res = final_res
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

            for output in final_res.outputs:
                assert request.max_tokens is not None
                
                # Initialize Harmony-specific fields (must be before if/else for echo)
                output_thinking = None
                output_tool_calls = None
                
                if request.echo:
                    if request.return_token_ids:
                        prompt_text = ""
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

                        output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    
                    # Parse Harmony output for non-streaming (similar to serving_chat.py)
                    # Only parse if VLLM_PARSE_HARMONY_OUTPUT is enabled (default: True)
                    if self.use_harmony and envs.VLLM_PARSE_HARMONY_OUTPUT:
                        # Use parse_chat_output to extract reasoning and content
                        # Similar to serving_chat.py line 1520
                        reasoning, content, _ = parse_chat_output(token_ids)
                        output_thinking = reasoning
                        output_text = content or ""
                        
                        # Parse tool calls from the token IDs
                        # Similar to Ollama's approach - parse at completion
                        parser = get_streamable_parser_for_assistant()
                        for token_id in token_ids:
                            parser.process(token_id)
                        
                        # Extract tool calls from completed messages
                        # Built-in tools can be on commentary OR analysis channel
                        tool_calls_list = []
                        for msg in parser.messages:
                            is_tool_call = False
                            function_name = None
                            
                            # Check commentary channel for all tools
                            if msg.channel == "commentary" and msg.recipient:
                                is_tool_call = True
                            # Check analysis channel for built-in tools (python, browser)
                            elif msg.channel == "analysis" and msg.recipient:
                                if msg.recipient.startswith("browser.") or msg.recipient == "python":
                                    is_tool_call = True
                            
                            if is_tool_call and msg.recipient:
                                # Parse both custom and built-in tools
                                if msg.recipient.startswith("functions."):
                                    # Custom tools: functions.calc → "calc"
                                    function_name = msg.recipient.split("functions.", 1)[1]
                                elif msg.recipient.startswith("browser."):
                                    # Built-in browser tools: browser.search → "browser.search"
                                    function_name = msg.recipient
                                elif msg.recipient == "python":
                                    # Built-in python tool: python → "python"
                                    function_name = "python"
                                
                                if function_name:
                                    arguments = msg.content[0].text if msg.content else ""
                                    tool_calls_list.append(
                                        ToolCall(
                                            function=FunctionCall(
                                                name=function_name,
                                                arguments=arguments,
                                            )
                                        )
                                    )
                        
                        if tool_calls_list:
                            output_tool_calls = tool_calls_list
                        
                        if tool_calls_list:
                            tool_names = [tc.function.name for tc in tool_calls_list]
                            logger.debug(
                                f"[Harmony Output] Parsed - content: {repr(output_text)}, "
                                f"thinking: {repr(reasoning if reasoning else None)}, "
                                f"tool_calls: {tool_names}"
                            )
                        else:
                            logger.debug(
                                f"[Harmony Output] Parsed - content: {repr(output_text)}, "
                                f"thinking: {repr(reasoning if reasoning else None)}"
                            )
                    else:
                        output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                    prompt_token_ids=(
                        prompt_token_ids if request.return_token_ids else None
                    ),
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                    # Add Harmony-specific fields
                    thinking=output_thinking,
                    tool_calls=output_tool_calls,
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        if (
            self.enable_prompt_tokens_details
            and last_final_res
            and last_final_res.num_cached_tokens
        ):
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=last_final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage
        if final_res_batch:
            kv_transfer_params = final_res_batch[0].kv_transfer_params

        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("cmpl-") else request_id

        # Payload logging for non-streaming response
        if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
            try:
                usage_dict = usage.model_dump() if usage else None
            except Exception:
                usage_dict = None
            resp_summary = {
                "id": rid_hint,
                "object": "text_completion",
                "created": created_time,
                "model": model_name,
                "choices": [],
                "usage": usage_dict,
                "stream": False,
            }
            try:
                payload_str = json.dumps(resp_summary, ensure_ascii=False)
            except Exception:
                payload_str = ""
            try:
                payload_logger.info(
                    "openai.response",
                    extra={
                        "rid": rid_hint,
                        "endpoint": self.__class__.__name__,
                        "payload": payload_str,
                    },
                )
            except Exception:
                pass

        # Output logging for non-streaming response
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                # Log reasoning with [reasoning] prefix
                if hasattr(choice, 'thinking') and choice.thinking:
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=f"[reasoning] {choice.thinking}",
                        output_token_ids=None,
                        finish_reason=None,
                        is_streaming=False,
                        delta=False,
                    )

                # Log content
                if choice.text:
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=choice.text,
                        output_token_ids=None,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

                # Log tool calls with [tool_calls] prefix
                if hasattr(choice, 'tool_calls') and choice.tool_calls:
                    tool_call_descriptions = []
                    for tc in choice.tool_calls:
                        if hasattr(tc.function, 'name') and hasattr(tc.function, 'arguments'):
                            tool_call_descriptions.append(
                                f"{tc.function.name}({tc.function.arguments})"
                            )
                    if tool_call_descriptions:
                        tool_calls_str = ", ".join(tool_call_descriptions)
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=f"[tool_calls: {tool_calls_str}]",
                            output_token_ids=None,
                            finish_reason=choice.finish_reason,
                            is_streaming=False,
                            delta=False,
                        )

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            kv_transfer_params=kv_transfer_params,
        )

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int,
        tokenizer: AnyTokenizer,
        initial_text_offset: int = 0,
        return_as_token_id: bool | None = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[float | None] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[dict[str, float] | None] = []

        last_token_len = 0

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append(
                    {
                        # Convert float("-inf") to the
                        # JSON-serializable float that OpenAI uses
                        self._get_decoded_token(
                            top_lp[1],
                            top_lp[0],
                            tokenizer,
                            return_as_token_id=should_return_as_token_id,
                        ): max(top_lp[1].logprob, -9999.0)
                        for i, top_lp in enumerate(step_top_logprobs.items())
                        if num_output_top_logprobs >= i
                    }
                )

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )

    def _build_render_config(
        self,
        request: CompletionRequest,
        max_input_length: int | None = None,
    ) -> RenderConfig:
        max_input_tokens_len = self.max_model_len - (request.max_tokens or 0)
        return RenderConfig(
            max_length=max_input_tokens_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
            cache_salt=request.cache_salt,
            needs_detokenization=bool(request.echo and not request.return_token_ids),
        )
