from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from rollout_agent import PRMAgent, RolloutAgent
from custom_types import RolloutOutcome, TurnResult
from env_client import TerminalEnvClient

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        *,
        rollout_agent: RolloutAgent,
        tool_schemas: List[Dict[str, Any]],
        env_client: TerminalEnvClient,
        lease_id: str,
        prm_agent: PRMAgent | None = None,
        max_iterations: int = 10,
        max_parse_errors: int = 3,
        log_tag: str = "",
    ) -> None:
        self._rollout_agent = rollout_agent
        self._tool_schemas = tool_schemas
        self._env_client = env_client
        self._lease_id = lease_id
        self._prm_agent = prm_agent
        self._max_iterations = max_iterations
        self._max_parse_errors = max_parse_errors
        self._log_tag = log_tag
        self._model_turn_count = 0

    def _reset(self, input_message: Any) -> None:
        self._model_turn_count = 0
        self._rollout_agent.start_turn_loop(input_message)
        self._rollout_agent.set_max_parse_errors(self._max_parse_errors)

    def _reached_iteration_limit(self) -> bool:
        return self._model_turn_count >= self._max_iterations

    def _reached_parse_error_limit(self) -> bool:
        return self._rollout_agent.parse_error_count >= self._max_parse_errors

    async def _get_turn_context(self) -> List[dict[str, Any]]:
        return await self._rollout_agent.get_turn_context()

    async def _run_model_turn(
        self, context_messages: List[Dict[str, Any]]
    ) -> TurnResult:
        self._model_turn_count += 1
        chat_completion, interaction = (
            await self._rollout_agent._sglang_client.generate(
                messages=context_messages,
                tools=self._tool_schemas,
                turn_idx=self._model_turn_count,
            )
        )
        model_response, tool_call_requests, parse_error_record = (
            await self._rollout_agent.consume_completion(chat_completion)
        )
        if parse_error_record:
            logger.warning(
                "%s Turn %d: tool-call parse error.",
                self._log_tag,
                self._model_turn_count,
            )
        return TurnResult(
            interaction=interaction,
            model_response=model_response,
            tool_call_requests=tool_call_requests,
            parse_error_record=parse_error_record,
        )

    async def run_episode(self, user_msg: Any) -> RolloutOutcome:
        prm_pending: list[tuple[int, asyncio.Task]] = []
        prm_turn_scores: dict[int, float] = {}
        prm_turn_details: list[dict[str, Any]] = []
        interactions = []
        final_model_response = None
        final_response = None
        reached_iteration_limit = False
        reached_parse_error_limit = False

        try:
            self._reset(user_msg)

            while True:
                if self._reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached.",
                        self._log_tag,
                        self._max_iterations,
                    )
                    reached_iteration_limit = True
                    break

                try:
                    context_messages: List[dict[str, Any]] = (
                        await self._get_turn_context()
                    )
                except Exception as exc:
                    logger.error(
                        "%s Failed to build model context for turn %d (%s): %s",
                        self._log_tag,
                        self._model_turn_count,
                        type(exc).__name__,
                        exc,
                    )
                    raise
                try:
                    turn_result: TurnResult = await self._run_model_turn(
                        context_messages
                    )
                except Exception as exc:
                    logger.error(
                        "%s Model turn %d failed (%s): %s",
                        self._log_tag,
                        self._model_turn_count,
                        type(exc).__name__,
                        exc,
                    )
                    raise
                # print(turn_result)

                interaction = turn_result.interaction
                turn_idx = interaction.turn_idx
                interactions.append(interaction)

                if turn_result.model_response is None:
                    logger.warning(
                        "%s Turn %d returned empty model_response.",
                        self._log_tag,
                        turn_idx,
                    )
                    break
                final_model_response = turn_result.model_response

                if turn_result.tool_call_requests:
                    print("Turn idx", turn_idx)
                    print(turn_result.tool_call_requests)
                    await self._execute_tool_calls(
                        turn_idx, turn_result.tool_call_requests
                    )

                self._prm_record_turn(turn_idx, interaction, turn_result)
                if self._prm_agent is not None:
                    prm_pending.append(
                        (
                            turn_idx,
                            asyncio.create_task(self._prm_agent.judge_turn(turn_idx)),
                        )
                    )

                if (
                    not turn_result.tool_call_requests
                    and not turn_result.parse_error_record
                ):
                    break  # natural completion

                if self._reached_parse_error_limit():
                    logger.error(
                        "%s Max parse errors (%d) reached at turn %d.",
                        self._log_tag,
                        self._max_parse_errors,
                        turn_idx,
                    )
                    reached_parse_error_limit = True
                    break

            try:
                final_response = self._rollout_agent.finalize_response(
                    final_model_response
                )
            except Exception as exc:
                logger.error(
                    "%s Finalize response failed after %d model turn(s) (%s): %s",
                    self._log_tag,
                    self._model_turn_count,
                    type(exc).__name__,
                    exc,
                )
                raise

            await self._drain_prm_pending(
                prm_pending, prm_turn_scores, prm_turn_details
            )

            logger.info(
                "%s Rollout finished: turns=%d parse_errors=%d iteration_limit=%s parse_error_limit=%s",
                self._log_tag,
                self._model_turn_count,
                self._rollout_agent.parse_error_count,
                reached_iteration_limit,
                reached_parse_error_limit,
            )

            return RolloutOutcome(
                interactions=interactions,
                final_response=final_response,
                reached_iteration_limit=reached_iteration_limit,
                reached_parse_error_limit=reached_parse_error_limit,
                model_turn_count=self._model_turn_count,
                parse_error_count=self._rollout_agent.parse_error_count,
                prm_turn_scores=prm_turn_scores,
                prm_turn_details=prm_turn_details,
            )
        finally:
            for _turn_idx, task in prm_pending:
                if not task.done():
                    task.cancel()

    def _prm_record_turn(
        self,
        turn_idx: int,
        interaction: Any,
        turn_result: TurnResult,
    ) -> None:
        if self._prm_agent is None:
            return

        tool_calls_for_prm = [
            {"tool_name": tool_call.tool_name, "args": tool_call.args}
            for tool_call in (turn_result.tool_call_requests or [])
        ]
        self._prm_agent.record_model_turn(
            turn_idx,
            assistant_text=interaction.output_text or "",
            tool_calls=tool_calls_for_prm or None,
            parse_error_record=turn_result.parse_error_record,
            finish_reason=interaction.finish_reason,
        )

    async def _execute_tool_calls(
        self,
        turn_idx: int,
        tool_call_requests: list[Any],
    ) -> None:
        logger.info(
            "%s Turn %d: executing %d tool call(s).",
            self._log_tag,
            turn_idx,
            len(tool_call_requests),
        )
        for tool_call_request in tool_call_requests:
            try:
                env_result = await self._env_client.exec_tool(
                    self._lease_id,
                    tool_call_request.tool_name,
                    tool_call_request.args,
                )
            except Exception as exc:
                logger.error(
                    "%s Turn %d tool %s failed (%s): %s",
                    self._log_tag,
                    turn_idx,
                    tool_call_request.tool_name,
                    type(exc).__name__,
                    exc,
                )
                raise
            self._rollout_agent.record_tool_result(tool_call_request, env_result)
            if self._prm_agent is not None:
                self._prm_agent.record_tool_result(
                    turn_idx, tool_call_request, env_result
                )

    async def _drain_prm_pending(
        self,
        prm_pending: list[tuple[int, asyncio.Task]],
        prm_turn_scores: dict[int, float],
        prm_turn_details: list[dict[str, Any]],
    ) -> None:
        for turn_idx, prm_task in prm_pending:
            try:
                output_text, score = await prm_task
                prm_turn_scores[turn_idx] = score
                prm_turn_details.append(
                    {
                        "turn_idx": turn_idx,
                        "score": score,
                        "output_text": output_text,
                    }
                )
                logger.info(
                    "%s PRM judge turn %d score=%.4f, output_text=%s",
                    self._log_tag,
                    turn_idx,
                    score,
                    output_text.replace("\n", ""),
                )
            except Exception as exc:
                logger.warning(
                    "%s PRM judge failed for turn %d (ignored): %s",
                    self._log_tag,
                    turn_idx,
                    exc,
                )
                prm_turn_scores[turn_idx] = 0.0
                prm_turn_details.append(
                    {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
                )
