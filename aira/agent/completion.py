"""Deterministic completion checker — all conditions must be met."""


def is_task_complete(
    planner_done: bool,
    last_tool_success: bool,
    validation_errors: list,
    output_valid: bool,
) -> bool:
    """
    Check if a task is complete. ALL conditions must be true.

    Args:
        planner_done: The planner signals the task is finished.
        last_tool_success: The last tool execution succeeded.
        validation_errors: List of current validation errors.
        output_valid: The output passes deterministic validation.

    Returns:
        True only if ALL conditions are met.
    """
    return all([
        planner_done,
        last_tool_success,
        len(validation_errors) == 0,
        output_valid,
    ])
