import numpy as np

from overcooked_ai_py.planning.planners import MediumLevelActionManager


def get_plan(start_pos_and_or, motion_goals, mlam: MediumLevelActionManager):
    """
    Chooses motion goal that has the lowest cost action plan.
    Returns the plan to the goal.
    """
    min_cost = np.inf
    best_plan = []
    for goal in motion_goals:
        action_plan, _, plan_cost = mlam.motion_planner.get_plan(
            start_pos_and_or, goal
        )
        if plan_cost < min_cost:
            best_plan = action_plan
            min_cost = plan_cost
    return best_plan
