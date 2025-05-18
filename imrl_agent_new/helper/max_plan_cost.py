from overcooked_ai_py.planning.planners import MotionPlanner


def max_plan_cost(mp: MotionPlanner):
    """
    Return the largest shortest-path cost among *all* pairs of walkable tiles
    in this MotionPlanner.  Called once at env reset.
    """
    return max(plan[2] for plan in mp.all_plans.values())
