"""Monte Carlo Tree Search Root Node.
"""

import time
from faronona.faronona_action import FarononaAction
from .node import Node


class Search(object):
    """MTCS entry point."""

    def __init__(self, node: Node, max_rollout_depth: int = float('inf')) -> None:
        """Initializer for search.

        Args:
            node (Node): Root node.
            max_rollout_depth (int): Maximum depth to look into future during rollout. Defauls to end of game.
        """
        self.root = node
        self.max_rollout_depth = max_rollout_depth

    def best_action(self, n_iterations: int = None, time_iterations: float = None, epsilon: float = .1) -> FarononaAction:
        """Search the best action to make.

        Args:
            n_simulation (int, optional): [description]. Defaults to None.
            time_simulation (float, optional): [description]. Defaults to None.
        """
        if n_iterations is None :
            assert(time_iterations is not None)
            end_time = time.time() + time_iterations
            while time.time() < end_time:
                self.run_iteration()
        else:
            for _ in range(n_iterations):            
                self.run_iteration()
        # to select best child go for exploitation only
        best_child = self.root.best_child(epsilon=epsilon)
        action, _ = best_child.parent
        return action

    def run_iteration(self):
        """Run a single iteration."""
        v = self._tree_policy()
        reward = v.rollout(max_depth=self.max_rollout_depth)
        v.backpropagate(reward)

    def _tree_policy(self):
        """Select node to run rollout."""
        current_node: Node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
