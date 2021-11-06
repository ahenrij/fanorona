"""Monte Carlo Tree Search Root Node.
"""

import time
from faronona.faronona_action import FarononaAction
from .node import Node


class Search(object):
    """MTCS entry point."""

    EPSILON = .1 # Exploration probability

    def __init__(self, node: Node) -> None:
        """Initializer for search.

        Args:
            node (Node): Root node.
        """
        self.root = node

    def best_action(self, n_simulations: int = None, time_simulations: float = None) -> FarononaAction:
        """Search the best action to make.

        Args:
            n_simulation (int, optional): [description]. Defaults to None.
            time_simulation (float, optional): [description]. Defaults to None.
        """
        if n_simulations is None :
            assert(time_simulations is not None)
            end_time = time.time() + time_simulations
            while time.time() < end_time:
                self.run_simulation()
        else:
            for _ in range(n_simulations):            
                self.run_simulation()
        # to select best child go for exploitation only
        best_child = self.root.best_child(epsilon=self.EPSILON)
        action, _ = best_child.parent
        return action

    def run_simulation(self):
        """Run a single simulation"""
        v = self._tree_policy()
        reward = v.rollout()
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
