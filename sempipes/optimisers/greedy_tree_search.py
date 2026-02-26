from sempipes.logging import get_logger
from sempipes.optimisers.search_policy import _ROOT_TRIAL, OperatorMemories, SearchNode, SearchPolicy

logger = get_logger()


class TreeSearch(SearchPolicy):
    def __init__(self, min_num_drafts: int = 2):
        super().__init__()
        self.min_num_drafts = min_num_drafts

    def clone_empty(self) -> SearchPolicy:
        return TreeSearch(min_num_drafts=self.min_num_drafts)

    def create_next_search_node(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        assert self.root_node is not None
        nodes_with_children = {outcome.search_node.parent_trial for outcome in self.outcomes}
        unprocessed_draft_nodes = [
            outcome.search_node
            for outcome in self.outcomes
            if outcome.search_node.trial not in nodes_with_children and outcome.search_node.parent_trial is _ROOT_TRIAL
        ]
        num_unprocessed_draft_nodes = len(unprocessed_draft_nodes)

        if num_unprocessed_draft_nodes < self.min_num_drafts:  # pylint: disable=no-else-return
            logger.info("TREE_SEARCH> Drafting new node")
            next_search_node = self._draft(trial, operator_to_evolve, all_operator_names)
            return next_search_node
        else:
            next_search_node = self._improve_best(trial, operator_to_evolve, all_operator_names)
            logger.info(f"TREE_SEARCH> Trying to improve node with score {next_search_node.parent_score}")
            return next_search_node

    def _draft(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        root_outcome = next(filter(lambda outcome: outcome.search_node is self.root_node, self.outcomes), None)
        assert root_outcome is not None

        draft_node = self._expand_tree(trial, operator_to_evolve, all_operator_names, root_outcome)

        # TODO: Must be done for all operators
        inspirations = OperatorMemories()
        for outcome in self.outcomes:
            if outcome != root_outcome and outcome.score > root_outcome.score:
                inspirations.append(
                    operator_to_evolve, {"state": outcome.states.get(operator_to_evolve), "score": outcome.score}
                )

        draft_node.inspirations = inspirations

        return draft_node

    def _improve_best(self, trial: int, operator_to_evolve: str, all_operator_names: list[str]) -> SearchNode:
        best_outcome = max(self.outcomes, key=lambda outcome: (outcome.score, -outcome.search_node.trial))

        return self._expand_tree(trial, operator_to_evolve, all_operator_names, best_outcome)
