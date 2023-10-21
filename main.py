from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict

import numpy as np
from multiset import Multiset

InformationSetKey = Tuple[str, str]

# Define game-specific constants and rules
NUM_PLAYERS = 2
STACK_SIZE = 3  # Initial stack size for both players (5BB)
POSTED_POT_PLAYER1 = 0.5  # Player 1 posts 0.5BB
POSTED_POT_PLAYER2 = 1.0  # Player 2 posts 1BB
NUM_CARDS = 1  # Number of cards each player is dealt
N_ITERATIONS = 1000
DRAW_PREFIX = "DRAW_"

# Ranks from 2 to Ace (ignoring suits)
RANKS = [str(rank) for rank in range(2, 10)] + ["J", "Q", "K", "A"]
RANK_VALUES = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,  # Jack
    "Q": 10,  # Queen
    "K": 11,  # King
    "A": 12,  # Ace
}
CARDS_PER_RANK = 1
FULL_DECK = Multiset(RANKS * CARDS_PER_RANK)


@dataclass(frozen=True, eq=True, order=True)
class InformationSetKey:
    card: str
    history: str

    def __str__(self):
        return f"{self.card} {self.history}"


class Player(Enum):
    PLAYER_1 = "PLAYER_1"
    PLAYER_2 = "PLAYER_2"
    TERMINAL = "TERMINAL"


def action_to_history_char(action: str) -> str:
    """
    Args:
        action: Action str. Must be ALL_IN, FOLD, or DRAW_{N}

    Returns:
        representation for history char
    """

    if action == "ALL_IN":
        return "a"
    elif action == "FOLD":
        return "f"
    elif action.startswith(DRAW_PREFIX):
        return action[-1]
    else:
        raise RuntimeError(f"Invalid action: {action}")


def history_to_player(history: str) -> Player:
    """
    Args:
        history: History str

    Returns:
        Whether it's player 1 or player 2 at the current node
    """

    if is_terminal(history):
        return Player.TERMINAL
    if len(history) == 2 or len(history) == 5:
        return Player.PLAYER_1
    return Player.PLAYER_2


def next_history_util_multiplier(history: str, next_history: str) -> float:
    """

    Args:
        history: Current history
        next_history: Next history

    Returns:
        1.0 if:
            - Current history is PLAYER_1 and next history is PLAYER_1 or TERMINAL
            - Current history is PLAYER_2 and next history is PLAYER_2
        -1.0 otherwise

        Note that this works because TERMINAL util is stated from the perspective of player 1

    """

    history_player = history_to_player(history)
    next_history_player = history_to_player(next_history)

    if (
        history_player == Player.PLAYER_1
        and next_history_player in (Player.PLAYER_1, Player.TERMINAL)
    ) or (history_player == Player.PLAYER_2 and next_history_player == Player.PLAYER_2):
        return 1.0
    else:
        return -1.0


class InformationSet:
    def __init__(self, key: InformationSetKey):
        self.key = key
        # TODO adjust for multiple cards. For one card, this could be ALL_IN/FOLD or DRAW_0/DRAW_1
        if len(key.history) == 2 or len(key.history) == 3:
            self.actions = ["ALL_IN", "FOLD"]
        elif len(key.history) == 4 or len(key.history) == 5:
            self.actions = [f"DRAW_{i}" for i in range(NUM_CARDS + 1)]
        else:
            self.actions = []

        self.n_actions = len(self.actions)

        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)
        self.strategy = np.repeat(1 / self.n_actions, self.n_actions)
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def next_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.strategy = self.calc_strategy()
        self.reach_pr_sum += self.reach_pr
        self.reach_pr = 0

    def calc_strategy(self):
        """
        Calculate current strategy from the sum of regret.
        """
        strategy = self.make_positive(self.regret_sum)
        total = sum(strategy)
        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.repeat(1 / self.n_actions, self.n_actions)

        return strategy

    def get_average_strategy(self):
        """
        Calculate average strategy over all iterations. This is the
        Nash equilibrium strategy.
        """
        strategy = self.strategy_sum / self.reach_pr_sum

        # Purify to remove actions that are likely a mistake
        strategy = np.where(strategy < 0.001, 0, strategy)

        # Re-normalize
        total = sum(strategy)
        strategy /= total

        return strategy

    def make_positive(self, x):
        return np.where(x > 0, x, 0)

    def __str__(self):
        actions = self.actions
        strategies = ["{:03.2f}".format(x) for x in self.get_average_strategy()]
        actions_to_strategies = {
            action: strategy for action, strategy in zip(actions, strategies)
        }
        return "{} {}".format(str(self.key).ljust(10), actions_to_strategies)


def cfr(
    *,
    information_set_map: Dict[InformationSetKey, InformationSet],
    history: str,
    card_1: str,
    card_2: str,
    player_1_discarded: Multiset,
    player_2_discarded: Multiset,
    pr_1: float,
    pr_2: float,
    pr_c: float,
) -> np.array:
    """
    Counterfactual regret minimization algorithm.

    Parameters
    ----------

    information_set_map: dict
        Dictionary of all information sets.
    history : str
        A string representation of the game tree path we have taken.
        Each character of the string represents a single action:

        'd': deal action
        'c': check action
        'b': bet action
    card_1 : int
        player A's cards. Sorted tuple
    card_2 : int
        player B's cards. Sorted tuple
    pr_1 : (0, 1.0), float
        The probability that player A reaches `history`.
    pr_2 : (0, 1.0), float
        The probability that player B reaches `history`.
    pr_c: (0, 1.0), float
        The probability contribution of chance events to reach `history`.
    """
    if is_deal_node(history):
        return chance_util(information_set_map)

    if is_terminal(history):
        return terminal_util(history, card_1, card_2)

    n = len(history)
    player = history_to_player(history)
    info_set = get_info_set(
        information_set_map, card_1 if player == Player.PLAYER_1 else card_2, history
    )

    strategy = info_set.strategy
    if player == Player.PLAYER_1:
        info_set.reach_pr += pr_1
    else:
        info_set.reach_pr += pr_2

    # Counterfactual utility per action.
    action_utils = np.zeros(info_set.n_actions)

    for i, action in enumerate(info_set.actions):
        history_char = action_to_history_char(action)
        next_history = history + history_char
        util_multiplier = next_history_util_multiplier(history, next_history)
        # TODO adjust for multiple cards
        if action.startswith(DRAW_PREFIX) and action == DRAW_PREFIX + "1":
            n_cards_to_draw = int(history_char)
            if n_cards_to_draw == 1:
                expected_value = 0
                if player == Player.PLAYER_1:
                    player_1_discarded = Multiset([card_1])
                    cards_to_draw_from = (
                        FULL_DECK - Multiset([card_1, card_2]) - player_2_discarded
                    )
                    for card in cards_to_draw_from:
                        card_1 = card
                        expected_value += util_multiplier * cfr(
                            information_set_map=information_set_map,
                            history=next_history,
                            card_1=card_1,
                            card_2=card_2,
                            player_1_discarded=player_1_discarded,
                            player_2_discarded=player_2_discarded,
                            pr_1=pr_1 * strategy[i],
                            pr_2=pr_2,
                            pr_c=pr_c * 1 / len(cards_to_draw_from),
                        )
                    action_utils[i] = expected_value / len(cards_to_draw_from)
                else:
                    player_2_discarded = Multiset([card_2])
                    cards_to_draw_from = FULL_DECK - Multiset([card_1, card_2])
                    for card in cards_to_draw_from:
                        card_2 = card
                        expected_value += util_multiplier * cfr(
                            information_set_map=information_set_map,
                            history=next_history,
                            card_1=card_1,
                            card_2=card_2,
                            player_1_discarded=player_1_discarded,
                            player_2_discarded=player_2_discarded,
                            pr_1=pr_1,
                            pr_2=pr_2 * strategy[i],
                            pr_c=pr_c * 1 / len(cards_to_draw_from),
                        )
                    action_utils[i] = expected_value / len(cards_to_draw_from)

        else:
            if player == Player.PLAYER_1:
                action_utils[i] = util_multiplier * cfr(
                    information_set_map=information_set_map,
                    history=next_history,
                    card_1=card_1,
                    card_2=card_2,
                    player_1_discarded=player_1_discarded,
                    player_2_discarded=player_2_discarded,
                    pr_1=pr_1 * strategy[i],
                    pr_2=pr_2,
                    pr_c=pr_c,
                )
            else:
                action_utils[i] = util_multiplier * cfr(
                    information_set_map=information_set_map,
                    history=next_history,
                    card_1=card_1,
                    card_2=card_2,
                    player_1_discarded=player_1_discarded,
                    player_2_discarded=player_2_discarded,
                    pr_1=pr_1,
                    pr_2=pr_2 * strategy[i],
                    pr_c=pr_c,
                )

    # Utility of information set.
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    if player == Player.PLAYER_1:
        info_set.regret_sum += pr_2 * pr_c * regrets
    else:
        info_set.regret_sum += pr_1 * pr_c * regrets

    return util


def is_deal_node(history):
    """
    Determine if we are at the deal node based on tree history.
    """
    return history == ""


def chance_util(information_set_map) -> float:
    # TODO adjust for multiple cards
    expected_value = 0
    for card_1 in RANKS:
        for card_2 in RANKS:
            if card_1 != card_2:
                probability = 1 / 78
                expected_value += probability * cfr(
                    information_set_map=information_set_map,
                    history="rr",
                    card_1=card_1,
                    card_2=card_2,
                    player_1_discarded=Multiset(),
                    player_2_discarded=Multiset(),
                    pr_1=1,
                    pr_2=2,
                    pr_c=probability,
                )
    return expected_value


def is_terminal(history: str):
    """
    Returns True if the history is a terminal history.
    """
    # TODO adjust for multiple cards
    if history == "rrf" or history == "rraf" or len(history) == 6:
        return True
    return False


def terminal_util(history: str, card_1: str, card_2: str):
    """
    Returns the utility of a terminal history from the perspective of player 1
    """
    # TODO adjust for multiple cards
    if history == "rrf":
        return -POSTED_POT_PLAYER1
    if history == "rraf":
        return POSTED_POT_PLAYER2
    return STACK_SIZE if RANK_VALUES[card_1] < RANK_VALUES[card_2] else -STACK_SIZE


def get_info_set(
    i_map: Dict[InformationSetKey, InformationSet], card: str, history: str
) -> InformationSet:
    """
    Retrieve information set from dictionary.
    """
    # TODO adjust for multiple cards
    key = InformationSetKey(card, history)

    if key not in i_map:
        info_set = InformationSet(key)
        i_map[key] = info_set
        return info_set

    return i_map[key]


def display_results(
    ev: float, information_set_map: Dict[InformationSetKey, InformationSet]
):
    print("player 1 expected value: {}".format(ev))
    print("player 2 expected value: {}".format(-1 * ev))

    sorted_items = sorted(
        information_set_map.items(), key=lambda x: RANK_VALUES[x[0].card]
    )

    print()
    print("player 1 strategies pre-flop:")
    for iset in [
        iset
        for iset_key, iset in sorted_items
        if history_to_player(iset_key.history) == Player.PLAYER_1
        and len(iset_key.history) == 2
    ]:
        print(iset)
    print()
    print("player 1 strategies drawing:")
    for iset in [
        iset
        for iset_key, iset in sorted_items
        if history_to_player(iset_key.history) == Player.PLAYER_1
        and len(iset_key.history) == 5
    ]:
        print(iset)
    print()
    print("player 2 strategies pre-flop:")
    for iset in [
        iset
        for iset_key, iset in sorted_items
        if history_to_player(iset_key.history) == Player.PLAYER_2
        and len(iset_key.history) == 3
    ]:
        print(iset)
    print()
    print("player 2 strategies draw:")
    for iset in [
        iset
        for iset_key, iset in sorted_items
        if history_to_player(iset_key.history) == Player.PLAYER_2
        and len(iset_key.history) == 4
    ]:
        print(iset)


def main():
    """
    Run iterations of counterfactual regret minimization algorithm.
    """
    information_set_map = {}  # map of information sets
    expected_game_value_sum = 0

    for i in range(1, N_ITERATIONS + 1):
        expected_game_value_sum += cfr(
            information_set_map=information_set_map,
            history="",
            card_1=-1,
            card_2=-1,
            player_1_discarded=Multiset(),
            player_2_discarded=Multiset(),
            pr_1=1,
            pr_2=1,
            pr_c=1,
        )
        for _, v in information_set_map.items():
            v.next_strategy()
        expected_game_value = expected_game_value_sum / i
        print(f"Iteration {i}. {expected_game_value=}")
    print()

    display_results(expected_game_value, information_set_map)


if __name__ == "__main__":
    main()
