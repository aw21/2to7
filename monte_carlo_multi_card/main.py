import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import pandas as pd
import numpy as np
import json
from multiset import Multiset

InformationSetKey = Tuple[str, str]

# Define game-specific constants and rules
NUM_PLAYERS = 2
STACK_SIZE = 3  # Initial stack size for both players
POSTED_POT_PLAYER1 = 0.5  # Player 1 posts 0.5BB
POSTED_POT_PLAYER2 = 1.0  # Player 2 posts 1BB
NUM_CARDS = 3  # Number of cards each player is dealt
NUM_CARDS_PER_RANK = 3
INITIAL_POT = POSTED_POT_PLAYER1 + POSTED_POT_PLAYER2
TERMINATION_THRESHOLD_PERCENTAGE = 0.001 # Percent of pot exploitability needed for termination
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
FULL_DECK = Multiset(RANKS * NUM_CARDS_PER_RANK)
FULL_DECK_AS_LIST = list(FULL_DECK)

random.seed("CS229")


def calculate_hand_rank(cards: Tuple[str, ...]) -> Tuple[Tuple[int, int], ...]:
    """
    Calculate the rank of a poker hand based on Texas Hold'em rules.

    Args:
        cards (list): A list of card ranks (e.g., ['2', '5', '7', '9', 'A']). If there are >5 cards, throw an error.

    Returns:
        int: The hand rank, where lower values indicate weaker hands.
    """
    num_cards = len(cards)
    if num_cards > 5:
        raise ValueError(
            f"Provided cards {cards}. This has length {len(cards)}, but only 5 cards or less are supported"
        )

    # Create a dictionary to count the frequency of each rank
    card_count = {}
    for card in cards:
        rank = RANK_VALUES[card]
        if rank in card_count:
            card_count[rank] += 1
        else:
            card_count[rank] = 1

    quads = list(
        reversed(sorted([rank for rank, count in card_count.items() if count == 4]))
    )
    trips = list(
        reversed(sorted([rank for rank, count in card_count.items() if count == 3]))
    )
    pairs = list(
        reversed(sorted([rank for rank, count in card_count.items() if count == 2]))
    )
    no_pairs = list(
        reversed(sorted([rank for rank, count in card_count.items() if count == 1]))
    )

    hand_rank = []
    for rank in quads:
        hand_rank.append((4, rank))
    for rank in trips:
        hand_rank.append((3, rank))
    for rank in pairs:
        hand_rank.append((2, rank))
    for rank in no_pairs:
        hand_rank.append((1, rank))

    while len(hand_rank) < num_cards:
        hand_rank.append((0, 0))

    hand_rank = tuple(hand_rank)
    return hand_rank


@dataclass(frozen=True, eq=True, order=True)
class InformationSetKey:
    cards: Tuple[str, ...]
    history: str

    def __str__(self):
        return f"{''.join(self.cards)}_{self.history}"


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
        return np.clip(x, 0, None)

    def actions_to_strategies(self):
        return {
            action: strategy
            for action, strategy in zip(self.actions, self.get_average_strategy())
        }

    def __str__(self):
        actions = self.actions
        strategies = ["{:03.2f}".format(x) for x in self.get_average_strategy()]
        actions_to_strategies = {
            action: strategy for action, strategy in zip(actions, strategies)
        }
        return "{} {}".format(str(self.key).ljust(10), actions_to_strategies)


def draw_result(current_cards: Tuple[str], cards_to_draw_from: List[str], n_cards: int) -> Tuple[Tuple[str,...], Multiset]:
    """

    Args:
        current_cards: Current cards in player's hand
        cards_to_draw_from: List of cards to draw from
        n_cards: Number of cards to draw

    Replace n cards.

    Returns:
        new cards, discarded cards

    """
    card_count = {}
    for card in current_cards:
        if card in card_count:
            card_count[card] += 1
        else:
            card_count[card] = 1

    quads = list(
        sorted([card for card, count in card_count.items() if count == 4], key=lambda x: RANK_VALUES[x])
    )
    trips = list(
        sorted([card for card, count in card_count.items() if count == 3], key=lambda x: RANK_VALUES[x])
    )
    pairs = list(
        sorted([card for card, count in card_count.items() if count == 2], key=lambda x: RANK_VALUES[x])
    )
    no_pairs = list(
        sorted([card for card, count in card_count.items() if count == 1], key=lambda x: RANK_VALUES[x])
    )


    cards_to_discard = []

    while len(cards_to_discard) < n_cards:
        if len(quads) > 0:
            cards_to_discard.append(quads[-1])
            trips.append(quads[-1])
            trips = sorted(trips, key=lambda x: RANK_VALUES[x])
            quads = quads[:-1]
            continue
        if len(trips) > 0:
            cards_to_discard.append(trips[-1])
            pairs.append(trips[-1])
            pairs = sorted(pairs, key=lambda x: RANK_VALUES[x])
            trips = trips[:-1]
            continue
        if len(pairs) > 0:
            cards_to_discard.append(pairs[-1])
            no_pairs.append(pairs[-1])
            no_pairs = sorted(no_pairs, key=lambda x: RANK_VALUES[x])
            pairs = pairs[:-1]
            continue
        if len(no_pairs) > 0:
            cards_to_discard.append(no_pairs[-1])
            no_pairs = no_pairs[:-1]
            continue

        if len(no_pairs) == 0:
            raise RuntimeError("Discarded too many cards")


    cards_to_discard = Multiset(cards_to_discard)
    cards_drawn = random.sample(cards_to_draw_from, n_cards)
    not_discarded = Multiset(current_cards) - Multiset(cards_to_discard)
    new_cards = tuple(sorted([*not_discarded, *cards_drawn], key=lambda x: RANK_VALUES[x], reverse=True))

    return new_cards, cards_to_discard



def cfr(
    *,
    information_set_map: Dict[InformationSetKey, InformationSet],
    history: str,
    cards_1: Optional[Tuple[str, ...]],
    cards_2: Optional[Tuple[str, ...]],
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
    cards_1 : int
        player A's cards. Sorted tuple
    cards_2 : int
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
        return terminal_util(history, cards_1, cards_2)

    player = history_to_player(history)
    info_set = get_info_set(
        information_set_map, cards_1 if player == Player.PLAYER_1 else cards_2, history
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
        if action.startswith(DRAW_PREFIX) and action != DRAW_PREFIX + "0":
            n_cards_to_draw = int(history_char)
            if player == Player.PLAYER_1:
                cards_to_draw_from = list(
                    FULL_DECK - Multiset(cards_1) - Multiset(cards_2) - player_2_discarded
                )
                new_cards_1, player_1_discarded = draw_result(cards_1, cards_to_draw_from, n_cards_to_draw)
                action_utils[i] = util_multiplier * cfr(
                        information_set_map=information_set_map,
                        history=next_history,
                        cards_1=new_cards_1,
                        cards_2=cards_2,
                        player_1_discarded=player_1_discarded,
                        player_2_discarded=player_2_discarded,
                        pr_1=pr_1 * strategy[i],
                        pr_2=pr_2,
                        pr_c=pr_c,
                    )
            else:
                cards_to_draw_from = list(FULL_DECK - Multiset(cards_1) - Multiset(cards_2))
                new_cards_2, player_2_discarded = draw_result(cards_2, cards_to_draw_from, n_cards_to_draw)
                action_utils[i] = util_multiplier * cfr(
                    information_set_map=information_set_map,
                    history=next_history,
                    cards_1=cards_1,
                    cards_2=new_cards_2,
                    player_1_discarded=player_1_discarded,
                    player_2_discarded=player_2_discarded,
                    pr_1=pr_1,
                    pr_2=pr_2 * strategy[i],
                    pr_c=pr_c,
                )

        else:
            if player == Player.PLAYER_1:
                action_utils[i] = util_multiplier * cfr(
                    information_set_map=information_set_map,
                    history=next_history,
                    cards_1=cards_1,
                    cards_2=cards_2,
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
                    cards_1=cards_1,
                    cards_2=cards_2,
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
    cards = random.sample(FULL_DECK_AS_LIST, 2*NUM_CARDS)
    cards_1 = tuple(sorted(cards[:NUM_CARDS], key=lambda x: RANK_VALUES[x], reverse=True))
    cards_2 = tuple(sorted(cards[NUM_CARDS:], key=lambda x: RANK_VALUES[x], reverse=True))
    expected_value = cfr(
        information_set_map=information_set_map,
        history="rr",
        cards_1=cards_1,
        cards_2=cards_2,
        player_1_discarded=Multiset(),
        player_2_discarded=Multiset(),
        pr_1=1.0,
        pr_2=1.0,
        pr_c=1.0,
    )
    return expected_value


def is_terminal(history: str):
    """
    Returns True if the history is a terminal history.
    """
    if history == "rrf" or history == "rraf" or len(history) == 6:
        return True
    return False


def terminal_util(history: str, cards_1: Tuple[str], cards_2: Tuple[str]):
    """
    Returns the utility of a terminal history from the perspective of player 1
    """
    if history == "rrf":
        return -POSTED_POT_PLAYER1
    if history == "rraf":
        return POSTED_POT_PLAYER2
    reward = STACK_SIZE if calculate_hand_rank(cards_1) < calculate_hand_rank(cards_2) else -STACK_SIZE
    return reward


def get_info_set(
    i_map: Dict[InformationSetKey, InformationSet], cards: Tuple[str, ...], history: str
) -> InformationSet:
    """
    Retrieve information set from dictionary.
    """
    key = InformationSetKey(cards, history)

    if key not in i_map:
        info_set = InformationSet(key)
        i_map[key] = info_set
        return info_set

    return i_map[key]


def export_results(
    ev: float,
    information_set_map: Dict[InformationSetKey, InformationSet],
    iterations_dict: Dict,
    file_path: Path,
):
    results = {}
    results["metadata"] = {
        "stack_size": STACK_SIZE,
        "posted_pot_p1": POSTED_POT_PLAYER1,
        "posted_pot_p2": POSTED_POT_PLAYER2,
        "num_cards": NUM_CARDS,
        "num_cards_per_rank": NUM_CARDS_PER_RANK,
        "termination_threshold_percentage": TERMINATION_THRESHOLD_PERCENTAGE
    }
    results["player_1_expected_value"] = ev
    results["player_2_expected_value"] = -ev
    results["iteration_info"] = iterations_dict

    sorted_items = sorted(
        information_set_map.items(), key=lambda x: tuple(RANK_VALUES[card] for card in x[0].cards), reverse=True
    )

    results["player_1_preflop"] = {
        str(key): value.actions_to_strategies()
        for key, value in sorted_items
        if history_to_player(key.history) == Player.PLAYER_1 and len(key.history) == 2
    }
    results["player_1_postflop"] = {
        str(key): value.actions_to_strategies()
        for key, value in sorted_items
        if history_to_player(key.history) == Player.PLAYER_1 and len(key.history) == 5
    }
    results["player_2_preflop"] = {
        str(key): value.actions_to_strategies()
        for key, value in sorted_items
        if history_to_player(key.history) == Player.PLAYER_2 and len(key.history) == 3
    }
    results["player_2_postflop"] = {
        str(key): value.actions_to_strategies()
        for key, value in sorted_items
        if history_to_player(key.history) == Player.PLAYER_2 and len(key.history) == 4
    }

    # Serializing json
    json_object = json.dumps(results, indent=2)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    # Writing to sample.json
    with open(file_path, "w") as outfile:
        outfile.write(json_object)

def main():
    """
    Run iterations of counterfactual regret minimization algorithm.
    """
    information_set_map = {}  # map of information sets
    expected_game_value_sum = 0

    i = 0
    result_path = Path(f"results/push_fold_{NUM_CARDS}_cardsperplayer_{NUM_CARDS_PER_RANK}_cardsperrank_{STACK_SIZE}BB_stacksize.json")
    nash_distance_upper_bound = np.inf
    iterations = []
    expected_values = []
    regrets = []
    nash_distances = []
    while nash_distance_upper_bound > (TERMINATION_THRESHOLD_PERCENTAGE * INITIAL_POT):
        expected_game_value_sum += cfr(
            information_set_map=information_set_map,
            history="",
            cards_1=None,
            cards_2=None,
            player_1_discarded=Multiset(),
            player_2_discarded=Multiset(),
            pr_1=1,
            pr_2=1,
            pr_c=1,
        )
        i = i + 1
        for _, v in information_set_map.items():
            v.next_strategy()
        expected_game_value = expected_game_value_sum / i
        player_1_info_sets = [info_set for key, info_set in information_set_map.items() if key.history == 'rr']
        overall_regret_upper_bound = sum([max(max(info_set.regret_sum), 0) for info_set in player_1_info_sets]) / sum(info_set.reach_pr_sum for info_set in player_1_info_sets)
        nash_distance_upper_bound = 2 * overall_regret_upper_bound
        if (i==1) or (i < 1000 and i % 100 == 0) or (i % 1000 == 0):
            print(f"Time={pd.Timestamp.now()}, Iteration {i}. {expected_game_value=}, {overall_regret_upper_bound=}, {nash_distance_upper_bound=}")
            iterations.append(i)
            expected_values.append(expected_game_value)
            regrets.append(overall_regret_upper_bound)
            nash_distances.append(nash_distance_upper_bound)
            iterations_dict = {
                "iterations": iterations,
                "expected_values": expected_values,
                "regrets": regrets,
                "nash_distances": nash_distances
            }
            export_results(
                expected_game_value,
                information_set_map,
                iterations_dict,
                result_path
            )

    print(f"Time={pd.Timestamp.now()}. Finished after {i} iterations. {expected_game_value=}, {overall_regret_upper_bound=}, {nash_distance_upper_bound=}")
    iterations.append(i)
    expected_values.append(expected_game_value)
    regrets.append(overall_regret_upper_bound)
    nash_distances.append(nash_distance_upper_bound)
    iterations_dict = {
        "iterations": iterations,
        "expected_values": expected_values,
        "regrets": regrets,
        "nash_distances": nash_distances
    }
    export_results(
        expected_game_value,
        information_set_map,
        iterations_dict,
        result_path
    )


if __name__ == "__main__":
    main()
