from typing import List


# TODO hypothetical function to use when there are actually 5 cards
def calculate_hand_rank(cards: List[str]):
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
    rank_count = {}
    for card in cards:
        rank = card
        if rank in rank_count:
            rank_count[rank] += 1
        else:
            rank_count[rank] = 1

    quads = list(
        reversed(sorted([rank for card, count in rank_count.items() if count == 4]))
    )
    trips = list(
        reversed(sorted([rank for card, count in rank_count.items() if count == 3]))
    )
    pairs = list(
        reversed(sorted([rank for card, count in rank_count.items() if count == 2]))
    )
    no_pairs = list(
        reversed(sorted([rank for card, count in rank_count.items() if count == 1]))
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
