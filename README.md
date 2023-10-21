# 2to7
This repository contains code for solving heads-up 2to7 no-limit using Counterfactual Regret Minimization. It borrows heavily from the implementation in this article: https://justinsermeno.com/posts/cfr/, who wrote a very readable implementation of the CFR algorithm described by this paper: http://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf.

Currently, I've solved a push-fold strategy for 3BB (call it STACK_SIZE). The full game tree is:

Pre-flop:
1. BTN (PLAYER_1) posts 0.5BB, BB (PLAYER_2) posts 1BB.
2. PLAYER_1 can go ALL_IN (for STACK_SIZE) or FOLD. If FOLD, then payoff is (-0.5,0.5). If ALL_IN, PLAYER_2 can go ALL_IN or FOLD. If PLAYER_2 folds, then payoff is (1.0,-1.0). If both players go ALL_IN, we go post-flop.

Post-flop:
4. If both players go ALL_IN, the PLAYER_2 gets to draw any number of cards, and then the PLAYER_1 gets to draw any number of cards. 
5. Then, the hands go to showdown. If PLAYER_1 shows the lower hand, the payoff is (STACK_SIZE,-STACK_SIZE). Otherwise, the payoff is (-STACK_SIZE,STACK_SIZE). Ties are generally possible (however, not in the version with only one card dealt and one card/rank in the deck). In this case, the payoff is (0,0) for a chop.


There are several directions I can go, and I don't really know which one will be more interesting:
1. It appears that at above 3.9BB, BTN actually has negative EV in the game. My intuition is that BTN can certainly capture some positive EV with limping or smaller raises, but this requires implementing more complex game trees that may include betting after the draw.
2. Increase the number of cards dealt to each player, maybe until it gets to the real game with 5 cards.
