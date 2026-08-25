# WizardRL

## HOUSE RULES

### SUMMARY

The rules encoded in ./src/game.py (specifically _Round._winning_card_index(...) and _Round.play_card(...)), ./src/deck.py (specifically Deck.get_suit(...), Deck.player_has_suit(...), Deck.reveal_trump(...), Deck.wizard_suit, and Deck.fool_suit) and used to generate the legal moves mask (TODO: fill in methods) are intentionally different from the official rules of WIZARD. The behavior described below is the specification used by the program:

For the purposes of the follow suit rule, wizards are members of the trump suit. Their trick winning behavior remains that of a WIZARD: a wizard wins against any non wizard card.

If you wish to use this codebase, but train according to the official rules, you will need to modify the functions mentioned above. One possible approach is to apply the suit following logic used for fools to wizards, although the exact changes required will depend on the desired interpretation of the official rules.

### OFFICIAL RULES

In the official rules, wizards are treated as trumps but not considered part of the trump suit. They do not obey the follow suit rule. This means that players are not required to play them unless they can follow suit otherwise.

### WHERE THEY DIFFER

In these house rules, the follow suit rule does apply to wizards and they are considered as part of the trump suit. If a player has (a) wizard(s) but no other trump cards (and no fool(s)), and the leading suit is the trump suit, they are forced to play their wizard. In the final round (where there is no colored trump suit), the wizards form their own trump suit. During a trick, if the leading suit is the exclusive wizard trump suit, players are required to follow suit.

Fools behave according to the original rules of the game. They form the exception to the follow suit rule. They can be used at any time, even if the player could otherwise follow suit. For example, during a trick in the final round (where only wizards are trump), if a player has (a) wizard(s) and (a) fool(s), and the leading suit is wizard, then the player may still play a fool, rather than giving up their wizard.

### WHY THIS RULE WAS MODIFIED

This change was made because I consider wizards to be too powerful using the offical rules. A fun and powerful strategy in many trick taking games is to abuse a large quantity of cards in a particular suit. If a player's hand contains a large number of cards of a particular suit, they can repeatedly lead on this suit, while opponents are forced to follow until they are unable. Then the opponent's inability to follow can be exploited. Wizards by the offical rules nearly completely void this strategy. A player with a wizard can immediately take priority by deploying a wizard and play a different suit, making an extraordinary hand with a large quantity of cards in a particular suit a disadvantage rather than a strategic advantage. Furthermore, wizards cannot be forced out using the follow suit rule, meaning that they remain a threat without any meaningful counter play. In my personal opinion, the presence of fools according to the offical rules adds a nice balance, since wizards have some protection from being forced out.
