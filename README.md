# TCG

### Projects:  
1. Write a simple player to play Threes!.  
    * Time limit: 0.001 seconds/move  
    * Hint: use some simple heuristics to play; no search is required.  
2. Write a player to play Threes! with high winrates.  
    * Time limit: 0.01 second/move  
    * Hint: use TD learning only. No search is required. 
    ```bash
    ./threes --total=1000 --block=1000 --limit=1000 --play="alpha=0 load=weights.bin" --save stats.txt
    ```
    ```bash
    ./threes-judge --load stats.txt --judge version=2
    ```
3. Increase the winrate of the player.  
    * Time limit: 5 seconds/move (to be modified)  
    * Hint: incorporate expectimax search into the player.
    ```bash
    ./threes --total=1000 --block=1000 --limit=1000 --play="alpha=0 load=weights.bin" --save stats.txt
    ```
4. Write a MCTS program for the game of Hollow NoGo .  
    * Beat a weak program given by TA.
    ```bash
    # commands for player 1
    P1B='./nogo --shell --name="Hollow-Black" --black="mcts count=500 rootParallel"'
    P1W='./nogo --shell --name="Hollow-White" --white="mcts count=500 rootParallel"'
    ```
    ```bash
    rm -rf gogui-twogtp-2022*
    ```
5. Improve the above MCTS program to attend the final tournament.  
    * Hint: no limitation about the method you use. E.g., you may use AlphaZero to do it.  
