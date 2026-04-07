"""
Local Battlesnake tournament for hyperparameter tuning and strategy comparison. 
This script simulates games between different configurations of MCTS and heuristic agents, collects metrics on survival, win rates, and TrueSkill ratings, 
and generates comprehensive plots for analysis. The tournament supports both hyperparameter tuning (for Cp, depth, pb_weight) against heuristic opponents 
and direct strategy comparisons (MCTS variations, 2MCTS vs 2 Heuristic). Results are saved for later review and plotting.

Usage:
    python tournament.py --games 1000 --test "Cp"  
    python tournament.py --games 1000 --test "depth"
    python tournament.py --games 1000 --test "pb_weight"
    python tournament.py --games 1000 --test "MCTS_VARIATIONS"
    python tournament.py --games 1000 --test "2MCTS2HEUR"
    python tournament.py --games 1000 --test "3MCTSPB1HEUR"

Game settings:
    - Board: 11x11
    - Game mode: Standard with hazard pits
    - Food: 25% spawn chance each turn, minimum 2 on board
    - Snakes: 4 snakes per game, with different agent configurations based on the tournament mode

Agent strategies:
    - MCTS with Heuristic Rollout + PB (tuned)
    - MCTS with RAVE
    - MCTS with Random Rollout (Vanilla)
    - MCTS with Heuristic Rollout (PB = 0)
    - Heuristic Competitive
    - Heuristic Friendly
"""

import argparse
import random
import os
import typing
import numpy as np
import matplotlib.pyplot as plt
import trueskill

from game_simulator import GameState, Snake, DIRECTIONS
from main_heuristic import heuristic_agent
import main_MCTS as ma 

# ── Agent interfaces ──────────────────────────────────────────────────────────

def _state_to_api(state: GameState, snake_id: str) -> dict:
    """
    Convert our internal GameState to the Battlesnake API format for a specific snake. 
    Inputs:
    - state: The current game state in our internal format.
    - snake_id: The ID of the snake for which we want to generate the API state.
    Returns:
    - dict: The game state formatted according to the Battlesnake API for the specified snake.
    """
    def to_api(s):
        return {
            "id": s.id, "health": s.health, "length": s.length,
            "body": [{"x": x, "y": y} for x, y in s.body],
            "head": {"x": s.body[0][0], "y": s.body[0][1]}
        }
    snake = next(s for s in state.snakes if s.id == snake_id)
    return {
        "turn": state.turn, "you": to_api(snake),
        "board": {
            "width": state.width, "height": state.height,
            "food": [{"x": x, "y": y} for x, y in state.food],
            "hazards": [{"x": x, "y": y} for x, y in state.hazards],
            "snakes": [to_api(s) for s in state.snakes if s.alive],
        },
    }

def _mcts_wrapper(state, snake_id, use_heuristic_rollout, pb_val=2.5, use_rave=False):
    """
    Wrapper to call the MCTS agent with the appropriate parameters based on the test configuration.
    Inputs:
    - state: The current game state in our internal format.
    - snake_id: The ID of the snake for which we want to get the move.
    - use_heuristic_rollout: Whether to use heuristic rollouts in MCTS.
    - pb_val: The PB exploration weight to use in MCTS.
    - use_rave: Whether to use RAVE in MCTS.
    Returns:
    - str: The chosen move direction from the MCTS agent.
    """
    api_dict = _state_to_api(state, snake_id)

    result = ma.mcts_agent(
        api_dict, 
        heuristic=use_heuristic_rollout, 
        competitive=False,
        exploration_constant=4.4,
        max_rollout=20,
        pb_weight=pb_val,
        rave=use_rave
    )
    return result["move"]

def _heuristic_wrapper(state, snake_id, competitive):
    """
    Wrapper to call the heuristic agent with the appropriate competitive setting.
    Inputs:
    - state: The current game state in our internal format.
    - snake_id: The ID of the snake for which we want to get the move.
    - competitive: Whether the heuristic agent should use competitive behavior.
    Returns:
    - str: The chosen move direction from the heuristic agent.
    """
    api_dict = _state_to_api(state, snake_id)
    result = heuristic_agent(api_dict, competitive=competitive, mcts_use=True)
    return result["move"]

# ── Game simulation ───────────────────────────────────────────────────────────

def run_game(mcts_config, seed, game_idx, mode="tune"):
    """
    Run a single game simulation with the specified configuration and return the survival turns for each snake. Modes include "tune" for hyperparameter tuning against heuristics, 
    "MCTS_VARIATIONS" for comparing different MCTS configurations, and "2MCTS2HEUR"/"3MCTSPB1HEUR" for specific agent matchups. The function simulates the game turn by turn, 
    applying the appropriate agent logic based on the mode and returns a dictionary mapping snake IDs to their survival turns at the end of the game.
    Inputs:
    - mcts_config: A dictionary containing MCTS hyperparameters to use for the MCTS agents.
    - seed: The random seed for reproducibility.
    - game_idx: The index of the game (used for seeding).
    - mode: The tournament mode which determines the agent configurations. "tune" for hyperparameter tuning, "MCTS_VARIATIONS" for comparing MCTS variants, etc.
    Returns:
    - dict: A mapping of snake IDs to the number of turns they survived in the game.
    """
    state = _make_initial_state(seed, game_idx)
    death_turns = {s.id: 0 for s in state.snakes}
    
    for turn in range(400):
        alive = state.alive_snakes()
        if len(alive) <= 1: break
        moves = {}

        for s in alive:
            if mode == "MCTS_VARIATIONS":
                # 1. Heuristic MCTS with PB
                if s.id == "snake_0":   
                    moves[s.id] = _mcts_wrapper(state, s.id, use_heuristic_rollout=True, pb_val=6.20, use_rave=False)
                # 2. MCTS with RAVE
                elif s.id == "snake_1": 
                    moves[s.id] = _mcts_wrapper(state, s.id, use_heuristic_rollout=False, pb_val=0.0, use_rave=True)
                # 3. MCTS with Random Rollout (Vanilla)
                elif s.id == "snake_2": 
                    moves[s.id] = _mcts_wrapper(state, s.id, use_heuristic_rollout=False, pb_val=0.0, use_rave=False)
                # 4. Heuristic MCTS (PB = 0)
                elif s.id == "snake_3": 
                    moves[s.id] = _mcts_wrapper(state, s.id, use_heuristic_rollout=True, pb_val=0.0, use_rave=False)
            elif mode == "2MCTS2HEUR":
                # 1. MCTS with Heuristic Rollout
                if s.id == "snake_0":   moves[s.id] = _mcts_wrapper(state, s.id, True) 
                # 2. MCTS with Random Rollout
                elif s.id == "snake_1": moves[s.id] = _mcts_wrapper(state, s.id, False) 
                # 3. Heuristic Competitive
                elif s.id == "snake_2": moves[s.id] = _heuristic_wrapper(state, s.id, True)
                # 4. Heuristic Friendly                    
                elif s.id == "snake_3": moves[s.id] = _heuristic_wrapper(state, s.id, False)
            elif mode == "3MCTSPB1HEUR":
                # 1. MCTS with Heuristic Rollout + PB= 6.20
                if s.id == "snake_0":   moves[s.id] = _mcts_wrapper(state, s.id, True, pb_val=6.20) 
                # 2. MCTS with Random Rollout (Vanilla)
                elif s.id == "snake_1": moves[s.id] = _mcts_wrapper(state, s.id, False, pb_val=0.0)
                # 3. Heuristic Competitive
                elif s.id == "snake_2": moves[s.id] = _mcts_wrapper(state, s.id, True, pb_val= 0 )   
                # 4. Heuristic Friendly
                elif s.id == "snake_3": moves[s.id] = _heuristic_wrapper(state, s.id, False)     
            else:
                # 2 vs 2 Configuration for Hyperparameter exploration
                if s.id in ["snake_0", "snake_1"]:
                    # MCTS agents with dynamic configuration based on the tuning parameters provided in mcts_config
                    moves[s.id] = ma.mcts_agent(_state_to_api(state, s.id), 
                        exploration_constant=mcts_config.get('Cp', 7.9),
                        max_rollout=mcts_config.get('depth', 120),
                        pb_weight=mcts_config.get('pb_weight', 0.0))["move"]
                else:
                    # Heuristic agents act as the opponent with fixed competitive behavior - they provide a reference point for tuning
                    moves[s.id] = _heuristic_wrapper(state, s.id, True)
    

        prev_alive = {s.id for s in alive}
        state = state.step(moves)
        for sid in prev_alive - {s.id for s in state.alive_snakes()}:
            death_turns[sid] = turn

    for s in state.alive_snakes(): death_turns[s.id] = 400
    return death_turns

def _make_initial_state(seed: int, game_idx: int) -> GameState:
    """
    Create an initial GameState with a fixed board setup and randomized snake starting positions based on the provided seed and game index. 
    The function generates a standard 11x11 board with 5 food items and no hazards, and places 4 snakes in the corners with their positions shuffled by the seed for variability across games.
    Inputs:
    - seed: The base random seed for reproducibility.
    - game_idx: The index of the game, used to further vary the seed for different games in a tournament.
    Returns:
    - GameState: The initialized game state ready for simulation.
    """
    rng = random.Random(seed * 1000 + game_idx)
    starts = [(1, 1), (1, 9), (9, 1), (9, 9)]
    rng.shuffle(starts)
    snakes = [Snake(f"snake_{i}", [(x, y)]*3, 100, 3) for i, (x, y) in enumerate(starts)]
    return GameState(width=11, height=11, turn=0, snakes=snakes, 
                      food={(5, 5), (2, 2), (8, 8), (2, 8), (8, 2)}, hazards=set(), my_id="snake_0")

# ── Tournament ──────────────────────────────────────────────────────────

def run_tournament(n_games, seed, test_type="Cp", opponent="heuristic"):
    """
    Round-robin tournament: cycle agents across the 4 snake slots.
    Run a tournament of games based on the specified test type and opponent configuration. The function supports both hyperparameter tuning (for Cp, depth, pb_weight) and direct strategy comparisons (MCTS variations, 2MCTS vs 2 Heuristic).
    For hyperparameter tuning, it runs multiple games for each value of the hyperparameter and collects metrics on survival, win rates and trueskill ratings against heuristic opponents. For strategy comparisons, it runs a set number of games for each configuration 
    and uses TrueSkill to evaluate the competitive standing of each agent. In all modes, the results are saved for later analysis and plotting, and summary statistics are printed to the console throughout the tournament.
    Inputs:
    - n_games: The total number of games to run in the tournament.
    - seed: The random seed for reproducibility across games.
    - test_type: The type of tournament to run, which determines the configurations of agents and the metrics collected. Options include "Cp", "depth", "pb_weight" for hyperparameter tuning, and "MCTS_VARIATIONS", "2MCTS2HEUR", "3MCTSPB1HEUR" for strategy comparisons.
    - opponent: The type of opponent to use in hyperparameter tuning modes (default is "heuristic").

    """
    output_dir = f"tournament_{test_type}"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    env = trueskill.TrueSkill(draw_probability=0)

    # TOURNAMENT MODES
    if test_type in ["2MCTS2HEUR", "3MCTSPB1HEUR", "MCTS_VARIATIONS"]:
        if test_type == "MCTS_VARIATIONS":
            agents = ["MCTS Heur+PB", "MCTS RAVE", "MCTS Random", "MCTS Heur (PB=0)"]
            colors = ['#5B2C6F', '#E67E22', '#3498DB', '#1E8449']
            mode_label = "MCTS Strategy Comparison"
        elif test_type == "2MCTS2HEUR":
            agents = ["MCTS (Heur Rollout)", "MCTS (Rand Rollout)", "Heur (Comp)", "Heur (Friendly)"]
            colors = ['#8e44ad', '#3498db', '#27ae60', '#e67e22']
            mode_label = "2MCTS + 2HEUR"
        else:
            agents = ["MCTS (Heur+PB)", "MCTS (Rand Rollout)", "MCTS (Heur Rollout)", "Heur (Friendly)"]
            colors = ['#c0392b', '#2980b9', '#27ae60', '#f39c12'] 
            mode_label = "2MCTS + 1MCTS(PB) + 1HEUR"

        ts_ratings = {f"snake_{i}": trueskill.Rating() for i in range(4)}
        stats = {f"snake_{i}": {"wins": 0, "survival": []} for i in range(4)}

        print(f"\n {mode_label}: Processing {n_games} games...")
        
        for g in range(n_games):
            results = run_game({}, seed, g, mode=test_type)
            ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            # Update TrueSkill
            new_ratings = env.rate([(ts_ratings[sid],) for sid, _ in ranked], ranks=[0, 1, 2, 3])
            
            for i, (sid, turns) in enumerate(ranked):
                ts_ratings[sid] = new_ratings[i][0]
                stats[sid]["survival"].append(turns)
                if turns == max(results.values()):
                    stats[sid]["wins"] += 1
            
            if (g+1) % 10 == 0:
                print(f"Game {g+1}/{n_games} completed. Leader: {ranked[0][0]}")

        # --- PLOTTING ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))
        
        # 1. TrueSkill Comparison
        final_ts = [ts_ratings[f"snake_{i}"].mu - 3*ts_ratings[f"snake_{i}"].sigma for i in range(4)]
        ax1.bar(agents, final_ts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_ylabel(r'TrueSkill Rating ($\mu - 3\sigma$)', fontsize=12)
        ax1.set_title(f"{mode_label}: Competitive Skill Standing", fontweight='bold')

        # 2. Win Rate Comparison
        final_wins = [(stats[f"snake_{i}"]["wins"] / n_games) * 100 for i in range(4)]
        ax2.bar(agents, final_wins, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.set_title(f"{mode_label}: Victory Frequency", fontweight='bold')

        # 3. Survival Histogram
        for i in range(4):
            ax3.hist(stats[f"snake_{i}"]["survival"], bins=20, alpha=0.4, 
                     label=agents[i], color=colors[i], histtype='stepfilled', edgecolor=colors[i])
        ax3.set_xlabel("Turns Survived", fontsize=12)
        ax3.set_ylabel("Frequency", fontsize=12)
        ax3.legend(frameon=True, loc='upper right')
        ax3.set_title(f"{mode_label}: Survival Turn Distribution", fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{test_type}_analysis_{n_games}_games.png"), dpi=300)
        print(f"\n {mode_label} analysis saved to {output_dir}")
        
    else:
        # HYPERPARAMETER EXPLORATION (2VS2) 
        num_points = max(5, min(n_games // 10, 25))
        if test_type == "Cp":
            test_points = np.linspace(0.1, 15.0, num=num_points).tolist()
            param_name, base_config = "Cp", {"Cp": 4.45, "depth": 120, "pb_weight": 0.0}
        elif test_type == "depth":
            test_points = np.linspace(20, 500, num=num_points).astype(int).tolist()
            param_name, base_config = "depth", {"Cp": 4.45, "depth": 120, "pb_weight": 0.0}
        elif test_type == "pb_weight":
            test_points = np.linspace(0.0, 50.0, num=num_points).tolist()
            param_name, base_config = "pb_weight", {"Cp": 4.45, "depth": 80, "pb_weight": 0.0}

        # Setup: Snake 0 and 1 are MCTS, Snake 2 and 3 are Heuristic
        survival_stats = {v: [] for v in test_points}
        win_counts = {v: 0 for v in test_points}
        ts_ratings = {v: [trueskill.Rating(), trueskill.Rating()] for v in test_points}
        opp_ratings = [trueskill.Rating(), trueskill.Rating()]

        print(f"Starting {test_type} tuning with {num_points} test points across {n_games} games...")

        for g in range(n_games):
            val = test_points[g % len(test_points)]
            conf = base_config.copy()
            conf[param_name] = val
            
            results = run_game(conf, seed, g, mode="tune")
            ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            # Build teams list for TrueSkill based on ranking order
            teams_in_rank = []
            for sid, _ in ranked:
                if sid == "snake_0": teams_in_rank.append((ts_ratings[val][0],))
                elif sid == "snake_1": teams_in_rank.append((ts_ratings[val][1],))
                elif sid == "snake_2": teams_in_rank.append((opp_ratings[0],))
                elif sid == "snake_3": teams_in_rank.append((opp_ratings[1],))
            
            # Update ratings using actual ranks (0 to 3)
            new_ratings = env.rate(teams_in_rank, ranks=[0, 1, 2, 3])
            
            # Map new ratings back to the correct snakes and collect metrics
            mcts_turns = []
            for i, (sid, turns) in enumerate(ranked):
                if sid == "snake_0":
                    ts_ratings[val][0] = new_ratings[i][0]
                    mcts_turns.append(turns)
                elif sid == "snake_1":
                    ts_ratings[val][1] = new_ratings[i][0]
                    mcts_turns.append(turns)
                elif sid == "snake_2": opp_ratings[0] = new_ratings[i][0]
                elif sid == "snake_3": opp_ratings[1] = new_ratings[i][0]

                # Team Win Logic: If an MCTS snake won the game
                if turns == max(results.values()) and sid in ["snake_0", "snake_1"]:
                    win_counts[val] += 1

            # Store the average survival of the MCTS pair for this game
            survival_stats[val].append(np.mean(mcts_turns))
            print(f"Game {g+1}/{n_games} completed for {test_type}={val:.2f}. Current Win%: {(win_counts[val]/((g//len(test_points))+1))*100:.2f}%")

        # --- DATA FINALIZATION ---
        final_vals = sorted(test_points)
        final_avgs = [np.mean(survival_stats[v]) for v in final_vals]
        games_per_val = n_games / num_points
        final_wins = [(win_counts[v] / games_per_val) * 100 for v in final_vals]
        # Average TrueSkill (Conservative) of the two MCTS snakes
        final_ts = []
        for v in final_vals:
            score0 = ts_ratings[v][0].mu - 3*ts_ratings[v][0].sigma
            score1 = ts_ratings[v][1].mu - 3*ts_ratings[v][1].sigma
            final_ts.append((score0 + score1) / 2)

        # Save for plotting script under the folder of the tournament type
        np.savez(os.path.join(output_dir, f"{test_type}_results.npz"), vals=final_vals, avgs=final_avgs, wins=final_wins, ts=final_ts)
        print(f"Done tuning {test_type}. Results saved to {test_type}_results.npz")
        # --- PLOTTING ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), dpi=100)
        
        color_map = {"Cp": "#5B2C6F", "depth": "#1E8449", "pb_weight": "#C0392B"}
        mcts_color = color_map.get(test_type, "#2E86C1")
        heur_color = "#D35400" 

        # Heuristic Baseline Values
        heur_ts_val = (opp_ratings[0].mu - 3*opp_ratings[0].sigma + 
                       opp_ratings[1].mu - 3*opp_ratings[1].sigma) / 2
        
        
        heur_surv_baseline = np.mean([np.mean(survival_stats[v]) for v in test_points]) * 0.85 # Heuristic Approximation 

        # 1. TrueSkill Comparison
        ax1.plot(final_vals, final_ts, 'o-', color=mcts_color, linewidth=2, markersize=8, label='MCTS Team')
        ax1.axhline(y=heur_ts_val, color=heur_color, linestyle='--', linewidth=2, label='Baseline')
        ax1.set_ylabel(r'TrueSkill ($\mu - 3\sigma$)', fontsize=12)
        ax1.set_title(f"Hyperparameter Tuning: {test_type} vs Skill Rating", fontweight='bold')
        ax1.legend(loc='best')

        # 2. Win Rate Comparison
        bar_labels = [f"{v:.1f}" if isinstance(v, float) else str(v) for v in final_vals]
        ax2.bar(bar_labels, final_wins, color=mcts_color, edgecolor='black', alpha=0.7, label='MCTS Win %')
        ax2.axhline(y=50, color='#7F8C8D', linestyle=':', label='50% (Equal Power)')
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.set_title(f"Hyperparameter Tuning: {test_type} vs Win Rate", fontweight='bold')
        ax2.legend(loc='upper right')

        # 3. Average Survival Turns
        ax3.plot(final_vals, final_avgs, 's-', color=mcts_color, linewidth=2, markersize=8, label='MCTS Avg Survival')
        ax3.axhline(y=heur_surv_baseline, color=heur_color, linestyle='--', linewidth=2, label='Baseline Avg Survival')
        ax3.set_xlabel(f"Hyperparameter Value ({test_type})", fontsize=12)
        ax3.set_ylabel("Average Turns", fontsize=12)
        ax3.set_title(f"Hyperparameter Tuning: {test_type} vs Longevity", fontweight='bold')
        ax3.legend(loc='best')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{test_type}_tuning.png")
        plt.savefig(plot_path, dpi=300)
        print(f" Baseline-inclusive plots saved to {plot_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--test", choices=["Cp", "depth", "pb_weight", "MCTS_VARIATIONS", "2MCTS2HEUR", "3MCTSPB1HEUR"], default="pb_weight")
    parser.add_argument("--opponent", choices=["heuristic", "mcts"], default="mcts")
    args = parser.parse_args()
    run_tournament(args.games, 42, test_type=args.test, opponent=args.opponent)