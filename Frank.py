import math
import time
import random
import typing
import argparse
from collections import defaultdict, deque


move_counter = 0
def manhattan_dist(a, b):
        return abs(a['x'] - b['x']) + abs(a['y'] - b['y'])

def count_open_space(start_pos, body_coords, opponent_coords, hazard_coords, board_width, board_height, max_depth=5):
    """
    Breadth-first search to count the number of reachable open tiles from a starting position.
    Avoids the snake's own body, opponent bodies, and hazards. Used to estimate how much space is available for a move.

    Inputs:
    -start_pos (dict): Starting position {"x": int, "y": int}
    - body_coords (list): List of dicts for your snake's body segments
    - opponent_coords (set): Set of (x, y) tuples for all opponent body segments
    - hazard_coords (set): Set of (x, y) tuples for all hazard tiles
    - board_width (int): Board width
    - board_height (int): Board height
    - max_depth (int): Maximum search depth (default 5)

    Returns:
    - int: Number of reachable open tiles (including start_pos)
    """
    visited = set()
    queue = deque()
    queue.append((start_pos['x'], start_pos['y'], 0))
    while queue:
        x, y, depth = queue.popleft()
        if (x, y) in visited or depth > max_depth:
            continue
        visited.add((x, y))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < board_width and 0 <= ny < board_height:
                if {'x': nx, 'y': ny} in body_coords:
                    continue
                if (nx, ny) in opponent_coords:
                    continue
                if (nx, ny) in hazard_coords:
                    continue
                queue.append((nx, ny, depth+1))
    return len(visited)

def heuristic_agent(game_state: typing.Dict, competitive: bool = False, mcts_use = False) -> typing.Dict:
    """
    A heuristic-based Battlesnake agent that evaluates moves based on safety, proximity to food, and open space. Movers are based on a series of rules to avoid hazards, prevent collisions, and target food effectively.
    In competitive mode we prioritize food that is farther from opponents, while in peaceful mode we target the closest food. The agent also considers open space to avoid getting trapped.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    - competitive (bool): Whether to play competitively (default False).
    - mcts_use (bool): Whether to use MCTS for move selection (default False).
    Returns:
    - dict: The chosen move direction."""
    global move_counter
    move_counter += 1
    # Step 1: Avoid moving backwards
    hazards = game_state['board'].get('hazards', [])
    hazard_coords = set((hazard['x'], hazard['y']) for hazard in hazards)
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    my_head = game_state["you"]["body"][0]
    my_neck = game_state["you"]["body"][1]
    if my_neck["x"] < my_head["x"]:
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:
        is_move_safe["up"] = False

    # Step 2: Prevent moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    if my_head['x'] == 0:
        is_move_safe['left'] = False
    if my_head['x'] == board_width - 1:
        is_move_safe['right'] = False
    if my_head['y'] == 0:
        is_move_safe['down'] = False
    if my_head['y'] == board_height - 1:
        is_move_safe['up'] = False

    # Step 3: Prevent self-collision
    my_body = game_state['you']['body']
    next_positions = {
        "up":    {"x": my_head["x"],     "y": my_head["y"] + 1},
        "down":  {"x": my_head["x"],     "y": my_head["y"] - 1},
        "left":  {"x": my_head["x"] - 1, "y": my_head["y"]},
        "right": {"x": my_head["x"] + 1, "y": my_head["y"]},
    }
    body_coords = [{"x": segment["x"], "y": segment["y"]} for segment in my_body]
    for move_dir, pos in next_positions.items():
        if pos in body_coords:
            is_move_safe[move_dir] = False

    # Step 4: Prevent collision with other snakes
    opponents = game_state['board']['snakes']
    opponent_coords = set()
    for snake in opponents:
        for segment in snake['body']:
            opponent_coords.add((segment['x'], segment['y']))
    for move_dir, pos in next_positions.items():
        if (pos['x'], pos['y']) in opponent_coords:
            is_move_safe[move_dir] = False

    # Step 5: Avoid head-to-head collisions (competitive: only if not longer)
    my_length = len(game_state['you']['body'])
    opponent_heads = [snake['body'][0] for snake in opponents if snake['id'] != game_state['you']['id']]
    opponent_lengths = {snake['id']: len(snake['body']) for snake in opponents if snake['id'] != game_state['you']['id']}
    for move_dir, pos in next_positions.items():
        for opp_head in opponent_heads:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                opp_next = {'x': opp_head['x'] + dx, 'y': opp_head['y'] + dy}
                if 0 <= opp_next['x'] < board_width and 0 <= opp_next['y'] < board_height:
                    if opp_next == pos:
                        opp_snake = [snake for snake in opponents if snake['body'][0] == opp_head]
                        if competitive:
                            if opp_snake and opponent_lengths[opp_snake[0]['id']] >= my_length:
                                is_move_safe[move_dir] = False
                        else:
                            is_move_safe[move_dir] = False
    # Step 6: Build list of safe and hazard moves
    safe_moves = []
    hazard_moves = []
    for move_dir, isSafe in is_move_safe.items():
        if isSafe:
            pos = next_positions[move_dir]
            if (pos['x'], pos['y']) in hazard_coords:
                hazard_moves.append(move_dir)
            else:
                safe_moves.append(move_dir)
    # Prefer safe moves, but use hazard moves if no other options
    candidate_moves = safe_moves if safe_moves else hazard_moves

    # Step 7: If no moves are possible, default to down
    if not candidate_moves:
        if mcts_use == False:
            print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Step 8: Food targeting logic
    food = game_state['board']['food']
    closest_food = None
    if competitive:
        # In competitive mode, target food farthest from opponents
        best_food = None
        max_min_opp_dist = -1
        for f in food:
            my_dist = manhattan_dist(my_head, f)
            if opponent_heads:
                min_opp_dist = min(manhattan_dist(opp_head, f) for opp_head in opponent_heads)
            else:
                min_opp_dist = 0
            if min_opp_dist > max_min_opp_dist or (min_opp_dist == max_min_opp_dist and (best_food is None or my_dist < manhattan_dist(my_head, best_food))):
                max_min_opp_dist = min_opp_dist
                best_food = f
        closest_food = best_food
    else:
        # In peaceful mode, target closest food
        min_dist = float('inf')
        for f in food:
            dist = manhattan_dist(my_head, f)
            if dist < min_dist:
                min_dist = dist
                closest_food = f

    # Step 9: Score moves by open space and food distance
    move_scores = []
    for move_dir in candidate_moves:
        pos = next_positions[move_dir]
        temp_body = body_coords[:-1] if pos not in food else body_coords
        open_space = count_open_space(pos, temp_body, opponent_coords, hazard_coords, board_width, board_height)
        food_dist = manhattan_dist(pos, closest_food) if closest_food else 0
        move_scores.append((open_space, -food_dist, move_dir, food_dist))

    # Step 10: Choose best move based on health and scoring
    health = game_state['you']['health']
    if (competitive and health <= 60 and closest_food) or (not competitive and health <= 40 and closest_food):
        # When low on health, prioritize food
        min_food_dist = min(s[3] for s in move_scores)
        food_moves = [s for s in move_scores if s[3] == min_food_dist]
        max_open = max(s[0] for s in food_moves)
        best_moves = [s[2] for s in food_moves if s[0] == max_open]
        next_move = random.choice(best_moves)
    else:
        # Otherwise, prefer open space, then food
        if move_scores:
            max_open = max(s[0] for s in move_scores)
            open_moves = [s for s in move_scores if s[0] == max_open]
            best_score = max(s[1] for s in open_moves)
            best_moves = [s[2] for s in open_moves if s[1] == best_score]
            next_move = random.choice(best_moves)
        else:
            next_move = random.choice(candidate_moves)

    # Step 11: Return the chosen move
    if mcts_use == False:
        print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) representing a game state. Each node tracks its parent, children, the move that led to it, and statistics for wins and visits. The node also maintains a list of untried moves, 
    which are shuffled to ensure exploration diversity during expansion. The uct_select method implements the Upper Confidence Bound for Trees (UCT) formula to balance exploration and exploitation when selecting child nodes. 
    The update method updates the node's statistics based on the simulation results. 
    Inputs:
    - state (dict): The game state associated with this node.
    - parent (MCTSNode): The parent node in the search tree (default None).
    - move (str): The move that led to this node (default None).
    """
    def __init__(self, state, parent=None, move=None, is_opponent_turn=False):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.is_opponent_turn = is_opponent_turn 
        
        # RAVE/AMAF Statistics
        self.rave_visits = defaultdict(int)  # Key: move string, Value: count
        self.rave_wins = defaultdict(float)  # Key: move string, Value: sum of scores
        
        # Improvement: Progressive History / Heuristic Ranking
        # We initialize untried moves. If we want the teammate's ranking, 
        # we'll sort them during the first expansion step.
        self.untried_moves = ["up", "down", "left", "right"]
        random.shuffle(self.untried_moves)
        self.heuristic_score = 0.0

    def uct_select(self, exploration_constant=1.41, pb_weight=0.0, rave=False, rave_k=100):
        """
        Selection logic incorporating UCB1, Progressive Bias, and optionally RAVE.
        """
        log_parent = math.log(max(1, self.visits))

        def score_function(c):
            if c.visits == 0:
                return float('inf')
            
            # 1. Standard MCTS (Exploitation)
            q = c.wins / c.visits
            
            # 2. RAVE Integration
            if rave:
                rv = self.rave_visits[c.move]
                if rv > 0:
                    q_rave = self.rave_wins[c.move] / rv
                    # Beta parameter: decays as visits increase
                    beta = math.sqrt(rave_k / (3 * c.visits + rave_k))
                    q = (1 - beta) * q + beta * q_rave

            # 3. Exploration (UCB1)
            exploration = exploration_constant * math.sqrt(log_parent / c.visits)
            
            # 4. Progressive Bias
            bias = 0.0
            if pb_weight > 0:
                bias = (pb_weight * c.heuristic_score) / (c.visits + 1)
                
            return q + exploration + bias

        return max(self.children, key=score_function)

    def update(self, score):
        self.visits += 1
        self.wins += score

def rollout(state, max_rollout=80):
    """
    Simulates a random playout from the given state up tp the rollout depth or until a terminal state is reached. The function returns a heuristic score based on survival and length growth during the rollout.
    The survival component rewards the snake for staying alive for more turns, while the length component encourages growth and maintaining a longer body. The score is normalized to be between 0.0 and 1.0, 
    where higher scores indicate better outcomes for the snake. Contrary to the vanilla mcts rollout which only considers survival (+1) or loss (0), this enhanced rollout incorporates a more nuanced evaluation 
    of the snake's performance during the simulation, providing a richer signal for the MCTS algorithm to learn from.
    """
    if state is None: return 0
    
    curr = state
    steps = 0
    start_length = len(state['you']['body'])
    
    while steps < max_rollout:
        if is_terminal(curr):
            break
            
        valid_moves = []
        for m in ["up", "down", "left", "right"]:
            s = get_next_state(curr, m)
            if s is not None: 
                valid_moves.append(s)
            
        if not valid_moves: 
            break 
        
        #Random move selection
        curr = random.choice(valid_moves)
        steps += 1

    reward_score = reward(curr, steps, max_rollout, start_length)
    return reward_score
def rollout_heuristic(state, competitive: bool):
    """
    Simulates a playout guided by the heuristic agent from the given state up tp the rollout depth or until a terminal state is reached. The function returns a heuristic score based on survival and length growth during the rollout.
    The survival component rewards the snake for staying alive for more turns, while the length component encourages growth and maintaining a longer body. The score is normalized to be between 0.0 and 1.0, 
    where higher scores indicate better outcomes for the snake. Contrary to the vanilla mcts rollout which only considers survival (+1) or loss (0), this enhanced rollout incorporates a more nuanced evaluation 
    of the snake's performance during the simulation, providing a richer signal for the MCTS algorithm to learn from.
    """
    if state is None: return 0
    
    curr = state
    steps = 0
    max_rollout = 80 
    start_length = len(state['you']['body'])
    
    while steps < max_rollout:
        if is_terminal(curr):
            break
            
        # Instead of random moves, use the heuristic agent to guide the rollout
        if competitive:
            heuristic_result = heuristic_agent(curr, competitive=competitive, mcts_use=True)
        else:
            heuristic_result = heuristic_agent(curr, competitive=False, mcts_use=True)
        move = heuristic_result["move"]
        
        next_s = get_next_state(curr, move)
        if next_s is None: 
            break 
            
        curr = next_s
        steps += 1
    reward_score = reward(curr, steps, max_rollout, start_length)
    
    return reward_score

def rollout_with_history(state, max_rollout, heuristic, competitive):
    """Modified rollout that returns the score AND the list of moves made."""
    if state is None: return 0, []
    
    curr = state
    steps = 0
    start_length = len(state['you']['body'])
    move_history = []
    
    while steps < max_rollout:
        if is_terminal(curr): break
        
        if heuristic:
            res = heuristic_agent(curr, competitive=competitive, mcts_use=True)
            move = res["move"]
        else:
            valid = ["up", "down", "left", "right"]
            move = random.choice(valid)
            
        move_history.append(move)
        next_s = get_next_state(curr, move)
        if next_s is None: break
        curr = next_s
        steps += 1
        
    return reward(curr, steps, max_rollout, start_length), move_history

def reward(curr, steps, max_rollout, start_length):
    """
    Calculates a heuristic reward score based on the final state of the rollout, the number of steps taken, and the initial length of the snake. The reward is designed to encourage both survival and growth during the rollout.
    The survival component provides a significant portion of the reward, incentivizing the snake to stay alive for as long as possible. The length component rewards the snake for maintaining a longer body and for growth
    during the rollout, which can be crucial for success in Battlesnake. The final score is normalized to be between 0.0 and 1.0, providing a consistent scale for the MCTS algorithm to evaluate different moves based on their simulated outcomes.
    Inputs:
    - curr (dict): The final game state at the end of the rollout.
    - steps (int): The number of steps taken during the rollout.
    - max_rollout (int): The maximum number of steps allowed in the rollout.
    - start_length (int): The initial length of the snake at the start of the rollout.
    Returns:
    - float: A heuristic reward score between 0.0 and 1.0, where higher scores indicate better outcomes for the snake.
    """
    if not is_terminal(curr):
        survival_reward = 0.8
    else:
        survival_reward = (steps / max_rollout) * 0.4
    
    current_length = len(curr['you']['body'])
    growth = current_length - start_length
    length_reward = min(current_length / 20, 1.0) * 0.15 
    growth_reward = min(growth, 5) / 5 * 0.05
    return max(0, min(1.0, survival_reward + length_reward + growth_reward))

def get_next_state(game_state, my_move):
    """
    Get the next game state resulting from applying the given move to the current game state. The function checks for collisions with walls, hazards, and other snakes, as well as food consumption and health updates.
    If the move results in a collision or the snake's health drops to zero or below, the function returns None, indicating a terminal state. Otherwise, it returns the new game state after applying the move,
    including the updated position of the snake's head, body, and health. This function is crucial for simulating potential future states during the MCTS algorithm's expansion and simulation phases. 

    This is different than the vanilla mcts, here we consider stationary oponents and hazards, which allows us to prune "Death" states early in the search tree, improving efficiency and decision quality.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    - my_move (str): The move to apply ("up", "down", "left", "right").
    Returns:
    - dict or None: The new game state after applying the move, or None if the move results in a terminal state (collision or death).
    """
    if game_state is None: return None
    
    # Fast shallow copy
    new_state = game_state.copy()
    my_snake = game_state['you'].copy()
    my_snake['body'] = list(game_state['you']['body'])
    new_state['you'] = my_snake
    
    head = my_snake['head']
    if my_move == "up":    new_head_tup = (head["x"], head["y"] + 1)
    elif my_move == "down":  new_head_tup = (head["x"], head["y"] - 1)
    elif my_move == "left":  new_head_tup = (head["x"] - 1, head["y"])
    else:                    new_head_tup = (head["x"] + 1, head["y"])

    new_head_dict = {"x": new_head_tup[0], "y": new_head_tup[1]}

    # Wall Collision
    if (new_head_tup[0] < 0 or new_head_tup[0] >= game_state['board']['width'] or 
        new_head_tup[1] < 0 or new_head_tup[1] >= game_state['board']['height']):
        return None

    # Obstacle Collision
    if new_head_tup in game_state['_obstacle_set']:
        if any(seg == new_head_dict for seg in my_snake['body'][:-1]):
            return None
        for snake in game_state['board']['snakes']:
            if snake['id'] == my_snake['id']: continue
            if new_head_dict == snake['head'] and len(my_snake['body']) <= len(snake['body']):
                return None
            elif any(seg == new_head_dict for seg in snake['body']):
                return None

    # Hazard/Health
    if new_head_tup in game_state['_hazard_set']:
        my_snake['health'] -= 15 
    else:
        my_snake['health'] -= 1
    
    if my_snake['health'] <= 0: return None

    # Food
    if new_head_tup in game_state['_food_set']:
        my_snake['health'] = 100
    else:
        my_snake['body'].pop()

    my_snake['body'].insert(0, new_head_dict)
    my_snake['head'] = new_head_dict
    new_state['turn'] = game_state.get('turn', 0) + 1
    return new_state

def is_terminal(state):
    """
    Checks if the given game state is terminal, meaning the snake has collided with a wall, hazard, or another snake, or if its health has dropped to zero. 
    The function returns True if the state is terminal (death) and False otherwise.
    """
    if state is None: return True
    my_snake = state['you']
    head = my_snake['head']
    if (head['x'] < 0 or head['x'] >= state['board']['width'] or 
        head['y'] < 0 or head['y'] >= state['board']['height']):
        return True
    if my_snake['health'] <= 0 or state.get('turn', 0) >= 300:
        return True
    return False

def mcts_agent(game_state, heuristic=False, competitive=False, 
               exploration_constant=4.4, max_rollout=20, 
               pb_weight=6.2, rave=True) -> typing.Dict:
    """
    An implementation of a Monte Carlo Tree Search (MCTS) agent for Battlesnake. The agent builds a search tree of game states, simulates random playouts to evaluate potential moves, 
    and selects the move that leads to the most promising outcomes based on visit counts. The MCTS agent is operating within an 800ms time budget. The final move decision is based on 
    the child node with the highest visit count, which indicates the most explored and potentially best move.
    The agent consists of the following main components:
    1. Selection: Traverses the tree using the UCT formula to select child nodes until it reaches a node with untried moves or a terminal state.
    2. Expansion: If the selected node has untried moves, it expands the tree by creating a new child node for one of the untried moves.
    3. Simulation: Performs playout from the new node until a rollout depth is reached to evaluate the outcome of that move. When the heuristic flag is set to True, the simulation is guided by a heuristic agent instead of random moves,
                   which can lead to more informed evaluations and better performance.
    4. Backpropagation: Updates the statistics of the nodes along the path from the new node back to the root based on the simulation results.
    Inputs:
    - game_state (dict): The current game state provided by the Battlesnake engine.
    Returns:
    - dict: The chosen move direction based on the MCTS algorithm.
    """
    # Pre-calculate sets for collision detection
    game_state['_food_set'] = set((f['x'], f['y']) for f in game_state['board']['food'])
    game_state['_hazard_set'] = set((h['x'], h['y']) for h in game_state['board']['hazards'])
    game_state['_obstacle_set'] = set((p['x'], p['y']) for s in game_state['board']['snakes'] for p in s['body'])

    root = MCTSNode(game_state)
    start_time = time.time()
    
    # Heuristic Ranking for Root Expansion (Progressive History)
    if rave:
        # Sort root's untried moves using the heuristic before starting
        scored_moves = []
        for m in root.untried_moves:
            ns = get_next_state(game_state, m)
            score = reward(ns, 0, max_rollout, len(game_state['you']['body'])) if ns else -1
            scored_moves.append((m, score))
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        root.untried_moves = [m for m, s in scored_moves]

    # Time Budget: 0.1 for testing or 0.8 for production
    while time.time() - start_time < 0.1:
        node = root
        search_path = [node]
        
        # 1. Selection
        while node.children and not node.untried_moves:
            node = node.uct_select(exploration_constant=exploration_constant, 
                                   pb_weight=pb_weight, rave=rave)
            search_path.append(node)
            if node.state is None: break

        # 2. Expansion
        if node.untried_moves and node.state is not None:
            # If RAVE is on, we take the best heuristic move (front of list)
            move = node.untried_moves.pop(0 if rave else -1)
            next_state = get_next_state(node.state, move)
            new_node = MCTSNode(next_state, parent=node, move=move)
            
            if next_state is not None:
                new_node.heuristic_score = reward(next_state, 0, max_rollout, len(game_state['you']['body']))
            
            node.children.append(new_node)
            node = new_node
            search_path.append(node)

        # 3. Simulation
        # For RAVE, we need the sequence of moves made in the rollout
        if rave:
            score, rollout_moves = rollout_with_history(node.state, max_rollout, heuristic, competitive)
        else:
            if heuristic:
                score = rollout_heuristic(node.state, competitive=competitive)
            else:
                score = rollout(node.state, max_rollout=max_rollout)
            rollout_moves = []

        # 4. Backpropagation
        rollout_move_set = set(rollout_moves)
        for ancestor in reversed(search_path):
            ancestor.update(score)
            # RAVE Update: Update AMAF stats for every move seen in the rollout
            if rave:
                for m in rollout_move_set:
                    ancestor.rave_visits[m] += 1
                    ancestor.rave_wins[m] += score

    # Final Decision
    valid_children = [c for c in root.children if c.state is not None]
    if not valid_children:
        for m in ["up", "down", "left", "right"]:
            if get_next_state(game_state, m) is not None: return {"move": m}
        return {"move": "up"}
        
    best_child = max(valid_children, key=lambda c: c.visits)
    return {"move": best_child.move}

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Team 13",
        "color": getattr(info, "snake_color", "#472EA1"),
        "name": getattr(info, "snake_name", "MCTS-Snake"),
        "head": "shades",
        "tail": "bolt",
    }



   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", type=str, default="MCTS-Snake")
    parser.add_argument("--color", type=str, default="#695DD1")
    args = parser.parse_args()
    info.snake_color = args.color
    info.snake_name = args.name
    from server import run_server

    run_server({
        "info": info, 
        "start": start, 
        "move": lambda gs: mcts_agent(gs, heuristic=False, competitive=False), 
        "end": end
    }, port=args.port)