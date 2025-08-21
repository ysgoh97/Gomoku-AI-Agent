import os
import re
import json
from typing import Tuple, List, Dict
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient

class YSV7(Agent):

    # # Initialize agent
    # def __init__(self, agent_id: str):
    #     super().__init__(agent_id)
    #     print(f"🎓 Created: {agent_id}")

    # Setup agent
    def _setup(self):
        print("⚙️  Setting up LLM agent...")
        self.llm_client = OpenAIGomokuClient(
            model="gemma2-9b-it",
            api_key=os.environ["OPENAI_API_KEY"],
            endpoint=os.environ["OPENAI_BASE_URL"]
        )
        self.move_history = []
        self.invalid_moves = 0
        print("✅ Agent setup complete!")

    # Get winning moves, and oppoenent's winning moves and threats
    def _get_critical_moves(self, game_state: GameState) -> Dict:
        board_size = game_state.board_size
        player = self.player.value
        opponent = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        analysis = {
            'to_win': [],
            'to_defend': [],
            'to_defuse': [],
            'to_attack': [],
            'to_fork': []
        }

        for row in range(board_size):
            for col in range(board_size):
                if game_state.board[row][col] == '.':

                    # Get list of moves to win
                    if self._check_lines(game_state, row, col, player, 5):
                        analysis['to_win'].append((row, col))

                    # Get list of moves to defend (prevent opponent from winning in the next turn)
                    if self._check_lines(game_state, row, col, opponent, 5):
                        analysis['to_defend'].append((row, col))

                    # Get list of moves to defend (prevent opponent from winning in the next 2 turns)
                    if self._check_threats(game_state, row, col, opponent):
                        analysis['to_defuse'].append((row, col))

                    # Get list of moves to win in the next turn
                    if self._check_threats(game_state, row, col, opponent):
                        analysis['to_attack'].append((row, col))

                    # Get list of moves to fork (create two diagonal adjacencies)
                    if self._check_fork_opportunity(game_state, row, col, player):
                        analysis['to_fork'].append((row, col))

        return analysis

    # Check consecutive pieces
    def _check_lines(self, game_state: GameState, row: int, col: int, player: str, max_count: int) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1
            open = 0

            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < board_size and
                   0 <= c < board_size and
                   game_state.board[r][c] == player):
                count += 1
                r, c = r + dr, c + dc

            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < board_size and
                   0 <= c < board_size and
                   game_state.board[r][c] == player):
                count += 1
                r, c = r - dr, c - dc

            if count >= max_count:
                return True
        return False

    # Check open ends
    def _check_open(self, game_state: GameState, row: int, col: int) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            open = 0

            # Check in positive direction
            r, c = row + dr, col + dc
            if (0 <= r < board_size and
                0 <= c < board_size and
                game_state.board[r][c] == "."):
                open += 1

            # Count in negative direction
            r, c = row - dr, col - dc
            if (0 <= r < board_size and
                0 <= c < board_size and
                game_state.board[r][c] == "."):
                open += 1

            if open == 2:
                return True
        return False
    
    # Check for specific defuse patterns: ._xxx., .xxx_., .xx_x., .x_xx.
    def _check_threats(self, game_state: GameState, row: int, col: int, opponent: str) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for dr, dc in directions:
            # Check all 4 patterns in this direction
            patterns_to_check = [
                # Pattern ._xxx. where _ is at position 1 (0-indexed)
                ['.', '_', opponent, opponent, opponent, '.'],
                # Pattern .xxx_. where _ is at position 4 (0-indexed) 
                ['.', opponent, opponent, opponent, '_', '.'],
                # Pattern .xx_x. where _ is at position 3 (0-indexed)
                ['.', opponent, opponent, '_', opponent, '.'],
                # Pattern .x_xx. where _ is at position 2 (0-indexed)
                ['.', opponent, '_', opponent, opponent, '.']
            ]
            
            for pattern in patterns_to_check:
                d_position = pattern.index('_')
                
                # Check if current position matches
                start_row = row - d_position * dr
                start_col = col - d_position * dc
                
                # Verify the entire pattern matches
                if self._matches_pattern(game_state, start_row, start_col, dr, dc, pattern, row, col):
                    return True         
        return False

    # Helper method to check if a pattern matches at a given position
    def _matches_pattern(self, game_state: GameState, start_row: int, start_col: int, 
                        dr: int, dc: int, pattern: List[str], target_row: int, target_col: int) -> bool:
        board_size = game_state.board_size
        
        for i, expected in enumerate(pattern):
            curr_row = start_row + i * dr
            curr_col = start_col + i * dc
            
            # Check bounds
            if not (0 <= curr_row < board_size and 0 <= curr_col < board_size):
                return False
            
            if expected == '_':
                # This should be our target position (empty)
                if curr_row != target_row or curr_col != target_col:
                    return False
                if game_state.board[curr_row][curr_col] != '.':
                    return False
            else:
                # Check if the actual board position matches expected
                actual = game_state.board[curr_row][curr_col]
                if actual != expected:
                    return False
        return True
    
    # Check for fork opportunities - moves that create intersecting lines
    def _check_fork_opportunity(self, game_state: GameState, row: int, col: int, player: str) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        lines_count = 0
        
        # Check each direction
        for dr, dc in directions:
            line_length = 1
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < board_size and 
                   0 <= c < board_size and 
                   game_state.board[r][c] == player):
                line_length += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < board_size and 
                   0 <= c < board_size and 
                   game_state.board[r][c] == player):
                line_length += 1
                r, c = r - dr, c - dc
            
            # If this direction would create a line of 2 or more pieces, count it
            if line_length >= 2:
                lines_count += 1
        
        # A fork exists if placing here creates 2 or more intersecting lines
        return lines_count >= 2
    
    # Count adjacent own pieces for a given position
    def _count_adjacent_pieces(self, game_state: GameState, row: int, col: int, player: str) -> int:
        board_size = game_state.board_size
        count = 0
        # Check all 8 adjacent positions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if (0 <= r < board_size and 
                    0 <= c < board_size and 
                    game_state.board[r][c] == player):
                    count += 1
        return count

    # Sort moves by number of adjacent own pieces, then by distance to center
    def _sort_moves(self, move_list: List, game_state: GameState):
        n = game_state.board_size
        center = n // 2
        player = self.player.value
        
        if move_list:
            cx, cy = center, center
            def sort_key(move):
                r, c = move
                # Primary sort: number of adjacent own pieces (descending)
                adjacent_count = self._count_adjacent_pieces(game_state, r, c, player)
                # Secondary sort: distance to center (ascending)
                dr, dc = r - cx, c - cy
                center_dist = dr * dr + dc * dc
                # Return tuple: (-adjacent_count, center_dist)
                # Negative adjacent_count for descending order
                return (-adjacent_count, center_dist)
            move_list.sort(key=sort_key)
        return move_list

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        print(f"\n🧠 {self.agent_id} is thinking...")

        try:
            board_size = game_state.board_size
            player = self.player.value
            opponent = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

            # If critical moves exist, perform those first
            analysis = self._get_critical_moves(game_state)
            if analysis['to_win']:
                print(f"🏆 Win at: {analysis['to_win']}")
                return analysis['to_win'][0]
            elif analysis['to_defend']:
                print(f"🛡️ Defend at: {analysis['to_defend']}")
                return analysis['to_defend'][0]
            elif analysis['to_defuse']:
                print(f"💣 Defuse at: {analysis['to_defuse']}")
                analysis['to_defuse'] = self._sort_moves(analysis['to_defuse'], game_state)
                return analysis['to_defuse'][0]
            elif analysis['to_attack']:
                print(f"🗡️ Attack at: {analysis['to_attack']}")
                analysis['to_attack'] = self._sort_moves(analysis['to_attack'], game_state)
                return analysis['to_attack'][0]
            elif analysis['to_fork']:
                print(f"⚔️ Fork at: {analysis['to_fork']}")
                analysis['to_fork'] = self._sort_moves(analysis['to_fork'], game_state)
                # Let LLM decide where to fork

            # Otherwise, use LLM to strategize
            system_prompt = f"""
### Instruction:
You are an expert Gomoku player.\
You will be playing on a {board_size}x{board_size} board where rows and columns are indexed 0 to 7.\
Your pieces will be marked by '{player}', and your opponent's pieces will be marked by '{opponent}'.\
Your goal is to be the first to place 5 consecutive '{player}' horiontally, vertically, or diagionally.

### Strategy:
Before making any move, you must first analyze the board and apply these strategies.
- Build intersecting forks with open ends from multiple directions.
- Make use of diagonal attacks.
- Take control of the center of the board.
- Predict your opponent's moves, intercept, and surround them.

### Response:
Respond with a valid JSON using a ```json wrapper.\
Do not output any intermediate thinking, explanation, or additional remark.

```json
{{
    "analysis": "<brief analysis of the board state>",
    "strategy": "<which strategy you're applying>",
    "move": {{"row": <row_number>, "col": <col_number>}}
}}
```
""".strip()

            board_str = game_state.format_board(formatter="standard")
            unoccupied_pos = []
            for row in range(game_state.board_size):
                for col in range(game_state.board_size):
                    if game_state.board[row][col] not in [player, opponent]:
                        unoccupied_pos.append((row, col))

            board_prompt = f"Current board state:\n{board_str}\n"
            board_prompt += f"You are playing as: {player}\n"
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                board_prompt += f"Your last move was: ({last_move.row}, {last_move.col})\n"
            board_prompt += f"You can make moves at: {', '.join(f'({r},{c})' for r, c in unoccupied_pos)}\n"
            if analysis["to_fork"]:
                board_prompt += f"Consider fork opportunities at: {analysis['to_fork']}\n"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{board_prompt}Best move in JSON: "},
            ]

            print("💡 Full Prompt:\n\n")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print()

            response = await self.llm_client.complete(messages)

            print("💡 Response:\n\n")
            print(response)
            print()

            move = self._parse_move_response(response, game_state, analysis)
            return move

        # Use fallback if there are errors
        except Exception as e:
            print(f"🚫 LLM error for agent {self.agent_id}: {e}")
            self.invalid_moves += 1
            return self._get_fallback_move(game_state)

    # Parse LLM response
    def _parse_move_response(self, response: str, game_state: GameState, analysis: Dict) -> Tuple[int, int]:
        try:
            json_match = re.search(r"```json([^`]+)```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)

                if "move" in data:
                    move = data["move"]
                    row, col = move["row"], move["col"]

                    if game_state.is_valid_move(row, col):
                        return (row, col)

                    # Use fallback if move is invalid
                    else:
                        print(f"⚠️ Invalid move by {self.agent_id}: ({row}, {col})")
                        self.invalid_moves += 1
                        return self._get_fallback_move(game_state)

        # Use fallback if there are parsing errors
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            return self._get_fallback_move(game_state)

    # Fallback moves
    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:

        # Try center first if board is empty or nearly empty
        n = game_state.board_size
        center = n // 2
        if game_state.is_valid_move(center, center):
            # Count total pieces on board
            total_pieces = sum(1 for row in range(n) for col in range(n) 
                             if game_state.board[row][col] != '.')
            # Use center if very few pieces on board
            if total_pieces <= 2:
                return (center, center)

        # Otherwise, get all legal moves and sort by adjacency + center distance
        legal_moves = game_state.get_legal_moves()
        legal_moves = self._sort_moves(legal_moves, game_state)
        return legal_moves[0]
    import re
import json
from typing import Tuple, List, Dict
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient

class GoGomoku(Agent):

    # # Initialize agent
    # def __init__(self, agent_id: str):
    #     super().__init__(agent_id)
    #     print(f"🎓 Created: {agent_id}")

    # Setup agent
    def _setup(self):
        print("⚙️  Setting up LLM agent...")
        self.llm_client = OpenAIGomokuClient(model="glm-4-9b-0414")
        self.move_history = []
        self.invalid_moves = 0
        print("✅ Agent setup complete!")

    # Get winning moves, and oppoenent's winning moves and threats
    def _get_critical_moves(self, game_state: GameState) -> Dict:
        board_size = game_state.board_size
        player = self.player.value
        opponent = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        analysis = {
            'to_win': [],
            'to_defend': [],
            'to_defuse': [],
            'to_attack': [],
            'to_fork': []
        }

        for row in range(board_size):
            for col in range(board_size):
                if game_state.board[row][col] == '.':

                    # Get list of moves to win
                    if self._check_lines(game_state, row, col, player, 5):
                        analysis['to_win'].append((row, col))

                    # Get list of moves to defend (prevent opponent from winning in the next turn)
                    if self._check_lines(game_state, row, col, opponent, 5):
                        analysis['to_defend'].append((row, col))

                    # Get list of moves to defend (prevent opponent from winning in the next 2 turns)
                    if self._check_threats(game_state, row, col, opponent):
                        analysis['to_defuse'].append((row, col))

                    # Get list of moves to win in the next turn
                    if self._check_threats(game_state, row, col, opponent):
                        analysis['to_attack'].append((row, col))

                    # Get list of moves to fork (create two diagonal adjacencies)
                    if self._check_fork_opportunity(game_state, row, col, player):
                        analysis['to_fork'].append((row, col))

        return analysis

    # Check consecutive pieces
    def _check_lines(self, game_state: GameState, row: int, col: int, player: str, max_count: int) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1
            open = 0

            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < board_size and
                   0 <= c < board_size and
                   game_state.board[r][c] == player):
                count += 1
                r, c = r + dr, c + dc

            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < board_size and
                   0 <= c < board_size and
                   game_state.board[r][c] == player):
                count += 1
                r, c = r - dr, c - dc

            if count >= max_count:
                return True
        return False

    # Check open ends
    def _check_open(self, game_state: GameState, row: int, col: int) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            open = 0

            # Check in positive direction
            r, c = row + dr, col + dc
            if (0 <= r < board_size and
                0 <= c < board_size and
                game_state.board[r][c] == "."):
                open += 1

            # Count in negative direction
            r, c = row - dr, col - dc
            if (0 <= r < board_size and
                0 <= c < board_size and
                game_state.board[r][c] == "."):
                open += 1

            if open == 2:
                return True
        return False
    
    # Check for specific defuse patterns: ._xxx., .xxx_., .xx_x., .x_xx.
    def _check_threats(self, game_state: GameState, row: int, col: int, opponent: str) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for dr, dc in directions:
            # Check all 4 patterns in this direction
            patterns_to_check = [
                # Pattern ._xxx. where _ is at position 1 (0-indexed)
                ['.', '_', opponent, opponent, opponent, '.'],
                # Pattern .xxx_. where _ is at position 4 (0-indexed) 
                ['.', opponent, opponent, opponent, '_', '.'],
                # Pattern .xx_x. where _ is at position 3 (0-indexed)
                ['.', opponent, opponent, '_', opponent, '.'],
                # Pattern .x_xx. where _ is at position 2 (0-indexed)
                ['.', opponent, '_', opponent, opponent, '.']
            ]
            
            for pattern in patterns_to_check:
                d_position = pattern.index('_')
                
                # Check if current position matches
                start_row = row - d_position * dr
                start_col = col - d_position * dc
                
                # Verify the entire pattern matches
                if self._matches_pattern(game_state, start_row, start_col, dr, dc, pattern, row, col):
                    return True         
        return False

    # Helper method to check if a pattern matches at a given position
    def _matches_pattern(self, game_state: GameState, start_row: int, start_col: int, 
                        dr: int, dc: int, pattern: List[str], target_row: int, target_col: int) -> bool:
        board_size = game_state.board_size
        
        for i, expected in enumerate(pattern):
            curr_row = start_row + i * dr
            curr_col = start_col + i * dc
            
            # Check bounds
            if not (0 <= curr_row < board_size and 0 <= curr_col < board_size):
                return False
            
            if expected == '_':
                # This should be our target position (empty)
                if curr_row != target_row or curr_col != target_col:
                    return False
                if game_state.board[curr_row][curr_col] != '.':
                    return False
            else:
                # Check if the actual board position matches expected
                actual = game_state.board[curr_row][curr_col]
                if actual != expected:
                    return False
        return True
    
    # Check for fork opportunities - moves that create intersecting lines
    def _check_fork_opportunity(self, game_state: GameState, row: int, col: int, player: str) -> bool:
        board_size = game_state.board_size
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        lines_count = 0
        
        # Check each direction
        for dr, dc in directions:
            line_length = 1
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < board_size and 
                   0 <= c < board_size and 
                   game_state.board[r][c] == player):
                line_length += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < board_size and 
                   0 <= c < board_size and 
                   game_state.board[r][c] == player):
                line_length += 1
                r, c = r - dr, c - dc
            
            # If this direction would create a line of 2 or more pieces, count it
            if line_length >= 2:
                lines_count += 1
        
        # A fork exists if placing here creates 2 or more intersecting lines
        return lines_count >= 2
    
    # Count adjacent own pieces for a given position
    def _count_adjacent_pieces(self, game_state: GameState, row: int, col: int, player: str) -> int:
        board_size = game_state.board_size
        count = 0
        # Check all 8 adjacent positions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if (0 <= r < board_size and 
                    0 <= c < board_size and 
                    game_state.board[r][c] == player):
                    count += 1
        return count

    # Sort moves by number of adjacent own pieces, then by distance to center
    def _sort_moves(self, move_list: List, game_state: GameState):
        n = game_state.board_size
        center = n // 2
        player = self.player.value
        
        if move_list:
            cx, cy = center, center
            def sort_key(move):
                r, c = move
                # Primary sort: number of adjacent own pieces (descending)
                adjacent_count = self._count_adjacent_pieces(game_state, r, c, player)
                # Secondary sort: distance to center (ascending)
                dr, dc = r - cx, c - cy
                center_dist = dr * dr + dc * dc
                # Return tuple: (-adjacent_count, center_dist)
                # Negative adjacent_count for descending order
                return (-adjacent_count, center_dist)
            move_list.sort(key=sort_key)
        return move_list

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        print(f"\n🧠 {self.agent_id} is thinking...")

        try:
            board_size = game_state.board_size
            player = self.player.value
            opponent = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

            # If critical moves exist, perform those first
            analysis = self._get_critical_moves(game_state)
            if analysis['to_win']:
                print(f"🏆 Win at: {analysis['to_win']}")
                return analysis['to_win'][0]
            elif analysis['to_defend']:
                print(f"🛡️ Defend at: {analysis['to_defend']}")
                return analysis['to_defend'][0]
            elif analysis['to_defuse']:
                print(f"💣 Defuse at: {analysis['to_defuse']}")
                analysis['to_defuse'] = self._sort_moves(analysis['to_defuse'], game_state)
                return analysis['to_defuse'][0]
            elif analysis['to_attack']:
                print(f"🗡️ Attack at: {analysis['to_attack']}")
                analysis['to_attack'] = self._sort_moves(analysis['to_attack'], game_state)
                return analysis['to_attack'][0]
            elif analysis['to_fork']:
                print(f"⚔️ Fork at: {analysis['to_fork']}")
                analysis['to_fork'] = self._sort_moves(analysis['to_fork'], game_state)
                # Let LLM decide where to fork

            # Otherwise, use LLM to strategize
            system_prompt = f"""
### Instruction:
You are an expert Gomoku player.\
You will be playing on a {board_size}x{board_size} board where rows and columns are indexed 0 to 7.\
Your pieces will be marked by '{player}', and your opponent's pieces will be marked by '{opponent}'.\
Your goal is to be the first to place 5 consecutive '{player}' horiontally, vertically, or diagionally.

### Strategy:
Before making any move, you must first analyze the board and apply these strategies.
- Build intersecting forks with open ends from multiple directions.
- Make use of diagonal attacks.
- Take control of the center of the board.
- Predict your opponent's moves, intercept, and surround them.

### Response:
Respond with a valid JSON using a ```json wrapper.\
Do not output any intermediate thinking, explanation, or additional remark.

```json
{{
    "analysis": "<brief analysis of the board state>",
    "strategy": "<which strategy you're applying>",
    "move": {{"row": <row_number>, "col": <col_number>}}
}}
```
""".strip()

            board_str = game_state.format_board(formatter="standard")
            unoccupied_pos = []
            for row in range(game_state.board_size):
                for col in range(game_state.board_size):
                    if game_state.board[row][col] not in [player, opponent]:
                        unoccupied_pos.append((row, col))

            board_prompt = f"Current board state:\n{board_str}\n"
            board_prompt += f"You are playing as: {player}\n"
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                board_prompt += f"Your last move was: ({last_move.row}, {last_move.col})\n"
            board_prompt += f"You can make moves at: {', '.join(f'({r},{c})' for r, c in unoccupied_pos)}\n"
            if analysis["to_fork"]:
                board_prompt += f"Consider fork opportunities at: {analysis['to_fork']}\n"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{board_prompt}Best move in JSON: "},
            ]

            print("💡 Full Prompt:\n\n")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print()

            response = await self.llm_client.complete(messages)

            print("💡 Response:\n\n")
            print(response)
            print()

            move = self._parse_move_response(response, game_state, analysis)
            return move

        # Use fallback if there are errors
        except Exception as e:
            print(f"🚫 LLM error for agent {self.agent_id}: {e}")
            self.invalid_moves += 1
            return self._get_fallback_move(game_state)

    # Parse LLM response
    def _parse_move_response(self, response: str, game_state: GameState, analysis: Dict) -> Tuple[int, int]:
        try:
            json_match = re.search(r"```json([^`]+)```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)

                if "move" in data:
                    move = data["move"]
                    row, col = move["row"], move["col"]

                    if game_state.is_valid_move(row, col):
                        return (row, col)

                    # Use fallback if move is invalid
                    else:
                        print(f"⚠️ Invalid move by {self.agent_id}: ({row}, {col})")
                        self.invalid_moves += 1
                        return self._get_fallback_move(game_state)

        # Use fallback if there are parsing errors
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            return self._get_fallback_move(game_state)

    # Fallback moves
    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:

        # Try center first if board is empty or nearly empty
        n = game_state.board_size
        center = n // 2
        if game_state.is_valid_move(center, center):
            # Count total pieces on board
            total_pieces = sum(1 for row in range(n) for col in range(n) 
                             if game_state.board[row][col] != '.')
            # Use center if very few pieces on board
            if total_pieces <= 2:
                return (center, center)

        # Otherwise, get all legal moves and sort by adjacency + center distance
        legal_moves = game_state.get_legal_moves()
        legal_moves = self._sort_moves(legal_moves, game_state)
        return legal_moves[0]
    