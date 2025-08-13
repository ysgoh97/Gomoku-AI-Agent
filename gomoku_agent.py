import re
import json
from typing import Tuple, List, Dict, Optional
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient

class GoGomoku(Agent):

    # Initialize agent
    # def __init__(self, agent_id: str):
    #     super().__init__(agent_id)
    #     print(f"🎓 Created: {agent_id}")

    # Setup agent
    def _setup(self):
        print("⚙️  Setting up LLM agent...")
        self.llm_client = OpenAIGomokuClient(model="gemma-2-9b-it")
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
            'to_defuse': []
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

                    # Get list of moves to defuse (prevent opponent from winning in the next 2 turns)
                    if (self._check_lines(game_state, row, col, opponent, 4) and
                        self._check_open(game_state, row, col)):
                        analysis['to_defuse'].append((row, col))

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

    # Sort moves by distance to center
    def _sort_center(self, move_list: List, game_state: GameState):
        n = game_state.board_size
        center = n // 2
        if move_list:
            cx, cy = center, center
            def dist2(move):
                r, c = move
                dr, dc = r - cx, c - cy
                return dr * dr + dc * dc
            move_list.sort(key=dist2)
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
                analysis['to_defuse'] = self._sort_center(analysis['to_defuse'], game_state)
                return analysis['to_defuse'][0]

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

        # Try center first
        n = game_state.board_size
        center = n // 2
        if game_state.is_valid_move(center, center):
            return (center, center)

        # Otherwise, try legal moves close to center
        legal_moves = game_state.get_legal_moves()
        legal_moves = self._sort_center(legal_moves, game_state)
        return legal_moves[0]
    