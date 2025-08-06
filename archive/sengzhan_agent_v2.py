import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class GomokuAgent(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Create an OpenAI-compatible client using the Gemma2 model for move generation
        self.llm = OpenAIGomokuClient(model="gemma-2-9b-it")

    async def get_move(self, game_state):
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to a human-readable string format
        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        # Prepare the conversation messages for the language model
        messages = """
You are an expert Gomoku (Five-in-a-Row) player. Your goal is to get 5 of your pieces
in a row (horizontally, vertically, or diagonally) while preventing your opponent from doing the same.

Strategic priorities:
1. WIN: If you can make 5 in a row, do it immediately. 
2. BLOCK: If opponent can make 5 in a row, block them immediately
3. CREATE THREATS: Build sequences of 2-3 pieces to create multiple winning opportunities
4. CONTROL CENTER: The center area is most valuable for creating multiple directions
5. CREATE FORKS: Try to create situations where you have multiple ways to win

Plan to win: Prioritize playing near the center of the board (rows/cols 7–11). If you can make 4 in a row with an open end, play that move. Try to make 3 in a row with both ends open (Live Three).

Important: If the opponent has 4 in a row with an open end, block it. 

You must respond with valid JSON in this exact format:
{
    "reasoning": "Brief explanation of your strategic thinking",
    "row": <row_number>,
    "col": <col_number>
}

The row and col must be valid coordinates (0-indexed). Choose only empty positions marked with '.'.
"""

        # Send the messages to the language model and get the response
        content = await self.llm.complete(messages)

        # Parse the LLM response to extract move coordinates
        try:
            # Use regex to find JSON-like content in the response
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                # Parse the JSON to extract row and column
                move = json.loads(m.group(0))
                row, col = (move["row"], move["col"])

                # Validate that the proposed move is legal
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, continue to fallback strategy
            pass

        # Fallback: if LLM response is invalid, choose the first available legal move
        return game_state.get_legal_moves()[0]
