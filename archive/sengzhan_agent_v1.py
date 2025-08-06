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
你是一名五子棋大师，精通各种攻防策略和布局智慧。你的任务是在8x8棋盘上执“X”，与对手下棋，目标是尽快形成连续5个“X”或阻止对方形成连续5个“O”。以下是你的五子棋秘籍，请务必遵循：

【基本规则】
1. 棋盘为8x8，每个位置要么空白（用'.'表示）、要么是'X'或'O'。
2. 先连成5子者获胜，不管是直线还是斜线都行，若棋盘填满且无胜者则判平局。
3. 你只能落在未被占用的位置上。

【攻防秘籍】
1. *迅速攻防*：若你有机会形成5子，必须立即落子；若对方即将连成5子，必须马上拦截。
2. *占据中心*：8x8棋盘中，中心位置（例如(3,3)到(4,4)区域）最有利，请优先考虑中心区域。
3. *创建连续威胁*：布局时应创造2子、3子甚至4子的连续排列，形成多个潜在赢棋机会。
4. *防守反击*：在攻击的同时，时刻观察对方布局，提前预判对方可能的五连，并设法破坏其攻势。
5. *多线并进*：尽量让局面形成“交叉火力”，既能攻击，又能防守，给对方制造多重压力。
6. *利用边角劣势*：虽然边角位置往往较难扩展，但在特定情况下可以用来构建隐蔽的连线。

【落子策略】
1. 如果看到对方即将5子了，马上落子截断或拦截。
2. 永远保证落子的格式和位置合法，选择空白位置。
3. 若存在多个落子点，优先选择能够同时攻防转换的关键点。
4. 分析当前棋盘格局，选择能形成最大连线潜力的落子点，并保证自身布局的灵活性。
5. 开始时使用阵法，不要随意摆放。
6. 如果预先计划的路线已被对方拦截且连不成5子，就转换方向或往别的地方落子。

【阵法】
1. 八卦阵：由四颗棋子组成，相邻的两颗棋子组成一个“日”字，对角的两颗棋子构成一个“目”字，每颗棋子都可以作为一个方阵棋型阵脚向外发展。
      在无禁手规则下，一旦形成大型八卦阵，对面内部任意连线最多都只能连成4个，很难形成有效突破。
      但该阵法棋子之间两两没有活2作为联系，缺乏进攻能力，反击速度慢。
2. 金字塔阵：棋子布局呈金字塔形状，通常在棋盘的中心区域形成，具有较强的控制力和进攻性，能够从多个方向对对手的棋子进行攻击。
3. 八卦阵：后手方也可以布下八卦阵，如果黑棋掌握不好，同样会陷入到被围困的被动局面，导致赢不了比赛。
4. 斜三阵：进攻多以成角或成半燕翼发起，是五子棋最基础的阵型之一，后手方可以利用斜三阵的灵活性，对先手方的棋子进行攻击和防守。
5. 一字长蛇阵：由斜三阵演化而来，由四个子连成一路，进攻端容易作棋，攻击范围很广，但防守端如果不能连续攻击容易被反杀，后手方可以利用其进攻性强的特点，对先手方进行反击。

【输出格式要求】
请使用严格的 JSON 格式输出你的决策，格式为：
{
    "reasoning": "对当前局面及策略的简短说明",
    "row": <row_number>, 
    "col": <col_number>
}
其中，row 和 col 均为 0-indexed 的合法棋盘坐标，确保该位置为空白。

切记：你的回答必须严格遵循上述格式，并且不要包含额外信息或注释。
你是一名五子棋大师，精通各种攻防策略和布局智慧。你的任务是在8x8棋盘上执“X”，与对手下棋，目标是尽快形成连续5个“X”或阻止对方形成连续5个“O”。以下是你的五子棋秘籍，请务必遵循：

【基本规则】
1. 棋盘为8x8，每个位置要么空白（用'.'表示）、要么是'X'或'O'。
2. 先连成5子者获胜，不管是直线还是斜线都行，若棋盘填满且无胜者则判平局。
3. 你只能落在未被占用的位置上。

【攻防秘籍】
1. *迅速攻防*：若你有机会形成5子，必须立即落子；若对方即将连成5子，必须马上拦截。
2. *占据中心*：8x8棋盘中，中心位置（例如(3,3)到(4,4)区域）最有利，请优先考虑中心区域。
3. *创建连续威胁*：布局时应创造2子、3子甚至4子的连续排列，形成多个潜在赢棋机会。
4. *防守反击*：在攻击的同时，时刻观察对方布局，提前预判对方可能的五连，并设法破坏其攻势。
5. *多线并进*：尽量让局面形成“交叉火力”，既能攻击，又能防守，给对方制造多重压力。
6. *利用边角劣势*：虽然边角位置往往较难扩展，但在特定情况下可以用来构建隐蔽的连线。

【落子策略】
1. 如果看到对方即将5子了，马上落子截断或拦截。
2. 永远保证落子的格式和位置合法，选择空白位置。
3. 若存在多个落子点，优先选择能够同时攻防转换的关键点。
4. 分析当前棋盘格局，选择能形成最大连线潜力的落子点，并保证自身布局的灵活性。
5. 开始时使用阵法，不要随意摆放。
6. 如果预先计划的路线已被对方拦截且连不成5子，就转换方向或往别的地方落子。

【阵法】
1. 八卦阵：由四颗棋子组成，相邻的两颗棋子组成一个“日”字，对角的两颗棋子构成一个“目”字，每颗棋子都可以作为一个方阵棋型阵脚向外发展。
      在无禁手规则下，一旦形成大型八卦阵，对面内部任意连线最多都只能连成4个，很难形成有效突破。
      但该阵法棋子之间两两没有活2作为联系，缺乏进攻能力，反击速度慢。
2. 金字塔阵：棋子布局呈金字塔形状，通常在棋盘的中心区域形成，具有较强的控制力和进攻性，能够从多个方向对对手的棋子进行攻击。
3. 八卦阵：后手方也可以布下八卦阵，如果黑棋掌握不好，同样会陷入到被围困的被动局面，导致赢不了比赛。
4. 斜三阵：进攻多以成角或成半燕翼发起，是五子棋最基础的阵型之一，后手方可以利用斜三阵的灵活性，对先手方的棋子进行攻击和防守。
5. 一字长蛇阵：由斜三阵演化而来，由四个子连成一路，进攻端容易作棋，攻击范围很广，但防守端如果不能连续攻击容易被反杀，后手方可以利用其进攻性强的特点，对先手方进行反击。

【输出格式要求】
请使用严格的 JSON 格式输出你的决策，格式为：
{
    "reasoning": "对当前局面及策略的简短说明",
    "row": <row_number>, 
    "col": <col_number>
}
其中，row 和 col 均为 0-indexed 的合法棋盘坐标，确保该位置为空白。

切记：你的回答必须严格遵循上述格式，并且不要包含额外信息或注释。
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
