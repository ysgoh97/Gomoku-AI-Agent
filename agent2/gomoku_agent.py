import os
import re
import json
from gomoku.agents.base import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import GameState, Player
from typing import Tuple, Optional, List
import random

class SZT4(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.llm_client = None
        self.system_prompt = ""
        self.invalid_moves = 0

        # ===== 阵法状态 =====
        self.formation_active: bool = True            # 是否启用阵法
        self.formation_name: str = "diamond_then_plus"
        self.formation_plan_abs: Optional[List[Tuple[int, int]]] = None  # 绝对坐标序列
        self.formation_progress_idx: int = 0          # 已完成到第几个点
        self.formation_anchor: Optional[Tuple[int,int]] = None           # 阵法锚点（通常是中心或其邻近）
        self.formation_max_plies: int = 12            # 前期使用阵法（总回合数阈值，可调）

        try:
            self._setup()
        except Exception as e:
            print(f"_setup skipped: {e}")

    def _setup(self):
        self.invalid_moves = 0
        try:
            if OpenAIGomokuClient is not None:
                self.llm_client = OpenAIGomokuClient(
                    model="gemma2-9b-it",
                    api_key=os.environ["OPENAI_API_KEY"],
                    endpoint=os.environ["OPENAI_BASE_URL"]
                )
        except Exception as e:
            print(f"LLM client not available: {e}")
            self.llm_client = None
        self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        return (
            "You are an expert Gomoku (Five-in-a-Row) player. The board is 0-indexed.\n"
            "Goal: make 5 in a row and prevent opponent from doing so.\n\n"
            "ABSOLUTE PRIORITIES (apply in EXACT order each move):\n"
            "1) WIN: Complete 5-in-a-row immediately.\n"
            "2) BLOCK: If the opponent can complete 5-in-a-row next, block immediately.\n"
            "3) BLOCK EXISTING OPEN THREE: If opponent ALREADY has an open-three on board, block it.\n"
            "4) CREATE OPEN THREE: Make your own open-three.\n"
            "5) CREATE DOUBLE THREATS.\n"
            "6) IMPROVE POSITION: Extend your lines and control the center.\n\n"
            "HARD CONSTRAINTS:\n"
            "- Only place on '.'\n"
            "- Coordinates within [0..N-1]\n"
            "- Respond ONLY with a JSON object in ```json ... ```\n\n"
            "OUTPUT FORMAT:\n"
            "```json\n"
            "{\n"
            "  \"reasoning\": \"1–2 short sentences stating which rule you applied\",\n"
            "  \"move\": {\"row\": <int>, \"col\": <int>}\n"
            "}\n"
            "```"
        )

    # ======== 工具函数 ========
    @staticmethod
    def _inb(n: int, r: int, c: int) -> bool:
        return 0 <= r < n and 0 <= c < n

    @staticmethod
    def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # ===== 基础判断 =====
    def _has_five_if_place(self, board, r: int, c: int, ch: str) -> bool:
        n = len(board)
        original = board[r][c]
        board[r][c] = ch
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def count_line(rr, cc, dr, dc):
            cnt = 0
            i, j = rr, cc
            while 0 <= i < n and 0 <= j < n and board[i][j] == ch:
                cnt += 1
                i += dr
                j += dc
            return cnt

        for dr, dc in dirs:
            i, j = r, c
            while 0 <= i - dr < n and 0 <= j - dc < n and board[i - dr][j - dc] == ch:
                i -= dr
                j -= dc
            total = count_line(i, j, dr, dc)
            if total >= 5:
                board[r][c] = original
                return True

        board[r][c] = original
        return False

    def _is_open_three_if_place(self, board, r: int, c: int, ch: str) -> bool:
        """
        活三：三连，两端是空格，或跳三（中间一个空）
        仅用于“我方造活三”的评估；不会用于对手的预判式拦截。
        """
        n = len(board)
        original = board[r][c]
        board[r][c] = ch
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        found_open_three = False

        for dr, dc in dirs:
            line = []
            for step in range(-4, 5):
                rr, cc = r + dr * step, c + dc * step
                if 0 <= rr < n and 0 <= cc < n:
                    line.append(board[rr][cc])
                else:
                    line.append('#')
            for i in range(len(line) - 4):
                window = "".join(line[i:i+5])
                if window == f".{ch}{ch}{ch}.":  # .XXX.
                    found_open_three = True
                if window == f".{ch}{ch}.{ch}." or window == f".{ch}.{ch}{ch}.":  # .XX.X. / .X.XX.
                    found_open_three = True
                if found_open_three:
                    break
            if found_open_three:
                break

        board[r][c] = original
        return found_open_three

    # ===== 查找落子点（强优先） =====
    def _find_immediate_winning_move(self, game_state: GameState, player_char: str) -> Optional[Tuple[int, int]]:
        board = game_state.board
        for r, c in game_state.get_legal_moves():
            if self._has_five_if_place(board, r, c, player_char):
                return (r, c)
        return None

    def _find_open_three_move(self, game_state: GameState, player_char: str) -> Optional[Tuple[int, int]]:
        board = game_state.board
        for r, c in game_state.get_legal_moves():
            if self._is_open_three_if_place(board, r, c, player_char):
                return (r, c)
        return None

    def _find_block_for_existing_open_three(self, game_state: GameState, rival: str) -> Optional[Tuple[int, int]]:
        """
        只拦截“当前棋面已经存在的对手活三”。
        形态：
          A) . r r r .
          B) . r r . r  （跳三）
          C) . r . r r  （跳三）
        返回一个能打断它的落点；若当前没有现成活三则返回 None。
        """
        board = game_state.board
        n = len(board)
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def inb(i, j): return 0 <= i < n and 0 <= j < n

        candidate_blocks = set()

        for r in range(n):
            for c in range(n):
                for dr, dc in dirs:
                    line_cells = []
                    for step in range(-4, 5):
                        rr, cc = r + dr * step, c + dc * step
                        if inb(rr, cc):
                            line_cells.append((board[rr][cc], (rr, cc)))
                        else:
                            line_cells.append(('#', (rr, cc)))

                    for i in range(len(line_cells) - 4):
                        window = line_cells[i:i+5]
                        s = "".join(ch for ch, _ in window)

                        # A) . r r r .
                        if s == f".{rival}{rival}{rival}.":
                            for k in (0, 4):
                                _, (rr, cc) = window[k]
                                if inb(rr, cc) and board[rr][cc] == '.' and game_state.is_valid_move(rr, cc):
                                    candidate_blocks.add((rr, cc))

                        # B) . r r . r
                        if s == f".{rival}{rival}.{rival}":
                            for k in (0, 3):
                                _, (rr, cc) = window[k]
                                if inb(rr, cc) and board[rr][cc] == '.' and game_state.is_valid_move(rr, cc):
                                    candidate_blocks.add((rr, cc))

                        # C) . r . r r
                        if s == f".{rival}.{rival}{rival}":
                            for k in (0, 2):
                                _, (rr, cc) = window[k]
                                if inb(rr, cc) and board[rr][cc] == '.' and game_state.is_valid_move(rr, cc):
                                    candidate_blocks.add((rr, cc))

        if not candidate_blocks:
            return None

        center = game_state.board_size // 2
        return sorted(candidate_blocks, key=lambda m: (m[0]-center)**2 + (m[1]-center)**2)[0]

    # ===== 阵法：模板与旋转/镜像 =====
    def _formation_templates(self) -> List[List[Tuple[int,int]]]:
        """
        返回若干阵法模板（相对锚点的偏移序列）。
        顺序即落子优先级。可按需再扩展。
        """
        # 1) 菱形优先（围绕中心形成紧凑空当，利于后续分叉）
        diamond = [
            (0, 0),          # 锚点（通常是中心）
            (-1, -1), (-1, 1), (1, -1), (1, 1),   # 四个对角
            (0, -1), (0, 1), (-1, 0), (1, 0),     # 再补四正交
        ]
        # 2) 对角“5 连势”骨架（尽量在中路形成可连五主线）
        diag5 = [
            (0, 0), (1, 1), (2, 2), (-1, -1), (-2, -2)
        ]
        # 3) 十字强化（进一步扩展中心控制）
        plus_ring = [
            (0, 0),
            (0, 2), (0, -2), (2, 0), (-2, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]
        if self.formation_name == "diamond_then_plus":
            return [diamond, plus_ring]
        elif self.formation_name == "diag5_then_diamond":
            return [diag5, diamond]
        else:
            return [diamond, diag5, plus_ring]

    @staticmethod
    def _rotations_and_reflections(offsets: List[Tuple[int,int]]) -> List[List[Tuple[int,int]]]:
        """生成 8 种变换（4 次旋转 × 是否镜像）。"""
        def rot90(p):  return (-p[1], p[0])
        def rot180(p): return (-p[0], -p[1])
        def rot270(p): return (p[1], -p[0])
        def mirror(p): return (p[0], -p[1])  # y 轴镜像

        variants = []
        rots = []
        # 旋转
        rots.append([p for p in offsets])
        rots.append([rot90(p) for p in offsets])
        rots.append([rot180(p) for p in offsets])
        rots.append([rot270(p) for p in offsets])
        # 每个旋转再镜像
        for r in rots:
            variants.append(r)
            variants.append([mirror(p) for p in r])
        return variants

    def _select_anchor(self, game_state: GameState) -> Optional[Tuple[int,int]]:
        """优先选中心，若被占则选距离中心最近的空点作为锚点。"""
        n = game_state.board_size
        center = (n // 2, n // 2)
        if game_state.is_valid_move(*center):
            return center
        # 在中心 8 邻域内找空点
        candidates = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = center[0] + dr, center[1] + dc
                if self._inb(n, r, c) and game_state.is_valid_move(r, c):
                    candidates.append((r, c))
        if candidates:
            candidates.sort(key=lambda rc: self._manhattan(center, rc))
            return candidates[0]
        # 再不行，找全局最近的空点
        all_legals = game_state.get_legal_moves()
        if not all_legals:
            return None
        all_legals.sort(key=lambda rc: self._manhattan(center, rc))
        return all_legals[0]

    def _best_oriented_plan(self, game_state: GameState, me: str, anchor: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        """
        根据当前局面，从模板中选出一个“可用格子最多”的朝向，得到绝对坐标序列。
        """
        n = game_state.board_size
        board = game_state.board

        def score_plan(abs_seq: List[Tuple[int,int]]) -> Tuple[int,int]:
            usable = 0
            tight = 0
            # 评分：合法空位 + 紧凑度（靠近我方已有子）
            for (r, c) in abs_seq:
                if not self._inb(n, r, c):
                    continue
                if board[r][c] != '.':
                    continue
                usable += 1
                # 邻接我方子加分
                for rr in range(max(0, r-1), min(n, r+2)):
                    for cc in range(max(0, c-1), min(n, c+2)):
                        if board[rr][cc] == me:
                            tight += 1
                            break
            return usable, tight

        best_seq = None
        best_key = (-1, -1)
        for templ in self._formation_templates():
            for variant in self._rotations_and_reflections(templ):
                abs_seq = [(anchor[0]+dr, anchor[1]+dc) for (dr, dc) in variant]
                usable, tight = score_plan(abs_seq)
                if (usable, tight) > best_key:
                    best_key = (usable, tight)
                    best_seq = abs_seq
        return best_seq

    def _ensure_formation_initialized(self, game_state: GameState, me: str):
        """初始化阵法的锚点与绝对计划序列；若已有则自动跳过。”"""
        if not self.formation_active:
            return
        if self.formation_plan_abs is not None:
            return
        anchor = self._select_anchor(game_state)
        if anchor is None:
            self.formation_active = False
            return
        plan = self._best_oriented_plan(game_state, me, anchor)
        if plan is None:
            self.formation_active = False
            return
        self.formation_anchor = anchor
        self.formation_plan_abs = plan
        self.formation_progress_idx = 0

    def _next_formation_move(self, game_state: GameState) -> Optional[Tuple[int,int]]:
        """
        从阵法计划中顺序挑选“当前仍然可下”的下一个点。
        若计划点被占，则自动跳过继续找下一个。
        若没有可用点，则关闭阵法。
        """
        if not self.formation_active or self.formation_plan_abs is None:
            return None
        n = game_state.board_size
        board = game_state.board
        idx = self.formation_progress_idx

        # 若进入中后期或对抗激烈，则停止阵法
        total_moves = len(game_state.move_history)
        if total_moves >= self.formation_max_plies:
            self.formation_active = False
            return None

        while idx < len(self.formation_plan_abs):
            r, c = self.formation_plan_abs[idx]
            idx += 1
            # 过滤越界/占用/极端边线（避免“回第一行”）
            if not self._inb(n, r, c):
                continue
            if board[r][c] != '.':
                continue
            # 非必要不去边线（除非局面很拥挤）
            if (r in (0, n-1) or c in (0, n-1)) and total_moves < 8:
                continue
            # 找到可下点
            self.formation_progress_idx = idx
            return (r, c)

        # 阵法走完或被完全挡掉
        self.formation_active = False
        return None

    # ===== 核心接口 =====
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        try:
            me = game_state.current_player.value
            rival = 'O' if me == 'X' else 'X'

            # 1) 我方必胜
            win_move = self._find_immediate_winning_move(game_state, me)
            if win_move:
                return win_move

            # 2) 必堵对手必胜
            block_win = self._find_immediate_winning_move(game_state, rival)
            if block_win:
                return block_win

            # 3) 只在“当前棋面已有活三”时拦截（取消一切预判式拦截）
            block_existing_open3 = self._find_block_for_existing_open_three(game_state, rival)
            if block_existing_open3:
                return block_existing_open3

            # 3.5) —— 阵法（安全期优先执行，保持“继续上一个布置的棋子”）
            if self.formation_active:
                self._ensure_formation_initialized(game_state, me)
                fm = self._next_formation_move(game_state)
                if fm is not None and game_state.is_valid_move(*fm):
                    return fm

            # 4) 创造自己活三
            create_open3 = self._find_open_three_move(game_state, me)
            if create_open3:
                return create_open3

            # 5) LLM 决策（若可用）
            if self.llm_client is not None:
                try:
                    board_str = game_state.format_board(formatter="standard")
                    board_prompt = f"Current board state:\n{board_str}\n"
                    board_prompt += f"Current player: {me}\n"
                    board_prompt += f"Move count: {len(game_state.move_history)}\n"
                    if game_state.move_history:
                        last = game_state.move_history[-1]
                        board_prompt += f"Last move: {last.player.value} at ({last.row}, {last.col})\n"
                    # 可加 allowed_moves（将阵法剩余推荐也传给 LLM 作为参考，非必须）
                    if self.formation_active and self.formation_plan_abs is not None:
                        rest = [p for p in self.formation_plan_abs[self.formation_progress_idx:]
                                if game_state.is_valid_move(*p)]
                        if rest:
                            board_prompt += f"Recommended opening cells: {rest[:6]}\n"

                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"{board_prompt}\n\nPlease provide your next move as JSON."},
                    ]
                    response = await self.llm_client.complete(messages)
                    move = self._parse_move_response(response, game_state)
                    return move
                except Exception as le:
                    print(f"LLM failed, fallback: {le}")

            # 6) fallback
            return self._get_fallback_move(game_state)

        except Exception as e:
            print(f"get_move error: {e}")
            self.invalid_moves += 1
            return self._get_fallback_move(game_state)

    # ===== 解析 LLM 输出 =====
    def _extract_json_block(self, text: str) -> Optional[str]:
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        start = text.find("{")
        if start == -1:
            return None
        stack = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    return text[start:i+1].strip()
        return None

    def _parse_move_response(self, response: str, game_state: GameState) -> Tuple[int, int]:
        try:
            json_str = self._extract_json_block(response)
            if not json_str:
                raise ValueError("No JSON block found")
            data = json.loads(json_str)
            move = data.get("move", {})
            row, col = move.get("row"), move.get("col")
            if isinstance(row, int) and isinstance(col, int) and game_state.is_valid_move(row, col):
                return (row, col)
            return self._get_fallback_move(game_state)
        except Exception as e:
            print(f"Parse error: {e}")
            return self._get_fallback_move(game_state)

    # ===== fallback =====
    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        n = game_state.board_size
        center = n // 2

        # 优先中心；不行则选“最近中心的非边线空位”
        if game_state.is_valid_move(center, center):
            return (center, center)

        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            raise RuntimeError("No valid moves available")

        def is_edge(r, c): return r in (0, n-1) or c in (0, n-1)
        def score(m):
            r, c = m
            s = 0.0
            if not is_edge(r, c):
                s += 5.0
            # 轻度中心偏好（避免“回第一行”）
            s -= (abs(r - center) + abs(c - center)) * 0.3
            return s

        legal_moves.sort(key=score, reverse=True)
        return legal_moves[0]