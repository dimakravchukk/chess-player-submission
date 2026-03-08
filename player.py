"""
player.py — TransformerPlayer using State-Value transformer (SV model).

Architecture:
  Encoder-only transformer trained to predict White win probability
  via K-bin categorical distribution.

Move selection:
  - White: maximize adjusted E[V(s')]
  - Black: minimize adjusted E[V(s')]

Safety layer:
  1. Immediate checkmate → return immediately
  2. Opening book (Catalan / Caro-Kann)
  3. Mate-in-1 search when opponent has ≤1 material
  4. Tactical penalty (one clean pass, mutually exclusive checks)
  5. Endgame capture bonus when winning by ≥5 material
"""

import os
import chess
import torch
import torch.nn as nn

try:
    from chess_tournament.players import Player as _BasePlayer
except ImportError:
    _BasePlayer = None


# ── Tokenizer ────────────────────────────────────────────────────────────────

FEN_LEN = 76


def tokenize_fen(fen: str) -> list:
    board = chess.Board(fen)

    board_str = ""
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            board_str += piece.symbol() if piece else "."

    active = "w" if board.turn == chess.WHITE else "b"

    c = ""
    if board.has_kingside_castling_rights(chess.WHITE):  c += "K"
    if board.has_queenside_castling_rights(chess.WHITE): c += "Q"
    if board.has_kingside_castling_rights(chess.BLACK):  c += "k"
    if board.has_queenside_castling_rights(chess.BLACK): c += "q"
    if not c: c = "-"
    castling_str = c.ljust(4, ".")[:4]

    ep = board.ep_square
    ep_str = chess.square_name(ep) if ep is not None else "-."
    hm = str(board.halfmove_clock).ljust(2, ".")[:2]
    fm = str(board.fullmove_number).ljust(3, ".")[:3]

    full = board_str + active + castling_str + ep_str + hm + fm
    return [ord(ch) for ch in full]


# ── Model ─────────────────────────────────────────────────────────────────────

class ChessSVTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6,
                 seq_len=FEN_LEN, dropout=0.1, k=128):
        super().__init__()
        self.k = k
        self.embedding     = nn.Embedding(256, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, k)
        self.seq_len = seq_len

        centers = (torch.arange(k, dtype=torch.float) + 0.5) / k
        self.register_buffer("bin_centers", centers)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding(torch.arange(self.seq_len, device=input_ids.device))
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

    def expected_value(self, input_ids):
        probs = torch.softmax(self.forward(input_ids), dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)


# ── Chess helpers ─────────────────────────────────────────────────────────────

# Empirical piece values (Kaufman)
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.5,
    chess.BISHOP: 3.5,
    chess.ROOK:   5.25,
    chess.QUEEN:  10.0,
    chess.KING:   0.0,
}


def material_balance(board: chess.Board, color: bool) -> float:
    """Sum of piece values for `color`, excluding king."""
    return sum(
        PIECE_VALUES.get(board.piece_at(sq).piece_type, 0.0)
        for sq in chess.SQUARES
        if board.piece_at(sq)
        and board.piece_at(sq).color == color
        and board.piece_at(sq).piece_type != chess.KING
    )


# ── Tactical penalty ──────────────────────────────────────────────────────────

def see(board: chess.Board, sq: int, my_color: bool) -> float:
    """
    Simplified Static Exchange Evaluation on `sq` after opponent captures.
    Returns net material gain/loss for `my_color` from the exchange sequence.
    Positive = we gain, negative = we lose.
    board must be the position AFTER our move (opponent to move).
    """
    opponent = not my_color

    # Find cheapest opponent attacker
    opp_attackers = board.attackers(opponent, sq)
    if not opp_attackers:
        return 0.0  # nobody attacks — no exchange

    # Value of piece on the square (what opponent captures)
    victim = board.piece_at(sq)
    if victim is None:
        return 0.0
    victim_val = PIECE_VALUES.get(victim.piece_type, 0.0)

    # Find cheapest opponent attacker
    cheapest_opp = min(opp_attackers,
                       key=lambda s: PIECE_VALUES.get(board.piece_at(s).piece_type, 99))
    opp_piece_val = PIECE_VALUES.get(board.piece_at(cheapest_opp).piece_type, 0.0)

    # Opponent captures — what can we recapture with?
    b2 = board.copy()
    b2.push(chess.Move(cheapest_opp, sq))  # opponent takes

    my_attackers = b2.attackers(my_color, sq)
    if not my_attackers:
        # Opponent takes for free
        return -victim_val

    # Find cheapest our recapture
    cheapest_mine = min(my_attackers,
                        key=lambda s: PIECE_VALUES.get(b2.piece_at(s).piece_type, 99))
    my_recapture_val = PIECE_VALUES.get(b2.piece_at(cheapest_mine).piece_type, 0.0)

    # Sequence: opponent takes victim (val=victim_val) with cheapest attacker (val=opp_piece_val)
    #           we recapture with cheapest piece (val=my_recapture_val)
    # Net for us: -victim_val + opp_piece_val - my_recapture_val
    #   (lost our piece, gained opponent's piece, may lose recapturing piece)
    # If net < 0 → bad exchange for us
    return -victim_val + opp_piece_val - my_recapture_val


def tactical_penalty(board_before: chess.Board, move: chess.Move, my_color: bool) -> float:
    """
    Generalized tactical penalty based on SEE (Static Exchange Evaluation).

    For every piece of ours that is on an attacked square after our move,
    compute the exchange outcome. If the exchange is losing (negative SEE),
    add a penalty scaled by piece value and how bad the exchange is.

    Works symmetrically for both colors and all piece types.
    """
    moved_piece = board_before.piece_at(move.from_square)
    if moved_piece is None:
        return 0.0

    board_after = board_before.copy()
    board_after.push(move)
    penalty = 0.0

    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if p is None or p.color != my_color or p.piece_type == chess.KING:
            continue
        piece_val = PIECE_VALUES.get(p.piece_type, 0.0)
        if piece_val < 1.0:
            continue  # skip pawns — too many false positives

        exchange = see(board_after, sq, my_color)
        if exchange < 0:
            # We lose material on this square — penalty proportional to loss
            # Weight more heavily for valuable pieces
            penalty += piece_val * min(1.0, -exchange / piece_val)

    return penalty


# ── Opening book ──────────────────────────────────────────────────────────────
# White: Catalan — d4, c4, g3, Bg2, Nf3, 0-0  (move-number based, opponent-independent)
# Black: Caro-Kann Classical — c6, d5, context-aware move 3, Bf5, Bg6, Nd7, Ngf6

_CATALAN = {1: "d2d4", 2: "c2c4", 3: "g2g3", 4: "f1g2", 5: "g1f3", 6: "e1g1"}

_CARO_KANN = {1: "c7c6", 2: "d7d5", 4: "c8f5", 5: "f5g6", 6: "b8d7", 7: "g8f6"}


def _try(board: chess.Board, uci: str):
    """Return uci if legal, else None."""
    m = chess.Move.from_uci(uci)
    return uci if m in board.legal_moves else None


def opening_book_move(fen: str, board: chess.Board):
    parts    = fen.split()
    turn     = parts[1]
    fullmove = int(parts[5])

    if turn == 'w' and fullmove in _CATALAN:
        return _try(board, _CATALAN[fullmove])

    if turn == 'b' and fullmove <= 7:
        if fullmove == 1:
            # c6 vs e4 (Caro-Kann), d5 vs anything else
            return _try(board, "c7c6") if board.piece_at(chess.E4) else _try(board, "d7d5")

        if fullmove == 2:
            return _try(board, "d7d5")

        if fullmove == 3:
            # Advance variation (e5 pawn present) → Bf5
            # Classical (d5 can take e4) → dxe4
            # Exchange (c6 can take d5) → cxd5
            return (
                _try(board, "c8f5") if board.piece_at(chess.E5) else
                _try(board, "d5e4") or
                _try(board, "c6d5")
            )

        if fullmove in _CARO_KANN:
            return _try(board, _CARO_KANN[fullmove])

    return None


# ── Player base class ─────────────────────────────────────────────────────────
# Use chess_tournament.players.Player if available (tournament environment),
# otherwise fall back to local stub (local testing).

if _BasePlayer is not None:
    Player = _BasePlayer
else:
    class Player:
        def __init__(self, name: str):
            self.name = name
        def get_move(self, fen: str):
            raise NotImplementedError


# ── TransformerPlayer ─────────────────────────────────────────────────────────

class TransformerPlayer(Player):

    MODEL_PATH     = "chess_sv_best.pt"
    TACTICAL_LAMBDA = 0.20   # penalty weight: adj = raw ± lambda * penalty

    # Kept for notebook compatibility
    MATERIAL_GUARD   = 3.0
    CONSTRAINT_LAMBDA = TACTICAL_LAMBDA

    def _load_model(self) -> ChessSVTransformer:
        path = self.MODEL_PATH
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(__file__), self.MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {self.MODEL_PATH}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg  = ckpt.get("config", {})
        model = ChessSVTransformer(
            d_model  = cfg.get("d_model",  256),
            n_heads  = cfg.get("n_heads",  8),
            n_layers = cfg.get("n_layers", 6),
            k        = cfg.get("k",        128),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        return model

    @torch.no_grad()
    def _sv_score(self, fen: str) -> float:
        tokens = torch.tensor(tokenize_fen(fen), dtype=torch.long).unsqueeze(0).to(self.device)
        return self.model.expected_value(tokens).item()

    @staticmethod
    def _move_leads_to_draw(board: chess.Board, move: chess.Move, depth: int = 2) -> bool:
        """
        True if this move leads to stalemate, insufficient material,
        50-move draw, or threefold repetition within `depth` plies.
        """
        board.push(move)
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves() or board.is_repetition(2)):
            board.pop()
            return True
        if depth >= 2:
            for reply in board.legal_moves:
                board.push(reply)
                draw = (board.is_stalemate() or board.is_insufficient_material()
                        or board.is_repetition(2))
                board.pop()
                if draw:
                    board.pop()
                    return True
        board.pop()
        return False

    def __init__(self, name: str = "SVBot"):
        super().__init__(name)
        self.device = torch.device(
            "cuda"  if torch.cuda.is_available()          else
            "mps"   if torch.backends.mps.is_available()  else
            "cpu"
        )
        self.model = self._load_model()
        self.model.eval()
        self._losing_streak = 0  # consecutive moves where we're clearly losing
        self.debug = False
        self._stats = {"book": 0, "mate1": 0, "free_cap": 0, "king_filter": 0, "draw_filtered": 0, "sv_pick": 0}

    def reset_history(self):
        self._losing_streak = 0

    def get_move(self, fen: str):
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return None

        my_color = board.turn
        is_white = my_color == chess.WHITE
        sign     = 1.0 if is_white else -1.0

        # ── 1. Opening book ───────────────────────────────────────────────────
        book = opening_book_move(fen, board)
        if book:
            if self.debug: self._stats["book"] += 1
            return book

        opp_mat  = material_balance(board, not my_color)
        my_mat   = material_balance(board, my_color)
        winning  = my_mat - opp_mat >= 5

        # ── 2. Immediate checkmate (mate-in-1) ────────────────────────────────
        for move in legal:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                if self.debug: self._stats["mate1"] += 1
                return move.uci()
            board.pop()

        # ── 2b. Free capture ──────────────────────────────────────────────────
        # Take only if: no immediate recapture AND capturing piece won't be
        # attacked by a cheaper piece on opponent's next move.
        best_free = None
        best_free_val = 0.0
        for move in legal:
            captured = board.piece_at(move.to_square)
            if captured is None or captured.color == my_color:
                continue
            cap_val = PIECE_VALUES.get(captured.piece_type, 0.0)
            if cap_val <= best_free_val:
                continue
            moving_piece = board.piece_at(move.from_square)
            if moving_piece is None:
                continue
            moving_val = PIECE_VALUES.get(moving_piece.piece_type, 0.0)
            board.push(move)
            # No immediate recapture
            if not board.attackers(not my_color, move.to_square):
                # Check if after opponent's reply, our piece gets attacked by
                # something cheaper (e.g. queen grabbing pawn, then pawn chases queen)
                safe_after = True
                for reply in board.legal_moves:
                    board.push(reply)
                    for atk_sq in board.attackers(not my_color, move.to_square):
                        atk_val = PIECE_VALUES.get(board.piece_at(atk_sq).piece_type, 99)
                        if atk_val < moving_val:
                            safe_after = False
                            break
                    board.pop()
                    if not safe_after:
                        break
                if safe_after:
                    best_free = move.uci()
                    best_free_val = cap_val
            board.pop()
        if best_free:
            if self.debug: self._stats["free_cap"] += 1
            return best_free

        # ── 2c. Early king move filter ────────────────────────────────────────
        # Before castling rights are lost, never move the king unless in check
        # or it's the only legal move. King moves in the opening are almost
        # always catastrophic.
        fullmove = int(fen.split()[5])
        has_castling = bool(board.castling_rights)
        if fullmove <= 15 and has_castling and not board.is_check():
            non_king_legal = [m for m in legal
                              if board.piece_at(m.from_square) and
                              board.piece_at(m.from_square).piece_type != chess.KING]
            if non_king_legal:
                if self.debug: self._stats["king_filter"] += 1
                legal = non_king_legal

        # ── 3. Score all candidates ───────────────────────────────────────────
        board_before = chess.Board(fen)  # reuse for penalty — create once
        my_king_sq   = board.king(my_color)
        opp_king_sq  = board.king(not my_color)

        safe_candidates  = []
        risky_candidates = []

        for move in legal:
            # Stalemate filter — only depth=1, fast
            causes_draw = self._move_leads_to_draw(board, move, depth=2)

            board.push(move)
            raw = self._sv_score(board.fen())
            pen = tactical_penalty(board_before, move, my_color)
            adj = raw - sign * self.TACTICAL_LAMBDA * pen

            if winning:
                # Pawn push bonus — pawns create unstoppable promotions
                moved_piece = board_before.piece_at(move.from_square)
                if moved_piece and moved_piece.piece_type == chess.PAWN:
                    adj += sign * 0.06

                # King proximity bonus — bring our king closer to opponent king
                if my_king_sq is not None and opp_king_sq is not None:
                    new_my_king = board.king(my_color)
                    if new_my_king is not None:
                        dist = chess.square_distance(new_my_king, opp_king_sq)
                        adj += sign * (7 - dist) * 0.01

                # Capture bonus — prefer eating pieces when ahead
                captured = board_before.piece_at(move.to_square)
                if captured and captured.color != my_color:
                    cap_val = PIECE_VALUES.get(captured.piece_type, 0.0)
                    adj    += sign * min(0.02 * cap_val, 0.08)

            board.pop()

            if causes_draw:
                if self.debug: self._stats["draw_filtered"] += 1
                risky_candidates.append((move, adj))
            else:
                safe_candidates.append((move, adj))

        candidates = safe_candidates if safe_candidates else risky_candidates

        # ── 4. Pick best ──────────────────────────────────────────────────────
        best = max(candidates, key=lambda x: x[1]) if is_white else min(candidates, key=lambda x: x[1])
        if self.debug: self._stats["sv_pick"] += 1
        return best[0].uci()
