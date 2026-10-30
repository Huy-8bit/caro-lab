"use client";

import { useMemo } from "react";

const symbols: Record<string, string> = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function parseFen(fen: string) {
  const ranks = (fen || START_FEN).split(" ")[0].split("/");
  return ranks.map((rank) => {
    const row: string[] = [];
    for (const token of rank) {
      if (/\d/.test(token)) row.push(...Array(Number(token)).fill(""));
      else row.push(token);
    }
    return row;
  });
}

export function ChessBoard({ fen = START_FEN, orientation = "white", lastMove, compact = false }: {
  fen?: string; orientation?: "white" | "black"; lastMove?: string; compact?: boolean;
}) {
  const squares = useMemo(() => {
    const rows = parseFen(fen);
    const result = rows.flatMap((row, rank) => row.map((piece, file) => ({
      piece, file, rank, name: `${"abcdefgh"[file]}${8 - rank}`,
    })));
    return orientation === "black" ? result.reverse() : result;
  }, [fen, orientation]);
  const highlighted = new Set(lastMove ? [lastMove.slice(0, 2), lastMove.slice(2, 4)] : []);
  return (
    <div className={`chessboard ${compact ? "compactBoard" : ""}`} aria-label={`Chess board viewed from ${orientation}`}>
      {squares.map((square) => (
        <div
          key={square.name}
          className={`boardSquare ${(square.rank + square.file) % 2 ? "boardDark" : "boardLight"} ${highlighted.has(square.name) ? "lastSquare" : ""}`}
          data-square={square.name}
        >
          {square.file === (orientation === "white" ? 0 : 7) && <span className="rankLabel">{8 - square.rank}</span>}
          {square.rank === (orientation === "white" ? 7 : 0) && <span className="fileLabel">{"abcdefgh"[square.file]}</span>}
          {square.piece && <span className={`piece ${square.piece === square.piece.toUpperCase() ? "whitePiece" : "blackPiece"}`}>{symbols[square.piece]}</span>}
        </div>
      ))}
    </div>
  );
}
