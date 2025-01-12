package main

import (
    "testing"
    "reflect"
    "github.com/chandler37/gobackgammon/brd"
)

// Helper function to clear all pieces from the board
func clearBoard(b *brd.Board) {
    for i := 1; i <= 24; i++ {
        numCheckers := b.Pips[i].NumCheckers()
        for j := 0; j < numCheckers; j++ {
            b.Pips[i].Subtract()
        }
    }
}

// Helper function to add multiple pieces to a position
func addPieces(b *brd.Board, pos int, color brd.Checker, count int) {
    for i := 0; i < count; i++ {
        b.Pips[pos].Add(color)
    }
}

// Helper function to setup terminal state test board
func setupTerminalBoard(setupFunc func(*brd.Board) brd.Checker) uint64 {
    b := brd.New(false)
    clearBoard(b)
    roller := setupFunc(b)
    id := newBoardID(b, 0, 0, 0)
 
    if b.Roller != roller {
        b.Roller = b.Roller.OtherColor()
     }

    //TakeTurnbyStateID(id) //only checks conditions for the player that rolls
    TakeTurnbyStateID(id) //so running twice so both red and white wins are checked

    return id
}

func TestMoveEncoding(t *testing.T) {
    tests := []struct {
        name     string
        sequence MoveSequence
    }{
        {
            name: "Single Move",
            sequence: MoveSequence{
                Moves: []Move{{From: 1, To: 7, Dice: 6}},
                Roll: brd.Roll{6, 0, 0, 0},
            },
        },
        {
            name: "Double Move",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 1, To: 7, Dice: 6},
                    {From: 7, To: 11, Dice: 4},
                },
                Roll: brd.Roll{6, 4, 0, 0},
            },
        },
        {
            name: "Bar Move",
            sequence: MoveSequence{
                Moves: []Move{{From: BAR_POS, To: 1, Dice: 1}},
                Roll: brd.Roll{1, 0, 0, 0},
            },
        },
        {
            name: "Bearing Off",
            sequence: MoveSequence{
                Moves: []Move{{From: 1, To: OFF_POS, Dice: 1}},
                Roll: brd.Roll{1, 0, 0, 0},
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            index := moveSequenceToIndex(tt.sequence)
            decoded := decodeIndexToMoveSequence(index)
            
            if !reflect.DeepEqual(tt.sequence.Moves, decoded.Moves) {
                t.Errorf("Move sequence encoding/decoding failed\nwant: %+v\ngot:  %+v",
                    tt.sequence.Moves, decoded.Moves)
            }
        })
    }
}

func TestTerminalStateDetection(t *testing.T) {
    tests := []struct {
        name         string
        setupBoard   func(*brd.Board) brd.Checker
        wantTerminal bool
        wantValue    float32
    }{
        // White winning positions
        {
            name: "Normal White Win",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffWhitePip, brd.White, 15) // White pieces borne off
                addPieces(b, 2 , brd.Red, 2)     // Red pieces on last point
                addPieces(b, brd.BorneOffRedPip , brd.Red, 13)     // Red pieces on last point
                return brd.White
            },
            wantTerminal: true,
            wantValue:    1.0,
        },
        {
            name: "White Gammon",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffWhitePip, brd.White, 15) // All white pieces borne off
                addPieces(b, 2, brd.Red, 15)                      // All red pieces in Red home
                return brd.White
            },
            wantTerminal: true,
            wantValue:    2.0,
            
        },
        {
            name: "White Backgammon on bar",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffWhitePip, brd.White, 15) // All white pieces borne off
                addPieces(b, brd.BarRedPip, brd.Red, 1)          // 1 on bar
                addPieces(b, 3, brd.Red, 14)                      // 4 Red pieces in Red Home
                return brd.White
            },
            wantTerminal: true,
            wantValue:    3.0,
        },
        {
            name: "White Backgammon on 1 in home",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffWhitePip, brd.White, 15) // All white pieces borne off
                addPieces(b, 22, brd.Red, 1)          // 1 red in white home
                addPieces(b, 3, brd.Red, 14)          // 4 Red pieces in Red Home
                return brd.White
            },
            wantTerminal: true,
            wantValue:    3.0,
        },
        // Red winning positions
        {
            name: "Normal Red Win",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffRedPip, brd.Red, 15)     // Red pieces borne off
                addPieces(b, brd.BorneOffWhitePip, brd.White, 4) // White pieces on last point
                addPieces(b, 22, brd.White, 11) // White pieces in white home
                return brd.Red
            },
            wantTerminal: true,
            wantValue:    -1.0,
        },
        {
            name: "Red Gammon",
            setupBoard: func(b *brd.Board)  brd.Checker {
                addPieces(b, brd.BorneOffRedPip, brd.Red, 15)    // All red pieces borne off
                addPieces(b, 23, brd.White, 15)                  // All white pieces in White home
                return brd.Red
            },
            wantTerminal: true,
            wantValue:    -2.0,
        },
        {
            name: "Red Backgammon on bar",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffRedPip, brd.Red, 15)     // All red pieces borne off
                addPieces(b, brd.BarWhitePip, brd.White, 15)      // All white pieces on bar
                return brd.Red
            },
            wantTerminal: true,
            wantValue:    -3.0,
        },
        {
            name: "Red Backgammon in home",
            setupBoard: func(b *brd.Board) brd.Checker {
                addPieces(b, brd.BorneOffRedPip, brd.Red, 15)     // All red pieces borne off
                addPieces(b, 3, brd.White, 1)      // All white pieces on bar
                addPieces(b, 21, brd.White, 14)      // All white pieces on bar
                return brd.Red
            },
            wantTerminal: true,
            wantValue:    -3.0,
        },
        {
            name: "Non-Terminal Position - Red",
            setupBoard: func(b *brd.Board) brd.Checker{
                *b = *brd.New(false) // Standard starting position
                return brd.Red
            },
            wantTerminal: false,
            wantValue:    0.0,
        },
        {
            name: "Non-Terminal Position - White",
            setupBoard: func(b *brd.Board) brd.Checker {
                *b = *brd.New(false) // Standard starting position
                return brd.White
            },
            wantTerminal: false,
            wantValue:    0.0,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            board := setupTerminalBoard(tt.setupBoard)
            
            for _, isTerminalFunc := range []struct {
                name string
                fn   func(uint64) (bool, float32)
            }{
                {"isTerminalState", isTerminalState},
                {"isTerminalStatev2", isTerminalStatev2},
            } {
                t.Run(isTerminalFunc.name, func(t *testing.T) {
                    gotTerminal, gotValue := isTerminalFunc.fn(board)
                    
                    if gotTerminal != tt.wantTerminal {
                        t.Errorf("%s() terminal = %v, want %v", 
                            isTerminalFunc.name, gotTerminal, tt.wantTerminal)
                    }
                    
                    if gotValue != tt.wantValue {
                        t.Errorf("%s() value = %v, want %v", 
                            isTerminalFunc.name, gotValue, tt.wantValue)
                    }
                })
            }
        })
    }
}

func TestGamePhaseDetection(t *testing.T) {
    tests := []struct {
        name       string
        setupBoard func() *brd.Board
        wantPhase  int
    }{
        {
            name: "Opening Game",
            setupBoard: func() *brd.Board {
                return brd.New(false) // Standard starting position
            },
            wantPhase: 0,
        },
        {
            name: "Middle Game",
            setupBoard: func() *brd.Board {
                b := brd.New(false)
                clearBoard(b)
                addPieces(b, 10, brd.White, 2)
                return b
            },
            wantPhase: 1,
        },
        {
            name: "Bearing In",
            setupBoard: func() *brd.Board {
                b := brd.New(false)
                clearBoard(b)
                addPieces(b, 1, brd.White, 3)
                addPieces(b, 2, brd.White, 3)
                addPieces(b, 3, brd.White, 4)
                return b
            },
            wantPhase: 2,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            board := tt.setupBoard()
            gotPhase := determineGamePhase(board)
            
            if gotPhase != tt.wantPhase {
                t.Errorf("determineGamePhase() = %v, want %v", 
                    gotPhase, tt.wantPhase)
            }
        })
    }
}