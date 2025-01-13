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
                Moves: []Move{{From: 24, To: OFF_POS, Dice: 1},
                              {From: 22, To: OFF_POS, Dice: 6}},
                Roll: brd.Roll{1, 6, 0, 0},
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

func TestCompleteActionSpace(t *testing.T) {
    // Map to track encoded indices
    seen := make(map[uint32]MoveSequence)

    var sequence  MoveSequence
    
    // Test single moves first
    for from1 := 0; from1 <= POINTS+1; from1++ {
        for die1 := 1; die1 <= 6; die1++ {
            for from2 := 0; from2 <= POINTS+1; from2++ {
                for die2 := 1; die2 <= 6; die2++ {
            
                    if die1 != die2 {
                        sequence = MoveSequence{
                            Moves: []Move{
                                {From: from1, Dice: brd.Die(die1), To: min(from1+die1, BEAR_OFF_POS)},
                                {From: from2, Dice: brd.Die(die2), To: min(from2+die2, BEAR_OFF_POS)},
                            },
                            Roll:  brd.Roll{brd.Die(die1), brd.Die(die2), 0, 0},
                        }
                        
                    } else {
                        sequence = MoveSequence{
                            Moves: []Move{
                                {From: from1, Dice: brd.Die(die1), To: min(from1+die1, BEAR_OFF_POS)},
                                {From: from1, Dice: brd.Die(die1), To: min(from1+die1, BEAR_OFF_POS)},
                                {From: from2, Dice: brd.Die(die1), To: min(from2+die2, BEAR_OFF_POS)},
                                {From: from2, Dice: brd.Die(die2), To: min(from2+die2, BEAR_OFF_POS)},
                            },
                            Roll:  brd.Roll{brd.Die(die1), brd.Die(die2), brd.Die(die2), brd.Die(die2)},
                        }
                    }

                    index := moveSequenceToIndex(sequence)
            
                    if existing, exists := seen[index]; exists {
                        t.Errorf("Index collision: %d maps to both %+v and %+v", 
                            index, existing, sequence)
                    }
                    if index != 0 {
                        seen[index] = sequence
                    }
                    
                }
            }
        }
    }
    
    // Count total unique indices
    t.Logf("Total unique single move indices: %d", len(seen))
}

func TestDoublesDiceEncoding(t *testing.T) {
    tests := []struct {
        name string
        roll brd.Roll
        moves []Move
        expectedValid bool
    }{
        {
            name: "Double 6s Four Moves",
            roll: brd.Roll{6, 6, 6, 6},
            moves: []Move{
                {From: 1, To: 7, Dice: 6},
                {From: 7, To: 13, Dice: 6},
                {From: 13, To: 19, Dice: 6},
                {From: 22, To: OFF_POS, Dice: 6},
            },
            expectedValid: true,
        },
        {
            name: "Double 3s Two Moves",
            roll: brd.Roll{3, 3, 3, 3},
            moves: []Move{
                {From: 4, To: 7, Dice: 3},
                {From: 7, To: 10, Dice: 3},
                {From: 7, To: 10, Dice: 3},
                {From: 7, To: 10, Dice: 3},
            },
            expectedValid: true,
        },
        {
            name: "Double 2s with Bar",
            roll: brd.Roll{2, 2, 2, 2},
            moves: []Move{
                {From: BAR_POS, To: 2, Dice: 2},
                {From: 2, To: 4, Dice: 2},
                {From: 4, To: 6, Dice: 2},
                {From: BAR_POS, To: 2, Dice: 2},
            },
            expectedValid: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            sequence := MoveSequence{
                Moves: tt.moves,
                Roll:  tt.roll,
            }
            
            index := moveSequenceToIndex(sequence)
            decoded := decodeIndexToMoveSequence(index)
            
            
            if !reflect.DeepEqual(sequence.Moves, decoded.Moves) {
                t.Errorf("Move sequence encoding/decoding failed\nwant: %+v\ngot:  %+v",
                    sequence.Moves, decoded.Moves)
            }
        })
    }
}

func TestMixedMoveScenarios(t *testing.T) {
    tests := []struct {
        name string
        sequence MoveSequence
        expectedValid bool
    }{
        {
            name: "Bar Move + Regular Move",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: BAR_POS, To: 1, Dice: 1},
                    {From: 1, To: 3, Dice: 2},
                },
                Roll: brd.Roll{1, 2, 0, 0},
            },
            expectedValid: true,
        },
        {
            name: "Bar Move + Bearing Off",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: BAR_POS, To: 3, Dice: 3},
                    {From: 23, To: OFF_POS, Dice: 2},
                },
                Roll: brd.Roll{3, 2, 0, 0},
            },
            expectedValid: true,
        },
        {
            name: "Multiple Pieces Same Point",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 12, To: 18, Dice: 6},
                    {From: 12, To: 15, Dice: 3},
                },
                Roll: brd.Roll{6, 3, 0, 0},
            },
            expectedValid: true,
        },
        {
            name: "Complex Mixed Sequence",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: BAR_POS, To: 1, Dice: 1},
                    {From: 1, To: 4, Dice: 3},

                },
                Roll: brd.Roll{1, 3, 6, 2},
            },
            expectedValid: true,
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

func TestInvalidMoves(t *testing.T) {
    tests := []struct {
        name string
        sequence MoveSequence
        expectError bool
    }{
        {
            name: "Move From Off Position",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: OFF_POS, To: 1, Dice: 1},
                },
                Roll: brd.Roll{1, 0, 0, 0},
            },
            expectError: true,
        },
        {
            name: "Invalid Point Numbers",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 30, To: 35, Dice: 5},
                },
                Roll: brd.Roll{5, 0, 0, 0},
            },
            expectError: true,
        },
        {
            name: "Dice Value Exceeds 6",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 1, To: 9, Dice: 8},
                },
                Roll: brd.Roll{8, 0, 0, 0},
            },
            expectError: true,
        },
        {
            name: "More Than Four Moves",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 1, To: 2, Dice: 1},
                    {From: 2, To: 3, Dice: 1},
                    {From: 3, To: 4, Dice: 1},
                    {From: 4, To: 5, Dice: 1},
                    {From: 5, To: 6, Dice: 1},
                },
                Roll: brd.Roll{1, 1, 1, 1},
            },
            expectError: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            index := moveSequenceToIndex(tt.sequence)

            // If we expected an error but got a valid index, that's a problem
            if tt.expectError && index != 0{
                t.Errorf("Expected invalid move to fail, got index: %d", index)
            } 
            
            // Try decoding and verify it fails appropriately
            decoded := decodeIndexToMoveSequence(index)

            if tt.expectError && len(decoded.Moves) > 0 {
                t.Error("Expected decode to fail and return 0 lenght moves")
            }

        
        })
    }
}


func TestBearingOffEncoding(t *testing.T) {
    tests := []struct {
        name     string
        sequence MoveSequence
    }{
        {
            name: "Single Bearing Off Move",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 19, To: OFF_POS, Dice: 6},
                },
                Roll: brd.Roll{6, 0, 0, 0},
            },
        },
        {
            name: "Double Dice Bearing Off",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 1, To: 7, Dice: 6},
                    {From: 7, To: 13, Dice: 6},
                    {From: 13, To: 19, Dice: 6},
                    {From: 19, To: OFF_POS, Dice: 6},
                },
                Roll: brd.Roll{6, 6, 6, 6},
            },
        },
        {
            name: "Mixed Regular and Bearing Off",
            sequence: MoveSequence{
                Moves: []Move{
                    {From: 13, To: 18, Dice: 5},
                    {From: 19, To: OFF_POS, Dice: 6},
                },
                Roll: brd.Roll{6, 5, 0, 0},
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            index := moveSequenceToIndex(tt.sequence)
            decoded := decodeIndexToMoveSequence(index)
            
            if !reflect.DeepEqual(tt.sequence.Moves, decoded.Moves) {
                t.Errorf("\nwant: %+v\ngot:  %+v",
                    tt.sequence.Moves, decoded.Moves)
            }
        })
    }
}