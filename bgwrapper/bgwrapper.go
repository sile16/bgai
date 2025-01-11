package main

/*
#cgo CFLAGS: -I.
#include <stdlib.h>
#include <stdio.h>
*/
import "C"
import (
    "encoding/json"
    "sync"
    "unsafe"
    //"fmt"

    "github.com/chandler37/gobackgammon/brd"
    "github.com/chandler37/gobackgammon/ai"
)

//-------------------------------------------------------------------
// GLOBAL BOARD STORE
//-------------------------------------------------------------------
var (
    mu     sync.Mutex
    nextID uint64
    boards = make(map[uint64]*brd.Board)
    simpleAIPlayer brd.Chooser
)

func newBoardID(b *brd.Board) uint64 {
    mu.Lock()
    defer mu.Unlock()
    nextID++
    boards[nextID] = b
    return nextID
}

func getBoard(id uint64) *brd.Board {
    mu.Lock()
    defer mu.Unlock()
    return boards[id]
}

func freeBoard(id uint64) {
    mu.Lock()
    defer mu.Unlock()
    delete(boards, id)
}

type BoardInfo struct {
    ID          int           `json:"id"`
    Player      float32       `json:"player"`    // White == 0, Red == 1
    Roll        [4]float32    `json:"roll"`      // Dice Roll
    RollUsed    [4]float32    `json:"roll_used"`      // Dice Roll
    WhitePips   [26]float32   `json:"white_pips"`
    RedPips     [26]float32   `json:"red_pips"`
}



//-------------------------------------------------------------------
// EXPORTED FUNCTIONS
//-------------------------------------------------------------------

//export FreeString
func FreeString(str *C.char) {
    //ptr := unsafe.Pointer(str)
    // Convert the pointer to an integer so we can log it
    //ptrAddr := uintptr(ptr)
    //fmt.Printf("Freeing pointer at 0x%x\n", ptrAddr)

    C.free(unsafe.Pointer(str))
}

func NewAIPlayer() {
    simpleAIPlayer = ai.MakePlayerConservative(0, nil)
}

//export NewBoard
func NewBoard() C.uint64_t {
    board := brd.New(false)

    // we are re-rolling because we want it possible for the first roll to be a double, 
    // because some Tavla variants do this, rather than backgammon's rule that the first roll cannot be a double.
    board.Roll.New(&board.RollUsed)  //this is from brd.go line 431
    
    return C.uint64_t(newBoardID(board))
}

//export ResetBoard
func ResetBoard(id C.uint64_t) {
    board := getBoard(uint64(id))
    if board == nil {
        return
    }
    *board = *brd.New(false)
}

//export GetBoardState
func GetBoardState(id C.uint64_t) *C.char {
    board := getBoard(uint64(id))
    if board == nil {
        return C.CString("{}")
    }
    data, err := json.Marshal(board)
    if err != nil {
        return C.CString("{}")
    }
    
    stringData := string(data)
    println("Allocating C string with length:", len(stringData))
    return C.CString(stringData)
}



func ApplyAIAction(id C.uint64_t) (C.uint64_t, C.int, C.int) {
    board := getBoard(uint64(id))
    if board == nil {
        return id, 0, 0
    }
    successors := board.LegalContinuations()
    analyzedCandidates := simpleAIPlayer(successors)

    var newBoard *brd.Board

    if len(analyzedCandidates) == 0 {
        newBoard = successors[0]
    } else {
        newBoard = analyzedCandidates[0].Board
    }

    acceptDoubleFunc := func(*brd.Board) bool {return false}
    victor, stakes, _ := newBoard.TakeTurn(nil, acceptDoubleFunc )

    // Optionally free the old board from the map to avoid leaks
    freeBoard(uint64(id))

    // Create a new ID for the successor
    return C.uint64_t(newBoardID(newBoard)), C.int(stakes), C.int(victor)
}


//export GetLegalMoves
//cache the legal moves for the board
func GetLegalMoves(id C.uint64_t) *C.char {
    board := getBoard(uint64(id))
    if board == nil {
        return C.CString("[]")
    }
    successors := board.LegalContinuations()

    moveList := make([]BoardInfo, 0, len(successors))
    for idx, successor := range successors {
        mi := EncodeBoard(successor)
        mi.ID = idx
        moveList = append(moveList, mi)
    }
    data, err := json.Marshal(moveList)
    if err != nil {
        return C.CString("[]")
    }
    return C.CString(string(data))
}

//export ApplyAction
// return the board ID, the points won by the victor (0 no winnder, 1, 2 or 3), the victor (nil, for none, 0 white, 1 for read),
func ApplyAction(id C.uint64_t, moveIndex C.int) (C.uint64_t, C.int, C.int) {
    board := getBoard(uint64(id))
    if board == nil {
        return id, 0, 0
    }

    successors := board.LegalContinuations()
    mIdx := int(moveIndex)
    if mIdx < 0 || mIdx >= len(successors) {
        // Invalid move index, return same board
        return id, 0, 0
    }

    // We get the new board from successors
    newBoard := successors[mIdx]

    acceptDoubleFunc := func(*brd.Board) bool {return false}
    victor, stakes, _ := newBoard.TakeTurn(nil, acceptDoubleFunc )

    // Optionally free the old board from the map to avoid leaks
    freeBoard(uint64(id))

    // Create a new ID for the successor
    return C.uint64_t(newBoardID(newBoard)), C.int(stakes), C.int(victor)
}

func EncodeBoard(board *brd.Board) BoardInfo {
    
    //new BoardInfo
    boardInfo := BoardInfo{}
    boardInfo.ID = int(0)

    if board.Roller == brd.White {
        boardInfo.Player = float32(0.0)
    } else {
        boardInfo.Player = float32(1.0)
    }

    for i := 0; i < 4; i++ {
        boardInfo.Roll[i] = float32(board.Roll[i])
        boardInfo.RollUsed[i] = float32(board.RollUsed[i])
    }

    for i := 1; i < 25; i++ {
        boardInfo.WhitePips[i-1] = float32(board.Pips[i].NumWhite())
        boardInfo.RedPips[i-1] = float32(board.Pips[i].NumRed())
    }
    boardInfo.RedPips[24] = float32(board.Pips[0].NumRed())  //red Pips off
    boardInfo.WhitePips[24] = float32(board.Pips[21].NumWhite())  //white Pips off
    boardInfo.RedPips[25] = float32(board.Pips[22].NumRed())  //red Bar
    boardInfo.WhitePips[25] = float32(board.Pips[23].NumWhite())  //white Bar
    
    return boardInfo
}

//export GetEncodedState
func GetEncodedState(id C.uint64_t) *C.char {
    board := getBoard(uint64(id))
    if board == nil {
        return C.CString("[]")
    }

    
    // - Dice values (2 values)
    // - Player to move (1 value)
    // Create a 4-plane encoding (196 features total):
    // Plane 0: White checker counts (28 positions)
    // Plane 1: Red checker counts (28 positions)
    // Plane 2: Current player's checkers (28 positions)
    // Plane 3: Opponent's checkers (28 positions)
    // Additional features:

    enc := EncodeBoard(board)

    data, err := json.Marshal(enc)
    if err != nil {
        return C.CString("[]")
    }
    return C.CString(string(data))
}


func main() {
    // Required for -buildmode=c-shared
}
