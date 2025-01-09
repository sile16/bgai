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
    "fmt"

    "github.com/chandler37/gobackgammon/brd"
)

//-------------------------------------------------------------------
// GLOBAL BOARD STORE
//-------------------------------------------------------------------
var (
    mu     sync.Mutex
    nextID uint64
    boards = make(map[uint64]*brd.Board)
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

//-------------------------------------------------------------------
// EXPORTED FUNCTIONS
//-------------------------------------------------------------------

//export FreeString
func FreeString(str *C.char) {

    ptr := unsafe.Pointer(str)

    // Convert the pointer to an integer so we can log it
    ptrAddr := uintptr(ptr)
    fmt.Printf("Freeing pointer at 0x%x\n", ptrAddr)

    C.free(unsafe.Pointer(str))
}

//export HelloWorld
func HelloWorld() *C.char {
    msg := "Hello from Go"
    size := len(msg) + 1
    ptr := C.malloc(C.size_t(size))


    fmt.Printf("String content: %s\n", msg)
    fmt.Printf("Go pointer integer: %d hex: 0x%x\n", uintptr(ptr), uintptr(ptr))

    // Convert the pointer to an integer so we can log it
    ptrAddr := uintptr(unsafe.Pointer(ptr))
    fmt.Printf("Allocating pointer at 0x%x with size %d\n", ptrAddr, size)

    // Convert that memory block to a Go slice
    buf := (*[1 << 30]byte)(ptr)[:size:size]
    copy(buf, msg)   // copy "Hello from Go"
    buf[len(msg)] = 0 // null-terminate


    return (*C.char)(ptr)
}


//export NewBoard
func NewBoard() C.uint64_t {
    board := brd.New(false)
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

// MoveInfo is an example struct describing a legal move.
type MoveInfo struct {
    ID int `json:"id"`
    // You can add more fields (startPip, endPip, diceUsed, etc.)
}

//export GetLegalMoves
func GetLegalMoves(id C.uint64_t) *C.char {
    board := getBoard(uint64(id))
    if board == nil {
        return C.CString("[]")
    }
    successors := board.LegalContinuations()

    moveList := make([]MoveInfo, 0, len(successors))
    for idx := range successors {
        moveList = append(moveList, MoveInfo{ID: idx})
    }
    data, err := json.Marshal(moveList)
    if err != nil {
        return C.CString("[]")
    }
    return C.CString(string(data))
}

//export ApplyAction
func ApplyAction(id C.uint64_t, moveIndex C.int) C.uint64_t {
    board := getBoard(uint64(id))
    if board == nil {
        return id
    }

    successors := board.LegalContinuations()
    mIdx := int(moveIndex)
    if mIdx < 0 || mIdx >= len(successors) {
        // Invalid move index, return same board
        return id
    }

    // We get the new board from successors
    newBoard := successors[mIdx]

    // Optionally free the old board from the map to avoid leaks
    freeBoard(uint64(id))

    // Create a new ID for the successor
    return C.uint64_t(newBoardID(newBoard))
}

type MoveInfo struct {
    ID          int     `json:"id"`
    StartPip    int     `json:"start_pip"`    // Origin pip (0-25, where 0=bar, 25=off)
    EndPip      int     `json:"end_pip"`      // Destination pip
    DiceUsed    int     `json:"dice_used"`    // Which die was used (value 1-6)
    IsHit       bool    `json:"is_hit"`       // Whether this move hits opponent's checker
    IsBearoff   bool    `json:"is_bearoff"`   // Whether this move bears off a checker
    IsBar       bool    `json:"is_bar"`       // Whether this move is from the bar
    MoveScore   float64 `json:"move_score"`   // Heuristic score (optional)
}

//export GetEncodedState
func GetEncodedState(id C.uint64_t) *C.char {
    board := getBoard(uint64(id))
    if board == nil {
        return C.CString("[]")
    }

    // Create a 4-plane encoding (196 features total):
    // Plane 0: White checker counts (28 positions)
    // Plane 1: Red checker counts (28 positions)
    // Plane 2: Current player's checkers (28 positions)
    // Plane 3: Opponent's checkers (28 positions)
    // Additional features:
    // - Dice values (2 values)
    // - Player to move (1 value)
    // - Crawford game flag (1 value)
    // - Match score (2 values)
    // - Pip counts (2 values)

    enc := make([]float32, 196)
    
    // Fill basic position planes
    for i := 0; i < 28; i++ {
        // White pieces
        enc[i] = float32(board.Pips[i].NumWhite())
        // Red pieces
        enc[28+i] = float32(board.Pips[i].NumRed())
        
        // Current player pieces
        if board.PlayerToMove == White {
            enc[56+i] = float32(board.Pips[i].NumWhite())
            enc[84+i] = float32(board.Pips[i].NumRed())
        } else {
            enc[56+i] = float32(board.Pips[i].NumRed())
            enc[84+i] = float32(board.Pips[i].NumWhite())
        }
    }

    // Add dice values
    offset := 112
    enc[offset] = float32(board.Dice[0])
    enc[offset+1] = float32(board.Dice[1])
    
    // Add player to move
    enc[offset+2] = float32(board.PlayerToMove)
    
    // Add match context features...
    // (additional features would go here)

    data, err := json.Marshal(enc)
    if err != nil {
        return C.CString("[]")
    }
    return C.CString(string(data))
}

func main() {
    // Required for -buildmode=c-shared
}
