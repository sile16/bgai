package main

// build with: go build -buildmode=c-shared -o libbackgammon.dylib 

/*
#cgo CFLAGS: -I.
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    unsigned long long new_id;
    int victor;
    int points;
} ActionResult;

typedef struct {
    int id;
    float player;
    float roll[4];
    float roll_used[4];
    float white_pips[26];
    float red_pips[26];
} MoveInfo;

typedef struct {
    int count;
    MoveInfo* moves;
} MoveList;
*/
import "C"
import (
    "encoding/json"
    "sync"
    "unsafe"
    "math/rand"

    "github.com/chandler37/gobackgammon/brd"
    "github.com/chandler37/gobackgammon/ai"
)


type BoardState struct {
    board *brd.Board
    parent uint64        // Add parent board reference
    legalMoves []*brd.Board
    moveCache []BoardInfo
    victor brd.Checker
    points int
}

var (
    mu     sync.Mutex
    nextID uint64
    boards = make(map[uint64]*BoardState)

    choosers [3]brd.Chooser
    worstChooser brd.Chooser

)

// AIType represents different types of AI players
type AIType int

const (
    Smart AIType = iota
    Random
    Worst
)


// makeWorstChooser creates a chooser that always picks the worst move
// makeWorstChooser creates a chooser that always picks the worst move
func makeWorstChooser() brd.Chooser {
    if worstChooser == nil {
        worstChooser = ai.MakePlayerConservative(0, nil)
    }

    return func(candidates []*brd.Board) []brd.AnalyzedBoard {
        analyzed := worstChooser(candidates)
        // Return only the last element in the list
        if len(analyzed) > 0 {
            return analyzed[len(analyzed)-1:]
        }
        return analyzed
    }
}

func newBoardID(b *brd.Board,  parentID uint64, victor brd.Checker, stakes int) uint64 {
    mu.Lock()
    defer mu.Unlock()
    nextID++
    state := &BoardState{
        board: b,
        parent: parentID,
        legalMoves: b.LegalContinuations(),
        victor: victor,
        points: stakes,
    }
    state.moveCache = encodeMoves(state.legalMoves)
    boards[nextID] = state
    return nextID
}

func getBoardState(id uint64) *BoardState {
    if id == 0 {
        return nil
    }
    mu.Lock()
    defer mu.Unlock()
    return boards[id]
}

func freeBoard(id uint64) {
    mu.Lock()
    defer mu.Unlock()
    state, exists := boards[id]
    if !exists {
        return
    }
    
    // Free the parent if this was the only reference to it
    if state.parent != 0 {
        parentState := boards[state.parent]
        if parentState != nil {
            // You might want to add reference counting if multiple boards could share the same parent
            freeBoard(state.parent)
        }
    }
    
    delete(boards, id)
}

type BoardInfo struct {
    ID          int           `json:"id"`
    Player      float32       `json:"player"`
    Roll        [4]float32    `json:"roll"`
    RollUsed    [4]float32    `json:"roll_used"`
    WhitePips   [26]float32   `json:"white_pips"`
    RedPips     [26]float32   `json:"red_pips"`
    Evaluation  float32       `json:"evaluation,omitempty"`
}

//export SetRandomSeed
func SetRandomSeed(seed C.uint64_t) {
    rand.New(rand.NewSource(int64(seed)))
}

//export FreeString
func FreeString(str *C.char) {
    C.free(unsafe.Pointer(str))
}



//export NewBoard
func NewBoard() C.uint64_t {
    board := brd.New(false)
    board.Roll.New(&board.RollUsed)
    return C.uint64_t(newBoardID(board, 0, 0, 0))
}

//export ResetBoard
func ResetBoard(id C.uint64_t) {
    state := getBoardState(uint64(id))
    if state == nil {
        return
    }
    newBoard := brd.New(false)
    state.board = newBoard
    state.legalMoves = newBoard.LegalContinuations()
    state.moveCache = encodeMoves(state.legalMoves)
}

func encodeMoves(moves []*brd.Board) []BoardInfo {
    moveList := make([]BoardInfo, 0, len(moves))
    for idx, move := range moves {
        mi := EncodeBoard(move)
        mi.ID = idx
        moveList = append(moveList, mi)
    }
    return moveList
}

//export GetBoardState
func GetBoardState(id C.uint64_t) *C.char {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.CString("{}")
    }
    data, err := json.Marshal(state.board)
    if err != nil {
        return C.CString("{}")
    }
    return C.CString(string(data))
}

//export GetLegalMoves
func GetLegalMoves(id C.uint64_t) C.MoveList {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.MoveList{count: 0, moves: nil}
    }

    moves := state.legalMoves
    moveCount := len(moves)
    if moveCount == 0 {
        return C.MoveList{count: 0, moves: nil}
    }

    // Allocate array of MoveInfo structs
    moveInfoArray := (*C.MoveInfo)(C.malloc(C.size_t(moveCount) * C.size_t(unsafe.Sizeof(C.MoveInfo{}))))
    moveInfoSlice := unsafe.Slice(moveInfoArray, moveCount)

    // Fill the array
    for i, move := range moves {
        moveInfo := encodeBoardToMoveInfo(move, i)
        moveInfoSlice[i] = moveInfo
    }

    return C.MoveList{
        count: C.int(moveCount),
        moves: moveInfoArray,
    }
}


// Struct to hold multiple return values
type ActionResult struct {
    newID   C.uint64_t
    victor  C.int
    points  C.int
}

//take a turn on the current board by state ID
func TakeTurnbyStateID(id uint64 ) ActionResult {
    state := getBoardState(uint64(id))
    if state == nil {
        return ActionResult{
            newID: C.uint64_t(id),
            victor: 0,
            points: 0,
        }
    }
    
    acceptDoubleFunc := func(*brd.Board) bool { return false }
    victor, stakes, _ := state.board.TakeTurn(nil, acceptDoubleFunc)

    state.victor = victor
    state.points = stakes

    return ActionResult{
        newID: C.uint64_t(id),
        victor: C.int(victor),
        points: C.int(stakes),
    }
}

//export ApplyAction
func ApplyAction(id C.uint64_t, moveIndex C.int) C.ActionResult {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.ActionResult{
            new_id: id,
            victor: 0,
            points: 0,
        }
    }

    mIdx := int(moveIndex)
    if mIdx < 0 || mIdx >= len(state.legalMoves) {
        return C.ActionResult{
            new_id: id,
            victor: 0,
            points: 0,
        }
    }

    newBoard := state.legalMoves[mIdx]
    acceptDoubleFunc := func(*brd.Board) bool { return false }
    victor, stakes, _ := newBoard.TakeTurn(nil, acceptDoubleFunc)

    newID := newBoardID(newBoard, uint64(id), victor, stakes)
    freeBoard(state.parent)

    return C.ActionResult{
        new_id: C.ulonglong(newID),
        victor: C.int(victor),
        points: C.int(stakes),
    }
}

//export FreeMoveList
func FreeMoveList(list C.MoveList) {
    if list.moves != nil {
        C.free(unsafe.Pointer(list.moves))
    }
}

//export SetWhiteAIChooser
func SetWhiteAIChooser(aiType C.int) {
    choosers[1] = getChooser(AIType(aiType))
}

//export SetRedAIChooser
func SetRedAIChooser(aiType C.int) {
    choosers[2] = getChooser(AIType(aiType))
}

func getChooser(aiType AIType) brd.Chooser {
    switch aiType {
    case Smart:
        return ai.MakePlayerConservative(0, nil)
    case Random:
        return ai.PlayerRandom
    case Worst:
        return makeWorstChooser()
    default:
        return ai.MakePlayerConservative(0, nil)
    }
}

//export ApplyAIAction
func ApplyAIAction(id C.uint64_t) C.ActionResult {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.ActionResult{
            new_id: id,
            victor: 0,
            points: 0,
        }
    }

    // Get the appropriate chooser based on current player
    playerIndex := state.board.Roller

    // Initialize chooser if not set
    if choosers[playerIndex] == nil {
        choosers[playerIndex] = getChooser(Smart) // Default to smart AI
    }

    // Get analyzed moves using the current player's chooser
    analyzedCandidates := choosers[playerIndex](state.legalMoves)
    
    var newBoard *brd.Board
    if len(analyzedCandidates) == 0 {
        if len(state.legalMoves) == 0 {
            return C.ActionResult{
                new_id: id,
                victor: 0,
                points: 0,
            }
        }
        newBoard = state.legalMoves[0]
    } else {
        newBoard = analyzedCandidates[0].Board
    }

    acceptDoubleFunc := func(*brd.Board) bool { return false }
    victor, stakes, _ := newBoard.TakeTurn(nil, acceptDoubleFunc)

    freeBoard(state.parent)
    newID := newBoardID(newBoard, uint64(id), victor, stakes)

    return C.ActionResult{
        new_id: C.ulonglong(newID),
        victor: C.int(victor),
        points: C.int(stakes),
    }
}

//export GetEncodedState
func GetEncodedState(id C.uint64_t) *C.char {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.CString("[]")
    }

    enc := EncodeBoard(state.board)
    data, err := json.Marshal(enc)
    if err != nil {
        return C.CString("[]")
    }
    return C.CString(string(data))
}

func encodeBoardToMoveInfo(board *brd.Board, id int) C.MoveInfo {
    var moveInfo C.MoveInfo
    moveInfo.id = C.int(id)
    
    if board.Roller == brd.White {
        moveInfo.player = C.float(0.0)
    } else {
        moveInfo.player = C.float(1.0)
    }

    // Copy roll and roll_used
    for i := 0; i < 4; i++ {
        moveInfo.roll[i] = C.float(board.Roll[i])
        moveInfo.roll_used[i] = C.float(board.RollUsed[i])
    }

    // Copy pip counts
    for i := 1; i < 25; i++ {
        moveInfo.white_pips[i-1] = C.float(board.Pips[i].NumWhite())
        moveInfo.red_pips[i-1] = C.float(board.Pips[i].NumRed())
    }
    
    // Special positions
    moveInfo.red_pips[24] = C.float(board.Pips[0].NumRed())      // red off
    moveInfo.white_pips[24] = C.float(board.Pips[21].NumWhite()) // white off
    moveInfo.red_pips[25] = C.float(board.Pips[22].NumRed())     // red bar
    moveInfo.white_pips[25] = C.float(board.Pips[23].NumWhite()) // white bar
    
    return moveInfo
}

func EncodeBoard(board *brd.Board) BoardInfo {
    boardInfo := BoardInfo{}
    
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
    boardInfo.RedPips[24] = float32(board.Pips[0].NumRed())
    boardInfo.WhitePips[24] = float32(board.Pips[21].NumWhite())
    boardInfo.RedPips[25] = float32(board.Pips[22].NumRed())
    boardInfo.WhitePips[25] = float32(board.Pips[23].NumWhite())
    
    return boardInfo
}

// Add new struct for encoded state
type EncodedState struct {
    StateData      []float32       `json:"state_data"`    // Flattened state tensor
    Shape          []int           `json:"shape"`         // Shape of the tensor (30, 24)
    LegalMoves     []uint32        `json:"legal_moves"`   // Encoded legal moves
    IsTerminal     bool            `json:"is_terminal"`   // Whether game is over
    CurrentValue   float32         `json:"value"`         // Current position value if terminal
    GamePhase      int             `json:"game_phase"`    // Added game phase tracking
    LegalMovesObj  []MoveSequence  `json:"legal_moves_obj"`   // json legal moves
}

func encodeStateToTensor(board *brd.Board) []float32 {
    // Initialize state tensor (30 channels x 24 positions)
    stateData := make([]float32, 30*24)
    
    // Encode piece positions (first 15 channels for each color)
    for i := 1; i < 25; i++ {
        whiteCount := board.Pips[i].NumWhite()
        redCount := board.Pips[i].NumRed()
        
        // Encode white pieces
        for w := 0; w < min(whiteCount, 15); w++ {
            stateData[w*24 + i-1] = 1.0
        }
        
        // Encode red pieces
        for r := 0; r < min(redCount, 15); r++ {
            stateData[(15+r)*24 + i-1] = 1.0
        }
    }
    
    // Current player channel (channel 29)
    playerIdx := 29 * 24
    if board.Roller == brd.White {
        for i := 0; i < 24; i++ {
            stateData[playerIdx+i] = 1.0
        }
    }
    
    // Dice values in last row of channel 29
    if board.Roll[0] > 0 {
        stateData[playerIdx+20] = float32(board.Roll[0]) / 6.0
        stateData[playerIdx+21] = float32(board.Roll[1]) / 6.0
        if board.Roll[2] > 0 {  // Handle doubles
            stateData[playerIdx+22] = float32(board.Roll[2]) / 6.0
            stateData[playerIdx+23] = float32(board.Roll[3]) / 6.0
        }
    }
    
    // Bar pieces encoded in channel 28
    barIdx := 28 * 24
    stateData[barIdx] = float32(board.Pips[23].NumWhite()) / 15.0      // White bar
    stateData[barIdx+23] = float32(board.Pips[22].NumRed()) / 15.0     // Red bar
    
    return stateData
}

// encodeStateToTensor converts board state to neural network input format
func oldEncodeStateToTensor(board *brd.Board) []float32 {
    // Initialize state tensor (30 channels x 24 positions)
    stateData := make([]float32, 30*24)
    
    // Encode piece positions (first 15 channels for each color)
    for i := 1; i < 25; i++ {
        whiteCount := board.Pips[i].NumWhite()
        redCount := board.Pips[i].NumRed()
        
        // Encode white pieces
        for w := 0; w < min(whiteCount, 15); w++ {
            stateData[w*24 + i-1] = 1.0
        }
        
        // Encode red pieces
        for r := 0; r < min(redCount, 15); r++ {
            stateData[(15+r)*24 + i-1] = 1.0
        }
    }
    
    // Current player channel
    playerIdx := 29 * 24
    if board.Roller == brd.White {
        for i := 0; i < 24; i++ {
            stateData[playerIdx+i] = 1.0
        }
    }
    
    // Dice values (normalized)
    diceStart := 30 * 24
    for i := 0; i < 4; i++ {
        if board.Roll[i] > 0 {
            stateData[diceStart+i] = float32(board.Roll[i]) / 6.0
        }
    }
    
    // Bar pieces
    barStart := 35 * 24
    stateData[barStart] = float32(board.Pips[23].NumWhite()) / 15.0      // White bar
    stateData[barStart+23] = float32(board.Pips[22].NumRed()) / 15.0     // Red bar
    
    return stateData
}


// encodeLegalMoves converts legal moves to integer indices
func encodeLegalMoves(stateID C.uint64_t) []uint32 {
    brdState := boards[uint64(stateID)]

    encoded := make([]uint32, len(brdState.legalMoves))
    for i := range encoded {
        encoded[i] = encodeMoveToIndex(brdState.board, brdState.legalMoves[i])
    }
    return encoded
}

// encodeMoveToIndex converts a move to a unique integer index
func encodeMoveToIndex(before *brd.Board, after *brd.Board ) uint32 {
// Get the parent board from the state

    sequence := extractMoveSequence(before, after)
    return moveSequenceToIndex(sequence)
}

func extractMoveSequence(before, after *brd.Board) MoveSequence {
    
    sequence := MoveSequence{
        Moves: make([]Move, 0, MAX_MOVES),
        Roll:  before.Roll,
    }
    
    // Compare pip counts to detect moves
    for i := 0; i <= POINTS+1; i++ {
        beforeWhite := before.Pips[i].NumWhite()
        afterWhite := after.Pips[i].NumWhite()
        beforeRed := before.Pips[i].NumRed()
        afterRed := after.Pips[i].NumRed()
        
        // Detect piece removals (from moves)
        if beforeWhite > afterWhite {
            for j := 0; j < beforeWhite-afterWhite; j++ {
                sequence.Moves = append(sequence.Moves, Move{From: i, To: NULL_POINT, Dice: NULL_POINT})
            }
        }
        if beforeRed > afterRed {
            for j := 0; j < beforeRed-afterRed; j++ {
                sequence.Moves = append(sequence.Moves, Move{From: i, To: NULL_POINT, Dice: NULL_POINT})
            }
        }
    }
    
    // Match moves with dice values and destinations
    matchMovesWithDice(before, after, &sequence)
    
    return sequence
}

func matchMovesWithDice(before, after *brd.Board, sequence *MoveSequence) {
    // For each preliminary move, find its destination and matching die
    for i := range sequence.Moves {
        from := sequence.Moves[i].From
        
        // Find destination by checking piece increases
        var to int
        for j := 0; j <= POINTS+1; j++ {
            if before.Roller == brd.White {
                if after.Pips[j].NumWhite() > before.Pips[j].NumWhite() {
                    to = j
                    break
                }
            } else {
                if after.Pips[j].NumRed() > before.Pips[j].NumRed() {
                    to = j
                    break
                }
            }
        }
        
        // Calculate dice value based on special positions
        var dice int
        switch {
        case from == BAR_POS:
            // Moving from bar - dice value is the destination point number
            if before.Roller == brd.White {
                dice = to
            } else {
                dice = 25 - to
            }
        case to == OFF_POS:
            // Bearing off - dice value is the distance from home
            if before.Roller == brd.White {
                dice = from
            } else {
                dice = 25 - from
            }
        default:
            // Normal move - calculate based on distance
            dice = abs(to - from)
            if before.Roller == brd.Red {
                dice = -dice // Preserve direction information
            }
        }
        
        sequence.Moves[i].To = to
        sequence.Moves[i].Dice = brd.Die(dice)
    }
}


func normalizePosition(pos int) int {
    if pos < 0 || pos > POINTS+1 {
        return 0
    }
    return pos
}

func denormalizePosition(pos int) int {
    return pos
}

func normalizeDice(dice, from, to int) int {
    if to == OFF_POS {
        return calculateBearingOffDice(from)
    }
    return abs(dice)
}

func calculateBearingOffDice(from int) int {
    if from <= 6 {
        return from
    }
    return 6
}

func isValidPosition(pos int) bool {
    return pos >= 0 && pos <= POINTS+1
}

// Additional helper function for doubles handling
func handleDoubles(sequence MoveSequence) bool {
    // Track used dice values for doubles
    usedDice := make(map[brd.Die]brd.Die)
    
    for _, move := range sequence.Moves {
        diceVal := move.Dice
        usedDice[diceVal]++
        
        // Check if we've used more dice than available
        if usedDice[diceVal] > 4 {
            return false
        }
    }
    
    return true
}

func isTerminalState( brdStateID uint64) (bool, float32) {

    boardState := getBoardState(brdStateID)

    if boardState.victor != 0 {
        if boardState.victor == brd.White {
            return true, float32(boardState.points) 

        } else {
            return true, -1.0 * float32(boardState.points) 
        }
    }
    return false, 0.0
}

func isTerminalStatev2(brdStateID uint64) (bool, float32) {

    boardState := getBoardState(brdStateID)
    board := boardState.board
    // Check if all pieces are borne off
    whiteOff := board.Pips[brd.BorneOffWhitePip].NumWhite() // White bearing off position
    redOff := board.Pips[brd.BorneOffRedPip].NumRed()      // Red bearing off position
    whiteBar := board.Pips[brd.BarWhitePip].NumWhite()      // White bar position
    redBar := board.Pips[brd.BarRedPip].NumRed()            // Red bar position

    
    // Handle white win conditions
    if whiteOff == 15 {
        // Check for backgammon (opponent has pieces on bar or in winner's home board)

        if redOff > 0 {
            return true, 1.0
        }
        
        if redBar > 0 || hasOpponentPiecesInHomeBoard(board, brd.White) {
            return true, 3.0 // White wins backgammon
        }

        return true, 2.0 // White wins gammon
    }
    
    // Handle red win conditions
    if redOff == 15 {
        if whiteOff > 0 {
            return true, -1.0
        }
        
        if whiteBar > 0 || hasOpponentPiecesInHomeBoard(board, brd.Red) {
            return true, -3.0 // White wins backgammon
        }

        return true, -2.0 // White wins gammon
    }
    
    return false, 0.0 // Game not over
}

func hasOpponentPiecesInHomeBoard(board *brd.Board, player brd.Checker) bool {
    if player == brd.Red {
        // Check White pieces in Reds's home board (points 1-6)
        for i := 1; i <= 6; i++ {
            if board.Pips[i].NumWhite() > 0 {
                return true
            }
        }
    } else {
        // Check Red pieces in White's home board (points 19-24)
        for i := 19; i <= 24; i++ {
            if board.Pips[i].NumRed() > 0 {
                return true
            }
        }
    }
    return false
}

//export GetEncodedStateForNN
func GetEncodedStateForNN(id C.uint64_t) *C.char {
    state := getBoardState(uint64(id))
    if state == nil {
        return C.CString("{}")
    }
    
    isTerminal, value := isTerminalState(uint64(id))
    encodedState := EncodedState{
        StateData:    encodeStateToTensor(state.board),
        Shape:        []int{30, 24},
        LegalMoves:   encodeLegalMoves(id),
        IsTerminal:   isTerminal,
        CurrentValue: value,
        GamePhase:    determineGamePhase(state.board),
    }
    
    data, err := json.Marshal(encodedState)
    if err != nil {
        return C.CString("{}")
    }
    return C.CString(string(data))
}

func determineGamePhase(board *brd.Board) int {
    // Game phases:
    // 0: Opening (first moves)
    // 1: Middle game
    // 2: Bearing in
    // 3: Bearing off
    
    // Count pieces in each player's home board
    whiteHome := 0
    redHome := 0
    
    for i := 1; i <= 6; i++ {
        whiteHome += board.Pips[i].NumWhite()
    }
    for i := 19; i <= 24; i++ {
        redHome += board.Pips[i].NumRed()
    }
    
    // Check if either player is bearing off
    if whiteHome == 15 || redHome == 15 {
        return 3
    }
    
    // Check if either player is bearing in
    if whiteHome >= 10 || redHome >= 10 {
        return 2
    }
    
    // Check for opening game - look at initial position pieces
    if board.Pips[1].NumWhite() >= 2 && board.Pips[12].NumWhite() >= 5 &&
       board.Pips[17].NumWhite() >= 3 && board.Pips[19].NumWhite() >= 5 {
        return 0
    }
    
    return 1 // Middle game
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

func main() {}