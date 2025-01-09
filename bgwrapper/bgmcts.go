package mcts

import (
    "math"
    "sync"
	"time"
    "github.com/chandler37/gobackgammon/brd"
)

// BatchEvaluator manages batched neural network evaluations
type BatchEvaluator struct {
    mutex       sync.Mutex
    batchSize   int
    maxWaitTime time.Duration
    queue       []*EvalRequest
    neural      NeuralNetEvaluator
    done        chan struct{}
}

type EvalRequest struct {
    Board    *brd.Board
    PolicyCh chan []float64
    ValueCh  chan float64
}

func NewBatchEvaluator(neural NeuralNetEvaluator, batchSize int) *BatchEvaluator {
    be := &BatchEvaluator{
        batchSize:   batchSize,
        maxWaitTime: 100 * time.Millisecond,
        queue:       make([]*EvalRequest, 0, batchSize),
        neural:      neural,
        done:        make(chan struct{}),
    }
    
    go be.processingLoop()
    return be
}

func (be *BatchEvaluator) Submit(board *brd.Board) ([]float64, float64) {
    policyCh := make(chan []float64, 1)
    valueCh := make(chan float64, 1)
    
    req := &EvalRequest{
        Board:    board,
        PolicyCh: policyCh,
        ValueCh:  valueCh,
    }
    
    be.mutex.Lock()
    be.queue = append(be.queue, req)
    shouldProcess := len(be.queue) >= be.batchSize
    be.mutex.Unlock()
    
    if shouldProcess {
        be.processQueue()
    }
    
    return <-policyCh, <-valueCh
}

func (be *BatchEvaluator) processingLoop() {
    ticker := time.NewTicker(be.maxWaitTime)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            be.mutex.Lock()
            if len(be.queue) > 0 {
                be.processQueue()
            }
            be.mutex.Unlock()
        case <-be.done:
            return
        }
    }
}

func (be *BatchEvaluator) processQueue() {
    be.mutex.Lock()
    if len(be.queue) == 0 {
        be.mutex.Unlock()
        return
    }
    
    // Get batch
    batchSize := min(len(be.queue), be.batchSize)
    batch := be.queue[:batchSize]
    be.queue = be.queue[batchSize:]
    be.mutex.Unlock()
    
    // Prepare batch for neural network
    boards := make([]*brd.Board, len(batch))
    for i, req := range batch {
        boards[i] = req.Board
    }
    
    // Evaluate batch
    policies, values := be.neural.EvaluateBatch(boards)
    
    // Distribute results
    for i, req := range batch {
        req.PolicyCh <- policies[i]
        req.ValueCh <- values[i]
    }
}

func (be *BatchEvaluator) Close() {
    close(be.done)
}

// Node represents a state in the MCTS tree
type Node struct {
    mutex       sync.RWMutex
    board       *brd.Board
    parent      *Node
    children    map[int]*Node  // move_id -> Node
    visits      int
    totalValue  float64
    priorProb   float64
    moveID      int           // move that led to this state
}

// MCTSConfig holds hyperparameters for MCTS
type MCTSConfig struct {
    NumSimulations   int
    CPuct           float64    // Exploration constant
    Temperature     float64    // For move selection
    DirichletAlpha  float64    // Noise parameter
    DirichletEps    float64    // Noise weight
}

// MCTS manages the tree search process
type MCTS struct {
    config  MCTSConfig
    root    *Node
    nn      *NeuralNetEvaluator
}

// NewMCTS creates a new MCTS instance
func NewMCTS(board *brd.Board, config MCTSConfig, nn *NeuralNetEvaluator) *MCTS {
    return &MCTS{
        config: config,
        root: &Node{
            board:    board.Clone(),
            children: make(map[int]*Node),
        },
        nn: nn,
    }
}

// RunSimulation performs one MCTS simulation
func (m *MCTS) RunSimulation() {
    node := m.root
    
    // Selection - travel down the tree
    for !node.board.IsTerminal() && len(node.children) > 0 {
        node = m.selectChild(node)
    }
    
    // Expansion and Evaluation
    if !node.board.IsTerminal() {
        moves := node.board.LegalContinuations()
        policyLogits, value := m.nn.Evaluate(node.board)
        
        // Add noise at root
        if node == m.root {
            policyLogits = m.addDirichletNoise(policyLogits)
        }
        
        // Create children
        node.mutex.Lock()
        for i, move := range moves {
            child := &Node{
                board:     move.Clone(),
                parent:   node,
                children: make(map[int]*Node),
                priorProb: policyLogits[i],
                moveID:    i,
            }
            node.children[i] = child
        }
        node.mutex.Unlock()
        
        // Backup
        m.backup(node, value)
    }
}

// selectChild uses PUCT formula for selection
func (m *MCTS) selectChild(node *Node) *Node {
    node.mutex.RLock()
    defer node.mutex.RUnlock()
    
    bestScore := -math.MaxFloat64
    var bestChild *Node
    
    parentVisits := float64(node.visits)
    
    for _, child := range node.children {
        child.mutex.RLock()
        
        // PUCT formula
        exploitation := -child.totalValue / float64(child.visits+1)
        exploration := m.config.CPuct * child.priorProb * 
                      math.Sqrt(parentVisits) / float64(child.visits+1)
        score := exploitation + exploration
        
        child.mutex.RUnlock()
        
        if score > bestScore {
            bestScore = score
            bestChild = child
        }
    }
    
    return bestChild
}

// backup updates statistics back up the tree
func (m *MCTS) backup(node *Node, value float64) {
    for node != nil {
        node.mutex.Lock()
        node.visits++
        node.totalValue += value
        node.mutex.Unlock()
        
        node = node.parent
        value = -value  // Flip value for opponent
    }
}

// GetActionProbs returns the action probabilities
func (m *MCTS) GetActionProbs() []float64 {
    visits := make([]float64, len(m.root.children))
    total := 0.0
    
    m.root.mutex.RLock()
    defer m.root.mutex.RUnlock()
    
    // Count visits
    for moveID, child := range m.root.children {
        child.mutex.RLock()
        visits[moveID] = math.Pow(float64(child.visits), 1.0/m.config.Temperature)
        child.mutex.RUnlock()
        total += visits[moveID]
    }
    
    // Normalize to probabilities
    for i := range visits {
        visits[i] /= total
    }
    
    return visits
}

// NeuralNetEvaluator interface for the Python neural network
type NeuralNetEvaluator interface {
    Evaluate(board *brd.Board) ([]float64, float64)  // Returns policy logits and value
}