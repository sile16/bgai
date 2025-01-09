package selfplay

import (
    "sync"
    "time"
    
    "github.com/chandler37/gobackgammon/brd"
    "github.com/chandler37/gobackgammon/mcts"
)

type GameRecord struct {
    States        [][]float32  `json:"states"`         // Encoded states
    PolicyTargets [][]float32  `json:"policy_targets"` // MCTS visit counts
    Values        []float32    `json:"values"`         // Game outcomes
    Scores        [2]int       `json:"scores"`         // Final score
}

type SelfPlayConfig struct {
    NumGames        int
    NumSimulations  int
    Temperature     float64
    NumWorkers      int
    BatchSize       int
}

type SelfPlayWorker struct {
    config     SelfPlayConfig
    evaluator  *mcts.BatchEvaluator
    gamesChan  chan *GameRecord
}

func NewSelfPlayWorker(config SelfPlayConfig, evaluator *mcts.BatchEvaluator) *SelfPlayWorker {
    return &SelfPlayWorker{
        config:    config,
        evaluator: evaluator,
        gamesChan: make(chan *GameRecord, config.NumGames),
    }
}

func (sp *SelfPlayWorker) GenerateGames() []*GameRecord {
    var wg sync.WaitGroup
    results := make([]*GameRecord, 0, sp.config.NumGames)
    
    // Start worker goroutines
    for i := 0; i < sp.config.NumWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            sp.workerLoop()
        }()
    }
    
    // Collect results
    go func() {
        for game := range sp.gamesChan {
            results = append(results, game)
        }
    }()
    
    wg.Wait()
    close(sp.gamesChan)
    
    return results
}

func (sp *SelfPlayWorker) workerLoop() {
    for {
        game := sp.playOneGame()
        sp.gamesChan <- game
    }
}

func (sp *SelfPlayWorker) playOneGame() *GameRecord {
    board := brd.New(false)
    record := &GameRecord{
        States:        make([][]float32, 0),
        PolicyTargets: make([][]float32, 0),
        Values:        make([]float32, 0),
    }
    
    tree := mcts.NewMCTS(board, mcts.MCTSConfig{
        NumSimulations: sp.config.NumSimulations,
        Temperature:    sp.config.Temperature,
    }, sp.evaluator)
    
    for !board.IsTerminal() {
        // Run MCTS simulations
        for i := 0; i < sp.config.NumSimulations; i++ {
            tree.RunSimulation()
        }
        
        // Record state and policy
        state := encodeState(board)
        policy := tree.GetActionProbs()
        
        record.States = append(record.States, state)
        record.PolicyTargets = append(record.PolicyTargets, policy)
        
        // Select and apply move
        moveIdx := sampleMove(policy, sp.config.Temperature)
        board = board.LegalContinuations()[moveIdx]
        
        // Update tree for next move
        tree.UpdateRoot(board)
    }
    
    // Calculate and record values
    outcome := calculateGameOutcome(board)
    for i := range record.States {
        if i%2 == 0 {
            record.Values = append(record.Values, outcome)
        } else {
            record.Values = append(record.Values, -outcome)
        }
    }
    
    return record
}